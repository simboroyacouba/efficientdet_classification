"""
Entraînement EfficientDet pour détection des toitures cadastrales
Dataset: Images aériennes annotées avec CVAT (format COCO)
Classes: Chargées depuis classes.yaml

Architecture duale :
  - Mode nadir   : classe panneau_solaire   (instances_nadir.json)
  - Mode oblique : classes batiment_*        (instances_oblique.json)
  - Mode dual    : nadir puis oblique séquentiellement

Backbone: EfficientNet-B0..B7 via effdet (timm/Ross Wightman)
Installation: pip install effdet timm

Usage :
  python train.py --mode simple
  python train.py --mode attention --cbam-reduction 16
  python train.py --mode optimize --n-trials 20
  python train.py --mode dual --aug panneau_solaire:3
  python train.py --mode dual --model-name tf_efficientdet_d1
"""

import os
import copy
import json
import yaml
import shutil
import argparse
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import time
import gc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms.functional as TF
from pycocotools.coco import COCO
import warnings
warnings.filterwarnings('ignore')

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict
    from effdet.efficientdet import HeadNet
    EFFDET_AVAILABLE = True
except ImportError:
    EFFDET_AVAILABLE = False
    print("effdet non installé. Lancez: pip install effdet timm")


# =============================================================================
# CLASSES PAR MODE
# =============================================================================

MODE_CLASSES = {
    "nadir":   ["panneau_solaire"],
    "oblique": [
        "batiment_peint", "batiment_non_enduit",
        "batiment_enduit", "menuiserie_metallique",
    ],
}

# Tailles d'image recommandées par variante
EFFICIENTDET_IMAGE_SIZES = {
    "tf_efficientdet_d0": 512,
    "tf_efficientdet_d1": 640,
    "tf_efficientdet_d2": 768,
    "tf_efficientdet_d3": 896,
    "tf_efficientdet_d4": 1024,
    "tf_efficientdet_d5": 1280,
    "tf_efficientdet_d6": 1280,
    "tf_efficientdet_d7": 1536,
}

OPTUNA_CONFIG = {
    "n_trials":           20,
    "n_epochs_per_trial": 3,
    "study_name":         "efficientdet_cadastral",
    "output_dir":         "./optuna_output",
}


# =============================================================================
# CONFIGURATION
# =============================================================================

def build_config(args):
    base_annotations = os.getenv(
        "DETECTION_DATASET_ANNOTATIONS_FILE",
        "../dataset1/annotations/instances_default.json",
    )
    ann_dir = os.path.dirname(os.path.abspath(base_annotations))

    annotations_file = args.annotations_file or base_annotations
    base_output      = os.getenv("OUTPUT_DIR", "./output")
    output_dir       = args.output_dir or base_output
    classes_file     = args.classes_file or os.getenv("CLASSES_FILE", "classes.yaml")

    return {
        "mode":             args.mode,
        "images_dir":       args.images_dir or os.getenv("DETECTION_DATASET_IMAGES_DIR", "../dataset1/images/default"),
        "annotations_file": annotations_file,
        "ann_dir":          ann_dir,
        "output_dir":       output_dir,
        "classes_file":     classes_file,
        "model_name":       args.model_name or os.getenv("EFFICIENTDET_MODEL", "tf_efficientdet_d0"),
        "num_epochs":       int(os.getenv("NUM_EPOCHS", "25")),
        "batch_size":       int(os.getenv("BATCH_SIZE", "2")),
        "learning_rate":    float(os.getenv("LEARNING_RATE", "1e-4")),
        "weight_decay":     float(os.getenv("WEIGHT_DECAY", "1e-4")),
        "image_size":       int(os.getenv("IMAGE_SIZE", "512")),
        "train_split":      float(os.getenv("TRAIN_SPLIT", "0.70")),
        "val_split":        float(os.getenv("VAL_SPLIT", "0.20")),
        "test_split":       float(os.getenv("TEST_SPLIT", "0.10")),
        "save_every":       int(os.getenv("SAVE_EVERY", "5")),
        "score_threshold":  float(os.getenv("SCORE_THRESHOLD", "0.5")),
        "pretrained":       os.getenv("PRETRAINED", "true").lower() == "true",
        "grad_clip":        float(os.getenv("GRAD_CLIP", "1.0")),
        "amp":              os.getenv("USE_AMP", "0") == "1",
        "classes":          None,
    }


def _build_sub_config(base_config, mode_label):
    """Construit un sous-config nadir ou oblique depuis le config de base."""
    ann_dir = base_config["ann_dir"]
    cfg = copy.deepcopy(base_config)
    cfg["mode"] = mode_label
    if mode_label == "nadir":
        cfg["annotations_file"] = os.path.join(ann_dir, "instances_nadir.json")
        cfg["output_dir"]       = os.path.join(base_config["output_dir"], "nadir")
    elif mode_label == "oblique":
        cfg["annotations_file"] = os.path.join(ann_dir, "instances_oblique.json")
        cfg["output_dir"]       = os.path.join(base_config["output_dir"], "oblique")
    return cfg


# =============================================================================
# CHARGEMENT DES CLASSES
# =============================================================================

def load_classes(yaml_path, mode_classes=None):
    """
    Charger et filtrer les classes.
    EfficientDet indexe les classes à partir de 1 (0 = background implicite).
    On inclut __background__ pour la cohérence avec les autres modèles.
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Fichier introuvable: {yaml_path}")
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    all_classes = [c for c in data.get('classes', []) if c != '__background__']
    if mode_classes is not None:
        filtered = [c for c in all_classes if c in mode_classes]
    else:
        filtered = all_classes
    classes = ['__background__'] + filtered
    print("Classes chargees:")
    for i, c in enumerate(classes):
        print(f"   [{i}] {c}")
    return classes


# =============================================================================
# PARSE AUG COEFFICIENTS
# =============================================================================

def parse_aug_coeffs(aug_args, classes):
    """
    Analyse --aug CLASS:COEFF avec correspondance partielle insensible à la casse.
    Retourne un dict {class_name: coeff} pour les classes connues.
    """
    coeffs = {}
    for entry in (aug_args or []):
        if ':' not in entry:
            print(f"   [WARN] --aug '{entry}' ignoré (format attendu CLASS:COEFF)")
            continue
        raw_name, raw_coeff = entry.rsplit(':', 1)
        try:
            coeff = int(raw_coeff)
        except ValueError:
            print(f"   [WARN] --aug '{entry}' ignoré (coefficient non entier)")
            continue
        if coeff < 1:
            print(f"   [WARN] --aug '{entry}' ignoré (coefficient < 1)")
            continue
        raw_lower = raw_name.lower()
        matched = [c for c in classes if raw_lower in c.lower()]
        if not matched:
            print(f"   [WARN] --aug '{raw_name}' ne correspond à aucune classe connue")
            continue
        if len(matched) > 1:
            print(f"   [WARN] --aug '{raw_name}' ambigu ({matched}), ignoré")
            continue
        coeffs[matched[0]] = coeff
    return coeffs


# =============================================================================
# UTILITAIRES
# =============================================================================

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds//60)}m {int(seconds%60)}s"
    else:
        return f"{int(seconds//3600)}h {int((seconds%3600)//60)}m"


def stratified_split(coco, train_split, val_split, test_split, seed=42):
    np.random.seed(seed)
    all_image_ids = [img_id for img_id in coco.imgs if coco.getAnnIds(imgIds=img_id)]
    np.random.shuffle(all_image_ids)
    n_total = len(all_image_ids)
    n_train = int(n_total * train_split)
    n_val   = int(n_total * val_split)
    n_test  = n_total - n_train - n_val
    if n_test < 1 and n_total > 2:
        n_test  = max(1, int(n_total * 0.10))
        n_train = n_total - n_val - n_test
    print(f"\n   Split des IMAGES (total: {n_total}):")
    print(f"      Train: {n_train} ({n_train/n_total*100:.1f}%)")
    print(f"      Val:   {n_val}   ({n_val/n_total*100:.1f}%)")
    print(f"      Test:  {n_test}  ({n_test/n_total*100:.1f}%)")
    train_ids = all_image_ids[:n_train]
    val_ids   = all_image_ids[n_train:n_train + n_val]
    test_ids  = all_image_ids[n_train + n_val:]
    stats = {'train': {}, 'val': {}, 'test': {}}
    for cat_id in coco.getCatIds():
        stats['train'][cat_id] = 0
        stats['val'][cat_id]   = 0
        stats['test'][cat_id]  = 0
    for img_id in train_ids:
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
            stats['train'][ann['category_id']] += 1
    for img_id in val_ids:
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
            stats['val'][ann['category_id']] += 1
    for img_id in test_ids:
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
            stats['test'][ann['category_id']] += 1
    return train_ids, val_ids, test_ids, stats


def print_split_stats(coco, stats):
    print("\n   Distribution des classes:")
    print(f"   {'Classe':<30} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print(f"   {'-'*70}")
    for cat_id in coco.getCatIds():
        name  = coco.cats[cat_id]['name']
        train = stats['train'].get(cat_id, 0)
        val   = stats['val'].get(cat_id, 0)
        test  = stats['test'].get(cat_id, 0)
        total = train + val + test
        warn  = " !" if val == 0 or test == 0 else ""
        print(f"   {name:<30} {train:>8} {val:>8} {test:>8} {total:>8}{warn}")
    print(f"   {'-'*70}")


# =============================================================================
# DATASET PYTORCH
# =============================================================================

class EfficientDetDataset(Dataset):
    """
    Dataset COCO pour EfficientDet (effdet).
    effdet attend les boxes au format [y1, x1, y2, x2] dans l'espace image_size.
    Les labels commencent à 1 (0 est réservé au background).
    """

    def __init__(self, images_dir, annotations_file, image_ids, cat_mapping,
                 image_size=512, augment=False):
        self.images_dir  = images_dir
        self.coco        = COCO(annotations_file)
        self.image_ids   = image_ids
        self.cat_mapping = cat_mapping
        self.image_size  = image_size
        self.augment     = augment

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id   = self.image_ids[idx]
        img_info = self.coco.imgs[img_id]
        img_path = os.path.join(self.images_dir, img_info['file_name'])

        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        image = image.resize((self.image_size, self.image_size))
        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h

        anns   = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        boxes  = []
        labels = []

        for ann in anns:
            if ann.get('iscrowd', 0):
                continue
            class_id = self.cat_mapping.get(ann['category_id'])
            if class_id is None:
                continue
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                continue
            x1 = max(0, x * scale_x)
            y1 = max(0, y * scale_y)
            x2 = min(self.image_size, (x + w) * scale_x)
            y2 = min(self.image_size, (y + h) * scale_y)
            if x2 > x1 and y2 > y1:
                boxes.append([y1, x1, y2, x2])   # format effdet: [y1,x1,y2,x2]
                labels.append(class_id)

        if self.augment:
            W = self.image_size
            if random.random() < 0.5:
                image = TF.hflip(image)
                boxes = [[y1, W - x2, y2, W - x1] for y1, x1, y2, x2 in boxes]
            if random.random() < 0.5:
                image = TF.adjust_brightness(image, random.uniform(0.7, 1.3))
            if random.random() < 0.5:
                image = TF.adjust_contrast(image, random.uniform(0.7, 1.3))
            if random.random() < 0.5:
                image = TF.adjust_saturation(image, random.uniform(0.7, 1.3))

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

        if boxes:
            boxes_t  = torch.tensor(boxes,  dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,),   dtype=torch.int64)

        target = {
            'bbox':      boxes_t,
            'cls':       labels_t.float(),
            'img_size':  torch.tensor([self.image_size, self.image_size], dtype=torch.float32),
            'img_scale': torch.tensor(1.0),
            'image_id':  torch.tensor([img_id]),
        }
        return image_tensor, target


def collate_fn(batch):
    images  = torch.stack([b[0] for b in batch])
    targets = {
        'bbox':      [b[1]['bbox']     for b in batch],
        'cls':       [b[1]['cls']      for b in batch],
        'img_size':  torch.stack([b[1]['img_size']  for b in batch]),
        'img_scale': torch.stack([b[1]['img_scale'] for b in batch]),
        'image_id':  torch.stack([b[1]['image_id']  for b in batch]),
    }
    return images, targets


# =============================================================================
# MODULES D'ATTENTION (CBAM) — injection post-BiFPN
# =============================================================================

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
        nn.init.zeros_(self.fc[-1].weight)

    def forward(self, x):
        return x * self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size=kernel_size,
                                 padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        nn.init.zeros_(self.conv.weight)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.sa(self.ca(x))


class PostBiFPNCBAM(nn.Module):
    """Wrapper post-BiFPN : applique CBAM sur chaque feature map de sortie du FPN."""
    def __init__(self, fpn, channels_list, cbam_reduction=16, cbam_kernel_size=7):
        super().__init__()
        self.fpn          = fpn
        self.cbam_modules = nn.ModuleList([
            CBAM(c, cbam_reduction, cbam_kernel_size) for c in channels_list
        ])

    def forward(self, x):
        features = self.fpn(x)
        return [
            self.cbam_modules[i](f) if i < len(self.cbam_modules) else f
            for i, f in enumerate(features)
        ]


def _probe_bifpn_channels(net, image_size):
    """Détecte les dimensions des feature maps de sortie du BiFPN via un forward factice."""
    net.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, image_size, image_size)
        x     = net.backbone(dummy)
        feats = net.fpn(x)
    return [f.shape[1] for f in feats]


# =============================================================================
# MODÈLE
# =============================================================================

def _build_efficientdet_net(model_name, num_classes, image_size, pretrained=True):
    """Construit l'EfficientDet net (pas encore wrappé dans DetBench)."""
    cfg = get_efficientdet_config(model_name)
    cfg.update({'num_classes': num_classes, 'image_size': [image_size, image_size]})
    net = EfficientDet(cfg, pretrained_backbone=pretrained)
    net.class_net = HeadNet(cfg, num_outputs=cfg.num_classes)
    return net, cfg


def get_model_simple(model_name, num_classes, image_size, pretrained=True):
    """EfficientDet standard sans attention, wrappé dans DetBenchTrain."""
    net, cfg = _build_efficientdet_net(model_name, num_classes, image_size, pretrained)
    return DetBenchTrain(net, cfg)


def get_model_attention(model_name, num_classes, image_size, pretrained=True,
                        cbam_reduction=16, cbam_kernel_size=7):
    """EfficientDet avec CBAM post-BiFPN."""
    net, cfg = _build_efficientdet_net(model_name, num_classes, image_size, pretrained)
    channels  = _probe_bifpn_channels(net, image_size)
    net.fpn   = PostBiFPNCBAM(net.fpn, channels, cbam_reduction, cbam_kernel_size)
    print(f"   PostBiFPNCBAM (CBAM r={cbam_reduction}, k={cbam_kernel_size}) "
          f"sur {len(channels)} niveaux BiFPN : {channels}")
    return DetBenchTrain(net, cfg)


def build_predict_model(model_name, num_classes, image_size, checkpoint_path, device):
    """Construire le modèle en mode inférence depuis un checkpoint."""
    cfg = get_efficientdet_config(model_name)
    cfg.update({'num_classes': num_classes, 'image_size': [image_size, image_size]})
    net = EfficientDet(cfg, pretrained_backbone=False)
    net.class_net = HeadNet(cfg, num_outputs=cfg.num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get('model_state_dict', checkpoint)
    state = {k.replace('model.', '', 1) if k.startswith('model.') else k: v
             for k, v in state.items()}
    net.load_state_dict(state, strict=False)
    model = DetBenchPredict(net)
    model.to(device)
    model.eval()
    return model


# =============================================================================
# PONDÉRATION PAR AUG_COEFFS (WeightedRandomSampler)
# =============================================================================

def compute_sample_weights(coco, image_ids, cat_mapping, classes, aug_coeffs):
    """
    Calcule les poids d'échantillonnage à partir des coefficients --aug.
    """
    class_idx_to_coeff = {
        classes.index(name): coeff
        for name, coeff in aug_coeffs.items()
        if name in classes
    }
    if not class_idx_to_coeff:
        return None

    sample_weights = []
    for img_id in image_ids:
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        w    = 1.0
        for ann in anns:
            cls_idx = cat_mapping.get(ann['category_id'])
            if cls_idx in class_idx_to_coeff:
                w = max(w, float(class_idx_to_coeff[cls_idx]))
        sample_weights.append(w)

    print("   Poids d'échantillonnage (aug_coeffs):")
    for name, coeff in aug_coeffs.items():
        if name in classes:
            print(f"      {name:<30} coeff={coeff}")

    return torch.tensor(sample_weights, dtype=torch.float32)


# =============================================================================
# MÉTRIQUES
# =============================================================================

def calculate_iou(box1, box2):
    """box format: [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    denom = a1 + a2 - inter
    return inter / denom if denom > 0 else 0


def compute_map_simple(predictions, ground_truths, class_names, iou_threshold=0.5):
    aps = {}
    for cls_id, name in enumerate(class_names, start=1):
        tps, fps, scores_list = [], [], []
        n_gt = sum((gt['labels'] == cls_id).sum() for gt in ground_truths)
        if n_gt == 0:
            continue
        for pred, gt in zip(predictions, ground_truths):
            mask_p  = pred['labels'] == cls_id
            mask_g  = gt['labels']   == cls_id
            p_boxes  = pred['boxes'][mask_p]
            p_scores = pred['scores'][mask_p]
            g_boxes  = gt['boxes'][mask_g]
            matched = set()
            order   = np.argsort(-p_scores) if len(p_scores) else []
            for i in order:
                scores_list.append(p_scores[i])
                if len(g_boxes) == 0:
                    tps.append(0); fps.append(1); continue
                ious   = [calculate_iou(p_boxes[i], g) for g in g_boxes]
                best_j = int(np.argmax(ious))
                if ious[best_j] >= iou_threshold and best_j not in matched:
                    matched.add(best_j); tps.append(1); fps.append(0)
                else:
                    tps.append(0); fps.append(1)
        if not scores_list:
            aps[name] = 0.0; continue
        order2  = np.argsort(-np.array(scores_list))
        tp_cum  = np.cumsum(np.array(tps)[order2])
        fp_cum  = np.cumsum(np.array(fps)[order2])
        prec    = tp_cum / (tp_cum + fp_cum + 1e-10)
        rec     = tp_cum / (n_gt + 1e-10)
        ap = sum(np.max(prec[rec >= t]) if (rec >= t).any() else 0
                 for t in np.arange(0, 1.1, 0.1)) / 11
        aps[name] = float(ap)
    return aps


# =============================================================================
# ENTRAÎNEMENT
# =============================================================================

def train_one_epoch(model, optimizer, dataloader, device, grad_clip=1.0, scaler=None):
    model.train()
    total_loss  = 0
    num_batches = 0
    use_amp = scaler is not None
    for images, targets in dataloader:
        images    = images.to(device)
        gt_boxes  = [t.to(device) for t in targets['bbox']]
        gt_labels = [t.to(device) for t in targets['cls']]
        target_dict = {
            'bbox':      gt_boxes,
            'cls':       gt_labels,
            'img_size':  targets['img_size'].to(device),
            'img_scale': targets['img_scale'].to(device),
        }
        if all(len(b) == 0 for b in gt_boxes):
            continue
        try:
            optimizer.zero_grad()
            if use_amp:
                with torch.cuda.amp.autocast():
                    loss_dict = model(images, target_dict)
                    loss = loss_dict['loss']
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_dict = model(images, target_dict)
                loss = loss_dict['loss']
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            total_loss  += loss.item()
            num_batches += 1
        except Exception as e:
            print(f"   Erreur batch: {e}")
            continue
    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate_epoch(predict_model, dataloader, device, class_names, score_threshold=0.3):
    predict_model.eval()
    all_preds = []
    all_gts   = []
    for images, targets in dataloader:
        images     = images.to(device)
        img_sizes  = targets['img_size'].to(device)
        img_scales = targets['img_scale'].to(device)
        img_info   = {'img_scale': img_scales, 'img_size': img_sizes}
        detections = predict_model(images, img_info)
        if isinstance(detections, torch.Tensor):
            det_list = [detections[i] for i in range(detections.shape[0])]
        else:
            det_list = detections
        for i, det in enumerate(det_list):
            det   = det.detach().cpu().numpy() if hasattr(det, 'detach') else np.array(det)
            valid = det[:, 4] > 0
            det   = det[valid]
            keep  = det[:, 4] >= score_threshold
            det   = det[keep]
            # effdet retourne [x1, y1, x2, y2, score, class]
            all_preds.append({
                'boxes':  det[:, :4] if len(det) else np.zeros((0, 4)),
                'scores': det[:, 4]  if len(det) else np.zeros(0),
                'labels': (det[:, 5].astype(int) + 1) if len(det) else np.zeros(0, dtype=int),
            })
            gt_b = targets['bbox'][i].numpy()
            gt_l = targets['cls'][i].numpy().astype(int)
            if len(gt_b):
                gt_b = gt_b[:, [1, 0, 3, 2]]   # yx -> xy
            all_gts.append({'boxes': gt_b, 'labels': gt_l})
    aps = compute_map_simple(all_preds, all_gts, class_names, iou_threshold=0.5)
    return float(np.mean(list(aps.values()))) if aps else 0.0


# =============================================================================
# BOUCLE D'ENTRAÎNEMENT PRINCIPALE
# =============================================================================

def _train_single(config, aug_coeffs, mode_label, training_mode,
                  cbam_reduction=16, cbam_kernel_size=7):
    """
    Lance un entraînement complet EfficientDet sur config.
    training_mode: 'simple' | 'attention'
    """
    classes           = config["classes"]
    class_names_no_bg = [c for c in classes if c != '__background__']
    num_classes       = len(class_names_no_bg)
    model_name        = config["model_name"]
    image_size        = config["image_size"]

    recommended = EFFICIENTDET_IMAGE_SIZES.get(model_name, 512)
    if image_size != recommended:
        print(f"   Note: taille recommandée pour {model_name}: {recommended}px")

    print("=" * 70)
    print(f"   EfficientDet ({model_name}) — Mode : {mode_label.upper()} [{training_mode}]")
    print("=" * 70)
    print(f"\n   Images:       {config['images_dir']}")
    print(f"   Annotations:  {config['annotations_file']}")
    print(f"   Classes:      {num_classes} ({class_names_no_bg})")
    print(f"   Epochs:       {config['num_epochs']} | Batch: {config['batch_size']} | LR: {config['learning_rate']}")
    print(f"   Image size:   {image_size}px")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device:       {device}")

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_dir   = os.path.join(config["output_dir"], f"efficientdet_{mode_label}_{timestamp}")
    weights_dir = os.path.join(train_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    coco      = COCO(config["annotations_file"])
    coco_cats = {cat['id']: cat['name'] for cat in coco.loadCats(coco.getCatIds())}
    cat_mapping = {}
    for cat_id, cat_name in coco_cats.items():
        if cat_name in classes:
            cat_mapping[cat_id] = classes.index(cat_name)
        else:
            print(f"   Catégorie COCO ignorée: '{cat_name}' (id={cat_id})")

    print(f"\n   cat_mapping: {[(coco_cats[k], v) for k, v in cat_mapping.items()]}")

    train_ids, val_ids, test_ids, split_stats = stratified_split(
        coco, config["train_split"], config["val_split"], config["test_split"], seed=42
    )
    print_split_stats(coco, split_stats)

    test_info_path = os.path.join(train_dir, "test_info.json")
    with open(test_info_path, 'w') as f:
        json.dump({
            'test_image_ids':   test_ids,
            'cat_mapping':      {str(k): v for k, v in cat_mapping.items()},
            'images_dir':       os.path.abspath(config["images_dir"]),
            'annotations_file': os.path.abspath(config["annotations_file"]),
            'num_test_images':  len(test_ids),
            'classes':          classes,
            'model_name':       model_name,
            'image_size':       image_size,
            'mode':             mode_label,
        }, f, indent=2)

    train_dataset = EfficientDetDataset(config["images_dir"], config["annotations_file"],
                                        train_ids, cat_mapping, image_size, augment=True)
    val_dataset   = EfficientDetDataset(config["images_dir"], config["annotations_file"],
                                        val_ids,   cat_mapping, image_size, augment=False)

    sample_weights = compute_sample_weights(coco, train_ids, cat_mapping,
                                            classes, aug_coeffs)
    if sample_weights is not None:
        sampler      = WeightedRandomSampler(sample_weights,
                                             num_samples=len(sample_weights),
                                             replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"],
                                  sampler=sampler, collate_fn=collate_fn, num_workers=0)
    else:
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"],
                                  shuffle=True, collate_fn=collate_fn, num_workers=0)

    val_loader = DataLoader(val_dataset, batch_size=1,
                            shuffle=False, collate_fn=collate_fn, num_workers=0)
    print(f"\n   Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_ids)}")

    print(f"\n   Chargement {model_name} (pretrained={config['pretrained']})...")
    if training_mode == "attention":
        train_model = get_model_attention(model_name, num_classes, image_size,
                                          config["pretrained"], cbam_reduction, cbam_kernel_size)
    else:
        train_model = get_model_simple(model_name, num_classes, image_size, config["pretrained"])
    train_model.to(device)
    predict_model = DetBenchPredict(train_model.model)
    predict_model.to(device)

    optimizer    = torch.optim.AdamW(train_model.parameters(),
                                     lr=config["learning_rate"],
                                     weight_decay=config["weight_decay"])
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["num_epochs"], eta_min=1e-6
    )

    use_amp = config.get("amp", False) and device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None

    print("\n" + "=" * 70)
    print(f"   ENTRAÎNEMENT — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if use_amp:
        print("   Mixed precision (AMP fp16) : ACTIVE")
    print("=" * 70)

    history    = {'train_loss': [], 'val_map50': [], 'lr': []}
    best_map50 = 0.0
    best_path  = os.path.join(weights_dir, "best.pth")
    start_time = time.time()

    for epoch in range(1, config["num_epochs"] + 1):
        epoch_start = time.time()
        print(f"\nEpoch [{epoch}/{config['num_epochs']}]")

        avg_loss  = train_one_epoch(train_model, optimizer, train_loader, device,
                                    config["grad_clip"], scaler=scaler)
        val_map50 = evaluate_epoch(predict_model, val_loader, device,
                                   class_names_no_bg, config["score_threshold"])
        lr_scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(avg_loss)
        history['val_map50'].append(val_map50)
        history['lr'].append(current_lr)

        print(f"   Loss: {avg_loss:.4f} | mAP@50: {val_map50:.4f} | LR: {current_lr:.2e} | {format_time(time.time()-epoch_start)}")

        if val_map50 > best_map50:
            best_map50 = val_map50
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     train_model.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'map50':                best_map50,
                'num_classes':          num_classes,
                'classes':              classes,
                'cat_mapping':          cat_mapping,
                'model_name':           model_name,
                'image_size':           image_size,
                'mode':                 mode_label,
                'training_mode':        training_mode,
            }, best_path)
            print(f"   Meilleur modèle sauvegardé (mAP@50: {best_map50:.4f})")

        if epoch % config["save_every"] == 0 or epoch == config["num_epochs"]:
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     train_model.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'map50':                val_map50,
                'num_classes':          num_classes,
                'classes':              classes,
                'cat_mapping':          cat_mapping,
                'model_name':           model_name,
                'image_size':           image_size,
                'mode':                 mode_label,
                'training_mode':        training_mode,
            }, os.path.join(weights_dir, "last.pth"))

    total_time = time.time() - start_time

    best_model_path = os.path.join(train_dir, "best_model.pth")
    for src, dst in [("best.pth", "best_model.pth"), ("last.pth", "final_model.pth")]:
        src_path = os.path.join(weights_dir, src)
        dst_path = os.path.join(train_dir, dst)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"   {dst} ({os.path.getsize(dst_path)/1024/1024:.1f} MB)")

    model_info = {
        "model":         f"EfficientDet_{mode_label}",
        "model_name":    model_name,
        "mode":          mode_label,
        "training_mode": training_mode,
        "best_model":    os.path.abspath(best_model_path),
        "train_dir":     os.path.abspath(train_dir),
        "test_info":     os.path.abspath(test_info_path),
        "classes":       classes,
        "num_classes":   num_classes,
        "image_size":    image_size,
        "best_map50":    best_map50,
        "timestamp":     timestamp,
    }
    info_path = os.path.join(config["output_dir"], f"model_info_{mode_label}.json")
    os.makedirs(config["output_dir"], exist_ok=True)
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    print(f"\n   model_info_{mode_label}.json -> {info_path}")

    history['best_map50'] = best_map50
    history['config']     = {k: str(v) for k, v in config.items()}
    with open(os.path.join(train_dir, "history.json"), 'w') as f:
        json.dump(history, f, indent=2, default=str)

    if history['train_loss']:
        epochs_r = range(1, len(history['train_loss']) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(epochs_r, history['train_loss'], 'b-')
        axes[0].set_title('Loss (train)'); axes[0].grid(True, alpha=0.3)
        axes[1].plot(epochs_r, history['val_map50'], 'g-')
        axes[1].set_title('mAP@50 (validation)')
        axes[1].set_ylim(0, 1); axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(train_dir, 'training_curves.png'), dpi=150)
        plt.close()

    with open(os.path.join(train_dir, "training_report.txt"), 'w', encoding='utf-8') as f:
        f.write(f"EfficientDet ({model_name}) — Mode {mode_label} [{training_mode}]\n{'='*50}\n\n")
        f.write(f"Mode:            {mode_label}\n")
        f.write(f"Training mode:   {training_mode}\n")
        f.write(f"Modèle:          {model_name}\n")
        f.write(f"Classes:         {classes}\n")
        f.write(f"Epochs:          {config['num_epochs']} | Batch: {config['batch_size']}\n\n")
        f.write(f"Meilleur mAP@50: {best_map50:.4f}\n")
        f.write(f"Temps total:     {format_time(total_time)}\n")
        f.write(f"Chemin:          {train_dir}\n")

    print("\n" + "=" * 70)
    print(f"   TERMINÉ — Mode {mode_label.upper()} [{training_mode}]")
    print("=" * 70)
    print(f"   Meilleur mAP@50: {best_map50:.4f} ({best_map50*100:.2f}%)")
    print(f"   Temps: {format_time(total_time)}")
    print(f"   Modèle: {best_model_path}")
    print("=" * 70)

    return train_model, history


# =============================================================================
# OPTIMISATION OPTUNA
# =============================================================================

def _run_optimization(config, aug_coeffs, mode_label, cbam_reduction, cbam_kernel_size,
                      n_trials, n_epochs_per_trial):
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner

    classes           = config["classes"]
    class_names_no_bg = [c for c in classes if c != '__background__']
    num_classes       = len(class_names_no_bg)
    model_name        = config["model_name"]
    image_size        = config["image_size"]
    device            = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    coco      = COCO(config["annotations_file"])
    coco_cats = {cat['id']: cat['name'] for cat in coco.loadCats(coco.getCatIds())}
    cat_mapping = {
        cat_id: classes.index(cat_name)
        for cat_id, cat_name in coco_cats.items()
        if cat_name in classes
    }

    train_ids, val_ids, _, _ = stratified_split(
        coco, config["train_split"], config["val_split"], config["test_split"], seed=42
    )

    sample_weights = compute_sample_weights(coco, train_ids, cat_mapping,
                                            classes, aug_coeffs)

    # Charge le modèle de référence une seule fois
    print(f"   Chargement modèle de référence pour Optuna...")
    ref_model     = get_model_simple(model_name, num_classes, image_size, config["pretrained"])
    initial_state = copy.deepcopy(ref_model.model.state_dict())
    del ref_model
    gc.collect()

    def objective(trial):
        lr           = trial.suggest_float("lr",           1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        use_cbam     = trial.suggest_categorical("use_cbam", [True, False])

        if use_cbam:
            reduction   = trial.suggest_categorical("cbam_reduction",   [8, 16, 32])
            kernel_size = trial.suggest_categorical("cbam_kernel_size",  [3, 5, 7])
            train_model = get_model_attention(model_name, num_classes, image_size,
                                              pretrained=False,
                                              cbam_reduction=reduction,
                                              cbam_kernel_size=kernel_size)
        else:
            train_model = get_model_simple(model_name, num_classes, image_size, pretrained=False)
            train_model.model.load_state_dict(copy.deepcopy(initial_state), strict=False)
        train_model.to(device)
        predict_model = DetBenchPredict(train_model.model)
        predict_model.to(device)

        train_dataset = EfficientDetDataset(config["images_dir"], config["annotations_file"],
                                            train_ids, cat_mapping, image_size, augment=True)
        val_dataset   = EfficientDetDataset(config["images_dir"], config["annotations_file"],
                                            val_ids,   cat_mapping, image_size, augment=False)

        if sample_weights is not None:
            sampler      = WeightedRandomSampler(sample_weights,
                                                 num_samples=len(sample_weights),
                                                 replacement=True)
            train_loader = DataLoader(train_dataset, batch_size=config["batch_size"],
                                      sampler=sampler, collate_fn=collate_fn, num_workers=0)
        else:
            train_loader = DataLoader(train_dataset, batch_size=config["batch_size"],
                                      shuffle=True, collate_fn=collate_fn, num_workers=0)

        val_loader = DataLoader(val_dataset, batch_size=1,
                                shuffle=False, collate_fn=collate_fn, num_workers=0)

        optimizer = torch.optim.AdamW(train_model.parameters(),
                                      lr=lr, weight_decay=weight_decay)

        best_val_map = 0.0
        for ep in range(n_epochs_per_trial):
            train_one_epoch(train_model, optimizer, train_loader, device, config["grad_clip"])
            val_map50 = evaluate_epoch(predict_model, val_loader, device,
                                       class_names_no_bg, config["score_threshold"])
            best_val_map = max(best_val_map, val_map50)
            trial.report(val_map50, ep)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        del train_model, predict_model
        gc.collect()
        return best_val_map

    output_dir = OPTUNA_CONFIG["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(),
        pruner=MedianPruner(),
        study_name=f"{OPTUNA_CONFIG['study_name']}_{mode_label}",
    )
    print(f"\n   Optuna: {n_trials} trials × {n_epochs_per_trial} epochs chacun")
    study.optimize(objective, n_trials=n_trials)

    best = study.best_params
    print(f"\n   Meilleurs hyperparamètres: {best}")

    with open(os.path.join(output_dir, f"optuna_best_{mode_label}.json"), 'w') as f:
        json.dump({"best_params": best, "best_value": study.best_value}, f, indent=2)

    return best


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EfficientDet — détection des toitures cadastrales",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode", default="simple",
        choices=["simple", "attention", "optimize", "dual",
                 "nadir", "oblique", "all"],  # nadir/oblique/all: alias dépréciés
        help="Mode d'entraînement",
    )
    parser.add_argument("--aug",              nargs="*", default=[],
                        metavar="CLASS:COEFF",
                        help="Coefficients d'oversampling par classe (ex: panneau_solaire:3)")
    parser.add_argument("--cbam-reduction",   type=int, default=16,
                        help="Facteur de réduction CBAM (ChannelAttention)")
    parser.add_argument("--cbam-kernel-size", type=int, default=7,
                        help="Taille du kernel CBAM (SpatialAttention)")
    parser.add_argument("--n-trials",         type=int, default=OPTUNA_CONFIG["n_trials"],
                        help="Nombre de trials Optuna")
    parser.add_argument("--n-epochs-trial",   type=int, default=OPTUNA_CONFIG["n_epochs_per_trial"],
                        help="Epochs par trial Optuna")
    parser.add_argument("--amp",              action="store_true",
                        default=os.getenv("USE_AMP", "0") == "1",
                        help="Mixed precision fp16 (GPU uniquement, recommandé pour D3+)")
    parser.add_argument("--images-dir",       default=None)
    parser.add_argument("--annotations-file", default=None)
    parser.add_argument("--output-dir",       default=None)
    parser.add_argument("--classes-file",     default=None)
    parser.add_argument("--model-name",       default=None,
                        help="Variante EfficientDet (tf_efficientdet_d0 .. d7)")
    args = parser.parse_args()

    if not EFFDET_AVAILABLE:
        print("Installez d'abord: pip install effdet timm"); return

    # Alias dépréciés
    mode = args.mode
    if mode == "nadir":
        print("[DEPRECATED] --mode nadir est déprécié. Utilisez --mode dual ou --mode simple avec --annotations-file instances_nadir.json")
        args.mode = "simple"
    elif mode == "oblique":
        print("[DEPRECATED] --mode oblique est déprécié. Utilisez --mode dual ou --mode simple avec --annotations-file instances_oblique.json")
        args.mode = "simple"
    elif mode == "all":
        print("[DEPRECATED] --mode all est déprécié. Utilisez --mode simple.")
        args.mode = "simple"
    mode = args.mode

    config = build_config(args)
    config["amp"] = args.amp or config.get("amp", False)

    if mode == "dual":
        nadir_cfg            = _build_sub_config(config, "nadir")
        nadir_cfg["classes"] = load_classes(config["classes_file"], MODE_CLASSES["nadir"])
        aug_coeffs = parse_aug_coeffs(args.aug, nadir_cfg["classes"])
        print(f"\n[DUAL] Phase 1 — Nadir ({[c for c in nadir_cfg['classes'] if c != '__background__']})")
        _train_single(nadir_cfg, aug_coeffs, "nadir", "simple",
                      args.cbam_reduction, args.cbam_kernel_size)

        oblique_cfg            = _build_sub_config(config, "oblique")
        oblique_cfg["classes"] = load_classes(config["classes_file"], MODE_CLASSES["oblique"])
        aug_coeffs = parse_aug_coeffs(args.aug, oblique_cfg["classes"])
        print(f"\n[DUAL] Phase 2 — Oblique ({[c for c in oblique_cfg['classes'] if c != '__background__']})")
        _train_single(oblique_cfg, aug_coeffs, "oblique", "simple",
                      args.cbam_reduction, args.cbam_kernel_size)

    elif mode == "optimize":
        all_mode_classes  = MODE_CLASSES["nadir"] + MODE_CLASSES["oblique"]
        config["classes"] = load_classes(config["classes_file"], all_mode_classes)
        aug_coeffs        = parse_aug_coeffs(args.aug, config["classes"])
        best_params = _run_optimization(
            config, aug_coeffs, "all",
            args.cbam_reduction, args.cbam_kernel_size,
            args.n_trials, args.n_epochs_trial,
        )
        use_cbam = best_params.get("use_cbam", False)
        config["learning_rate"] = best_params.get("lr",           config["learning_rate"])
        config["weight_decay"]  = best_params.get("weight_decay", config["weight_decay"])
        cbam_r = best_params.get("cbam_reduction",   args.cbam_reduction)
        cbam_k = best_params.get("cbam_kernel_size",  args.cbam_kernel_size)
        training_mode = "attention" if use_cbam else "simple"
        _train_single(config, aug_coeffs, "all", training_mode, cbam_r, cbam_k)

    elif mode == "attention":
        all_mode_classes  = MODE_CLASSES["nadir"] + MODE_CLASSES["oblique"]
        config["classes"] = load_classes(config["classes_file"], all_mode_classes)
        aug_coeffs        = parse_aug_coeffs(args.aug, config["classes"])
        _train_single(config, aug_coeffs, "all", "attention",
                      args.cbam_reduction, args.cbam_kernel_size)

    else:  # simple (+ anciens alias redirigés)
        all_mode_classes  = MODE_CLASSES["nadir"] + MODE_CLASSES["oblique"]
        config["classes"] = load_classes(config["classes_file"], all_mode_classes)
        aug_coeffs        = parse_aug_coeffs(args.aug, config["classes"])
        _train_single(config, aug_coeffs, "all", "simple",
                      args.cbam_reduction, args.cbam_kernel_size)


if __name__ == "__main__":
    main()
