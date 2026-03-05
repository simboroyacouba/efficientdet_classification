"""
Entraînement EfficientDet pour détection des toitures cadastrales
Dataset: Images aériennes annotées avec CVAT (format COCO)
Classes: Chargées depuis classes.yaml
Configuration: Chargée depuis .env

Backbone: EfficientNet-B0..B7 via effdet (timm/Ross Wightman)
Installation: pip install effdet timm
"""

import os
import json
import yaml
import shutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import time
import gc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
    print("❌ effdet non installé. Lancez: pip install effdet timm")


# =============================================================================
# CHARGEMENT DES CLASSES
# =============================================================================

def load_classes(yaml_path="classes.yaml"):
    """Charger les classes depuis YAML.
    EfficientDet (effdet) indexe les classes à partir de 1 (0 = background implicite).
    On exclut __background__ du comptage num_classes.
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Fichier introuvable: {yaml_path}")
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    classes = data.get('classes', [])
    if '__background__' not in classes:
        classes = ['__background__'] + classes
    print(f"📋 Classes chargées depuis {yaml_path}:")
    for i, c in enumerate(classes):
        print(f"   [{i}] {c}")
    return classes


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "images_dir":        os.getenv("DETECTION_DATASET_IMAGES_DIR", "../dataset1/images/default"),
    "annotations_file":  os.getenv("DETECTION_DATASET_ANNOTATIONS_FILE", "../dataset1/annotations/instances_default.json"),
    "output_dir":        os.getenv("OUTPUT_DIR", "./output"),
    "classes_file":      os.getenv("CLASSES_FILE", "classes.yaml"),
    "classes":           None,

    # Modèle EfficientDet
    # Valeurs: tf_efficientdet_d0 .. tf_efficientdet_d7
    # d0=leger/rapide, d4=bon equilibre, d7=lourd/précis
    "model_name":        os.getenv("EFFICIENTDET_MODEL", "tf_efficientdet_d0"),

    # Hyperparamètres
    "num_epochs":        int(os.getenv("NUM_EPOCHS", "25")),
    "batch_size":        int(os.getenv("BATCH_SIZE", "2")),
    "learning_rate":     float(os.getenv("LEARNING_RATE", "1e-4")),
    "weight_decay":      float(os.getenv("WEIGHT_DECAY", "1e-4")),
    "image_size":        int(os.getenv("IMAGE_SIZE", "512")),   # 512 standard pour d0/d1
    "train_split":       float(os.getenv("TRAIN_SPLIT", "0.70")),
    "val_split":         float(os.getenv("VAL_SPLIT", "0.20")),
    "test_split":        float(os.getenv("TEST_SPLIT", "0.10")),
    "save_every":        int(os.getenv("SAVE_EVERY", "5")),
    "score_threshold":   float(os.getenv("SCORE_THRESHOLD", "0.5")),
    "pretrained":        os.getenv("PRETRAINED", "true").lower() == "true",
    "grad_clip":         float(os.getenv("GRAD_CLIP", "1.0")),
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

    print(f"\n   📊 Split des IMAGES (total: {n_total}):")
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
    print("\n   📊 Distribution des classes (split 70/20/10):")
    print(f"   {'Classe':<30} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print(f"   {'-'*70}")
    for cat_id in coco.getCatIds():
        name  = coco.cats[cat_id]['name']
        train = stats['train'].get(cat_id, 0)
        val   = stats['val'].get(cat_id, 0)
        test  = stats['test'].get(cat_id, 0)
        total = train + val + test
        ok    = "⚠️" if val == 0 or test == 0 else "✅"
        print(f"   {name:<30} {train:>8} {val:>8} {test:>8} {total:>8} {ok}")
    print(f"   {'-'*70}")


# =============================================================================
# DATASET PYTORCH
# =============================================================================

class EfficientDetDataset(Dataset):
    """
    Dataset COCO pour EfficientDet (effdet).
    effdet attend les boxes au format [y1, x1, y2, x2] normalisé [0, image_size].
    Les labels commencent à 1 (0 est réservé au background).
    """

    def __init__(self, images_dir, annotations_file, image_ids, cat_mapping, image_size=512):
        self.images_dir  = images_dir
        self.coco        = COCO(annotations_file)
        self.image_ids   = image_ids
        self.cat_mapping = cat_mapping  # {coco_cat_id -> label 1-based}
        self.image_size  = image_size

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

        # Normalisation ImageNet
        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

        # Annotations — effdet attend [y1, x1, y2, x2]
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

        if boxes:
            boxes_t  = torch.tensor(boxes,  dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,),   dtype=torch.int64)

        target = {
            'bbox':       boxes_t,
            'cls':        labels_t.float(),
            'img_size':   torch.tensor([self.image_size, self.image_size], dtype=torch.float32),
            'img_scale':  torch.tensor(1.0),
            'image_id':   torch.tensor([img_id]),
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
# MODÈLE
# =============================================================================

def build_model(model_name, num_classes, image_size, pretrained=True):
    """
    Construire EfficientDet avec effdet.
    num_classes: nombre de classes SANS background.
    """
    config = get_efficientdet_config(model_name)
    config.update({'num_classes': num_classes, 'image_size': [image_size, image_size]})

    net = EfficientDet(config, pretrained_backbone=pretrained)

    # Remplacer la tête de classification
    net.class_net = HeadNet(
        config,
        num_outputs=config.num_classes,
    )

    # Wrapper entraînement (calcule la loss directement)
    model = DetBenchTrain(net, config)
    return model


def build_predict_model(model_name, num_classes, image_size, checkpoint_path, device):
    """Construire le modèle en mode inférence depuis un checkpoint."""
    config = get_efficientdet_config(model_name)
    config.update({'num_classes': num_classes, 'image_size': [image_size, image_size]})

    net = EfficientDet(config, pretrained_backbone=False)
    net.class_net = HeadNet(config, num_outputs=config.num_classes)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Charger les poids du réseau (sans le wrapper DetBenchTrain)
    state = checkpoint.get('model_state_dict', checkpoint)
    # Retirer le préfixe 'model.' si présent (ajouté par DetBenchTrain)
    state = {k.replace('model.', '', 1) if k.startswith('model.') else k: v
             for k, v in state.items()}
    net.load_state_dict(state, strict=False)

    model = DetBenchPredict(net)
    model.to(device)
    model.eval()
    return model


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


def yx_to_xy(boxes):
    """Convertir [y1,x1,y2,x2] -> [x1,y1,x2,y2]"""
    if len(boxes) == 0:
        return boxes
    return boxes[:, [1, 0, 3, 2]]


def compute_map_simple(predictions, ground_truths, class_names, iou_threshold=0.5):
    """Calculer mAP par classe (AP50)."""
    aps = {}
    for cls_id, name in enumerate(class_names, start=1):
        tps, fps, scores_list = [], [], []
        n_gt = sum((gt['labels'] == cls_id).sum() for gt in ground_truths)
        if n_gt == 0:
            continue

        for pred, gt in zip(predictions, ground_truths):
            mask_p = pred['labels'] == cls_id
            mask_g = gt['labels']  == cls_id
            p_boxes  = pred['boxes'][mask_p]
            p_scores = pred['scores'][mask_p]
            g_boxes  = gt['boxes'][mask_g]

            matched = set()
            order   = np.argsort(-p_scores) if len(p_scores) else []
            for i in order:
                scores_list.append(p_scores[i])
                if len(g_boxes) == 0:
                    tps.append(0); fps.append(1); continue
                ious    = [calculate_iou(p_boxes[i], g) for g in g_boxes]
                best_j  = int(np.argmax(ious))
                if ious[best_j] >= iou_threshold and best_j not in matched:
                    matched.add(best_j)
                    tps.append(1); fps.append(0)
                else:
                    tps.append(0); fps.append(1)

        if not scores_list:
            aps[name] = 0.0; continue

        order2   = np.argsort(-np.array(scores_list))
        tp_cum   = np.cumsum(np.array(tps)[order2])
        fp_cum   = np.cumsum(np.array(fps)[order2])
        prec     = tp_cum / (tp_cum + fp_cum + 1e-10)
        rec      = tp_cum / (n_gt + 1e-10)

        ap = 0
        for r_thresh in np.arange(0, 1.1, 0.1):
            mask = rec >= r_thresh
            ap  += np.max(prec[mask]) if mask.any() else 0
        aps[name] = ap / 11

    return aps


# =============================================================================
# ENTRAÎNEMENT
# =============================================================================

def train_one_epoch(model, optimizer, dataloader, device, grad_clip=1.0):
    model.train()
    total_loss = 0
    num_batches = 0

    for images, targets in dataloader:
        images = images.to(device)

        # Préparer les annotations pour effdet
        gt_boxes  = [t.to(device) for t in targets['bbox']]
        gt_labels = [t.to(device) for t in targets['cls']]

        # Construire le dict attendu par DetBenchTrain
        target_dict = {
            'bbox':      gt_boxes,
            'cls':       gt_labels,
            'img_size':  targets['img_size'].to(device),
            'img_scale': targets['img_scale'].to(device),
        }

        # Sauter les batches sans annotations
        if all(len(b) == 0 for b in gt_boxes):
            continue

        try:
            loss_dict = model(images, target_dict)
            # DetBenchTrain retourne un dict avec 'loss', 'class_loss', 'box_loss'
            loss = loss_dict['loss']

            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss  += loss.item()
            num_batches += 1

        except Exception as e:
            print(f"   ⚠️ Erreur batch: {e}")
            continue

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate_epoch(predict_model, dataloader, device, class_names, score_threshold=0.3):
    """
    Évaluation avec DetBenchPredict.
    Retourne mAP@50 moyen sur toutes les classes.
    """
    predict_model.eval()
    all_preds = []
    all_gts   = []

    for images, targets in dataloader:
        images     = images.to(device)
        img_sizes  = targets['img_size'].to(device)
        img_scales = targets['img_scale'].to(device)

        # DetBenchPredict: passer img_info comme dict (API effdet >= 0.3)
        img_info = {'img_scale': img_scales, 'img_size': img_sizes}
        detections = predict_model(images, img_info)

        # detections peut être un tensor [B, max_det, 6] ou une liste selon la version effdet
        if isinstance(detections, torch.Tensor):
            det_list = [detections[i] for i in range(detections.shape[0])]
        else:
            det_list = detections

        for i, det in enumerate(det_list):
            det  = det.detach().cpu().numpy() if hasattr(det, 'detach') else np.array(det)
            # Filtrer les détections vides (score == -1 ou padding)
            valid = det[:, 4] > 0
            det   = det[valid]
            keep  = det[:, 4] >= score_threshold
            det   = det[keep]

            # effdet retourne [x1, y1, x2, y2, score, class]
            all_preds.append({
                'boxes':  det[:, :4] if len(det) else np.zeros((0, 4)),
                'scores': det[:, 4]  if len(det) else np.zeros(0),
                'labels': det[:, 5].astype(int) if len(det) else np.zeros(0, dtype=int),
            })

            gt_b = targets['bbox'][i].numpy()
            gt_l = targets['cls'][i].numpy().astype(int)
            # Convertir [y1,x1,y2,x2] -> [x1,y1,x2,y2] pour le calcul IoU
            if len(gt_b):
                gt_b = gt_b[:, [1, 0, 3, 2]]
            all_gts.append({'boxes': gt_b, 'labels': gt_l})

    aps = compute_map_simple(all_preds, all_gts, class_names, iou_threshold=0.5)
    return float(np.mean(list(aps.values()))) if aps else 0.0


# =============================================================================
# MAIN
# =============================================================================

def train_efficientdet():
    if not EFFDET_AVAILABLE:
        print("❌ Installez d'abord: pip install effdet timm")
        return

    CONFIG["classes"] = load_classes(CONFIG["classes_file"])
    # Nombre de classes SANS background (effdet convention)
    class_names_no_bg = [c for c in CONFIG["classes"] if c != '__background__']
    num_classes = len(class_names_no_bg)

    # Adapter image_size au modèle si non surchargé
    recommended_size = EFFICIENTDET_IMAGE_SIZES.get(CONFIG["model_name"], 512)
    if CONFIG["image_size"] != recommended_size:
        print(f"   ℹ️  Taille recommandée pour {CONFIG['model_name']}: {recommended_size}px")

    print("=" * 70)
    print(f"   EfficientDet ({CONFIG['model_name']}) - Détection des Toitures")
    print("=" * 70)
    print(f"\n📋 CONFIG (.env)")
    print(f"   Images:       {CONFIG['images_dir']}")
    print(f"   Annotations:  {CONFIG['annotations_file']}")
    print(f"   Modèle:       {CONFIG['model_name']}")
    print(f"   Classes:      {num_classes} ({class_names_no_bg})")
    print(f"   Epochs:       {CONFIG['num_epochs']} | Batch: {CONFIG['batch_size']} | LR: {CONFIG['learning_rate']}")
    print(f"   Image size:   {CONFIG['image_size']}px")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device:       {device}")

    # Répertoire de sortie
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_dir   = os.path.join("runs", "detect", "train", f"efficientdet_{timestamp}")
    weights_dir = os.path.join(train_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Split dataset
    # -------------------------------------------------------------------------
    coco    = COCO(CONFIG["annotations_file"])
    cat_ids = coco.getCatIds()
    cat_mapping = {cat_id: idx + 1 for idx, cat_id in enumerate(cat_ids)}

    train_ids, val_ids, test_ids, split_stats = stratified_split(
        coco, CONFIG["train_split"], CONFIG["val_split"], CONFIG["test_split"], seed=42
    )
    print_split_stats(coco, split_stats)

    # Sauvegarder test_info.json
    test_info = {
        'test_image_ids':   test_ids,
        'cat_mapping':      {str(k): v for k, v in cat_mapping.items()},
        'images_dir':       os.path.abspath(CONFIG["images_dir"]),
        'annotations_file': os.path.abspath(CONFIG["annotations_file"]),
        'num_test_images':  len(test_ids),
        'classes':          CONFIG["classes"],
        'model_name':       CONFIG["model_name"],
        'image_size':       CONFIG["image_size"],
    }
    with open(os.path.join(train_dir, "test_info.json"), 'w') as f:
        json.dump(test_info, f, indent=2)

    # -------------------------------------------------------------------------
    # Datasets & DataLoaders
    # -------------------------------------------------------------------------
    train_dataset = EfficientDetDataset(CONFIG["images_dir"], CONFIG["annotations_file"],
                                        train_ids, cat_mapping, CONFIG["image_size"])
    val_dataset   = EfficientDetDataset(CONFIG["images_dir"], CONFIG["annotations_file"],
                                        val_ids,   cat_mapping, CONFIG["image_size"])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=1, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    print(f"\n   Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_ids)} images")

    # -------------------------------------------------------------------------
    # Modèle
    # -------------------------------------------------------------------------
    print(f"\n🧠 Chargement {CONFIG['model_name']} (pretrained={CONFIG['pretrained']})...")
    train_model = build_model(CONFIG["model_name"], num_classes,
                              CONFIG["image_size"], CONFIG["pretrained"])
    train_model.to(device)

    # Modèle inférence pour la validation (partagé, même réseau)
    predict_model = DetBenchPredict(train_model.model)
    predict_model.to(device)

    optimizer  = torch.optim.AdamW(train_model.parameters(),
                                   lr=CONFIG["learning_rate"],
                                   weight_decay=CONFIG["weight_decay"])
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["num_epochs"], eta_min=1e-6
    )

    # -------------------------------------------------------------------------
    # Boucle d'entraînement
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"   🚀 ENTRAÎNEMENT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    history = {
        'train_loss': [], 'val_map50': [], 'lr': [],
    }
    best_map50 = 0.0
    start_time = time.time()

    for epoch in range(1, CONFIG["num_epochs"] + 1):
        epoch_start = time.time()
        print(f"\n📅 Epoch [{epoch}/{CONFIG['num_epochs']}]")

        avg_loss = train_one_epoch(train_model, optimizer, train_loader, device, CONFIG["grad_clip"])
        val_map50 = evaluate_epoch(predict_model, val_loader, device,
                                   class_names_no_bg, CONFIG["score_threshold"])
        lr_scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(avg_loss)
        history['val_map50'].append(val_map50)
        history['lr'].append(current_lr)

        epoch_time = time.time() - epoch_start
        print(f"   Loss: {avg_loss:.4f} | mAP@50: {val_map50:.4f} | LR: {current_lr:.2e} | ⏱️ {format_time(epoch_time)}")

        # Sauvegarder le meilleur modèle
        if val_map50 > best_map50:
            best_map50 = val_map50
            torch.save({
                'epoch':            epoch,
                'model_state_dict': train_model.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'map50':            best_map50,
                'num_classes':      num_classes,
                'classes':          CONFIG["classes"],
                'cat_mapping':      cat_mapping,
                'model_name':       CONFIG["model_name"],
                'image_size':       CONFIG["image_size"],
            }, os.path.join(weights_dir, "best.pth"))
            print(f"   💾 Meilleur modèle sauvegardé (mAP@50: {best_map50:.4f})")

        # Sauvegarde périodique
        if epoch % CONFIG["save_every"] == 0 or epoch == CONFIG["num_epochs"]:
            torch.save({
                'epoch':            epoch,
                'model_state_dict': train_model.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'map50':            val_map50,
                'num_classes':      num_classes,
                'classes':          CONFIG["classes"],
                'cat_mapping':      cat_mapping,
                'model_name':       CONFIG["model_name"],
                'image_size':       CONFIG["image_size"],
            }, os.path.join(weights_dir, "last.pth"))

    total_time = time.time() - start_time

    # -------------------------------------------------------------------------
    # Copier les modèles
    # -------------------------------------------------------------------------
    print("\n📦 Copie des modèles...")
    for src, dst in [("best.pth", "best_model.pth"), ("last.pth", "final_model.pth")]:
        src_path = os.path.join(weights_dir, src)
        dst_path = os.path.join(train_dir, dst)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"   ✅ {dst} ({os.path.getsize(dst_path)/1024/1024:.1f} MB)")

    # -------------------------------------------------------------------------
    # Historique & graphiques
    # -------------------------------------------------------------------------
    history['time_stats'] = {
        'total_time':              total_time,
        'total_time_formatted':    format_time(total_time),
        'avg_epoch_time_formatted':format_time(total_time / CONFIG["num_epochs"]),
    }
    history['config']     = CONFIG
    history['best_map50'] = best_map50

    with open(os.path.join(train_dir, "history.json"), 'w') as f:
        json.dump(history, f, indent=2, default=str)

    if history['train_loss']:
        epochs = range(1, len(history['train_loss']) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(epochs, history['train_loss'], 'b-')
        axes[0].set_title('Loss (train)'); axes[0].grid(True, alpha=0.3)
        axes[1].plot(epochs, history['val_map50'], 'g-')
        axes[1].set_title('mAP@50 (validation)')
        axes[1].set_ylim(0, 1); axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(train_dir, 'training_curves.png'), dpi=150)
        plt.close()

    # Rapport
    with open(os.path.join(train_dir, "training_report.txt"), 'w', encoding='utf-8') as f:
        f.write(f"EfficientDet ({CONFIG['model_name']}) - Rapport\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Modèle: {CONFIG['model_name']}\n")
        f.write(f"Classes: {CONFIG['classes']}\n")
        f.write(f"Epochs: {CONFIG['num_epochs']} | Batch: {CONFIG['batch_size']}\n\n")
        f.write(f"Meilleur mAP@50: {best_map50:.4f}\n")
        f.write(f"Temps total: {format_time(total_time)}\n")
        f.write(f"Chemin: {train_dir}\n")

    print("\n" + "=" * 70)
    print("   🎉 TERMINÉ")
    print("=" * 70)
    print(f"   Meilleur mAP@50: {best_map50:.4f} ({best_map50*100:.2f}%)")
    print(f"   ⏱️  Temps: {format_time(total_time)}")
    print(f"   📁 Modèles: {train_dir}")
    print("=" * 70)

    return train_model, history


if __name__ == "__main__":
    train_efficientdet()
