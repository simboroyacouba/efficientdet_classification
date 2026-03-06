"""
Évaluation EfficientDet - Détection des toitures cadastrales
Évaluation sur le TEST SET (10% du dataset)
Configuration: .env + classes.yaml
"""

import os
import json
import yaml
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict
    from effdet.efficientdet import HeadNet
    EFFDET_AVAILABLE = True
except ImportError:
    EFFDET_AVAILABLE = False
    print("❌ effdet non installé. Lancez: pip install effdet timm")


# =============================================================================
# CLASSES
# =============================================================================

def load_classes(yaml_path="classes.yaml"):
    if not os.path.exists(yaml_path):
        return ['__background__', 'panneau_solaire', 'batiment_peint', 'batiment_non_enduit', 'batiment_enduit']
    with open(yaml_path, 'r', encoding='utf-8') as f:
        classes = yaml.safe_load(f).get('classes', [])
    if '__background__' not in classes:
        classes = ['__background__'] + classes
    return classes


# =============================================================================
# CONFIG
# =============================================================================

CONFIG = {
    "model_path":      os.getenv("MODEL_PATH", None),
    "output_dir":      os.getenv("EVALUATION_DIR", "./evaluation"),
    "classes_file":    os.getenv("CLASSES_FILE", "classes.yaml"),
    "classes":         None,
    "image_size":      int(os.getenv("IMAGE_SIZE", "512")),
    "score_threshold": float(os.getenv("SCORE_THRESHOLD", "0.3")),
    "iou_thresholds":  [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
}
CONFIG["classes"] = load_classes(CONFIG["classes_file"])


# =============================================================================
# CHARGEMENT DU MODÈLE
# =============================================================================

def load_model(model_path, device):
    print(f"🧠 Chargement du modèle: {model_path}")
    checkpoint  = torch.load(model_path, map_location=device)
    num_classes = checkpoint.get('num_classes')
    classes     = checkpoint.get('classes', CONFIG["classes"])
    cat_mapping = checkpoint.get('cat_mapping', {})
    model_name  = checkpoint.get('model_name', os.getenv("EFFICIENTDET_MODEL", "tf_efficientdet_d0"))
    image_size  = checkpoint.get('image_size', CONFIG["image_size"])

    if num_classes is None:
        num_classes = len([c for c in classes if c != '__background__'])

    config = get_efficientdet_config(model_name)
    config.update({'num_classes': num_classes, 'image_size': [image_size, image_size]})

    net = EfficientDet(config, pretrained_backbone=False)
    net.class_net = HeadNet(config, num_outputs=config.num_classes)

    state = checkpoint.get('model_state_dict', checkpoint)
    state = {k.replace('model.', '', 1) if k.startswith('model.') else k: v
             for k, v in state.items()}
    net.load_state_dict(state, strict=False)

    model = DetBenchPredict(net)
    model.to(device)
    model.eval()

    print(f"   Modèle:       {model_name}")
    print(f"   Epoch:        {checkpoint.get('epoch', '?')}")
    print(f"   mAP@50 (val): {checkpoint.get('map50', 0):.4f}")
    print(f"   Image size:   {image_size}px")

    return model, classes, cat_mapping, model_name, image_size


def find_model():
    path = CONFIG["model_path"]
    if path and os.path.exists(path):
        return path

    runs_base = os.path.join("runs", "detect", "train")
    if os.path.exists(runs_base):
        subdirs = sorted(
            [d for d in os.listdir(runs_base)
             if os.path.isdir(os.path.join(runs_base, d)) and 'efficientdet' in d],
            reverse=True
        )
        for subdir in subdirs:
            for fname in ["best_model.pth", "best.pth"]:
                candidate = os.path.join(runs_base, subdir, fname)
                if os.path.exists(candidate):
                    print(f"📁 Modèle trouvé: {candidate}")
                    return candidate

        subdirs = sorted(
            [d for d in os.listdir(runs_base) if os.path.isdir(os.path.join(runs_base, d))],
            reverse=True
        )
        for subdir in subdirs:
            for fname in ["best_model.pth", "best.pth"]:
                candidate = os.path.join(runs_base, subdir, fname)
                if os.path.exists(candidate):
                    print(f"📁 Modèle trouvé: {candidate}")
                    return candidate

    for root, dirs, files in os.walk("output"):
        for fname in ["best_model.pth", "best.pth"]:
            if fname in files:
                return os.path.join(root, fname)

    return None


def find_test_info(model_path):
    sibling = os.path.join(os.path.dirname(model_path), "test_info.json")
    if os.path.exists(sibling):
        return sibling

    runs_base = os.path.join("runs", "detect", "train")
    if os.path.exists(runs_base):
        subdirs = sorted(
            [d for d in os.listdir(runs_base) if os.path.isdir(os.path.join(runs_base, d))],
            reverse=True
        )
        for subdir in subdirs:
            candidate = os.path.join(runs_base, subdir, "test_info.json")
            if os.path.exists(candidate):
                return candidate

    for root, dirs, files in os.walk("output"):
        if "test_info.json" in files:
            return os.path.join(root, "test_info.json")

    return None


# =============================================================================
# DATASET TEST
# =============================================================================

class TestDataset(Dataset):
    def __init__(self, images_dir, annotations_file, image_ids, cat_mapping, image_size=512):
        self.images_dir  = images_dir
        self.coco        = COCO(annotations_file)
        self.image_ids   = image_ids
        self.cat_mapping = cat_mapping
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

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        boxes, labels = [], []
        for ann in anns:
            if ann.get('iscrowd', 0):
                continue
            class_id = self.cat_mapping.get(ann['category_id'])
            if class_id is None:
                continue
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                continue
            x1 = max(0, x * scale_x); y1 = max(0, y * scale_y)
            x2 = min(self.image_size, (x + w) * scale_x)
            y2 = min(self.image_size, (y + h) * scale_y)
            if x2 > x1 and y2 > y1:
                boxes.append([y1, x1, y2, x2])   # format effdet [y1,x1,y2,x2]
                labels.append(class_id)

        target = {
            'bbox':      torch.tensor(boxes,  dtype=torch.float32) if boxes  else torch.zeros((0, 4), dtype=torch.float32),
            'cls':       torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros((0,),   dtype=torch.float32),
            'img_size':  torch.tensor([self.image_size, self.image_size], dtype=torch.float32),
            'img_scale': torch.tensor(1.0),
            'image_id':  torch.tensor([img_id]),
        }
        return image_tensor, target


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    targets = {
        'bbox':      [b[1]['bbox']     for b in batch],
        'cls':       [b[1]['cls']      for b in batch],
        'img_size':  torch.stack([b[1]['img_size']  for b in batch]),
        'img_scale': torch.stack([b[1]['img_scale'] for b in batch]),
        'image_id':  torch.stack([b[1]['image_id']  for b in batch]),
    }
    return images, targets


# =============================================================================
# MÉTRIQUES
# =============================================================================

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    a2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    denom = a1 + a2 - inter
    return inter / denom if denom > 0 else 0


class MetricsCalculator:
    def __init__(self, class_names, iou_thresholds):
        self.class_names    = [c for c in class_names if c != '__background__']
        self.iou_thresholds = iou_thresholds
        self.tp  = defaultdict(lambda: defaultdict(int))
        self.fp  = defaultdict(lambda: defaultdict(int))
        self.fn  = defaultdict(lambda: defaultdict(int))
        self.all_ious = []

    def add_image(self, pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels):
        for iou_thresh in self.iou_thresholds:
            for class_id, name in enumerate(self.class_names, start=1):
                p_mask = pred_labels == class_id
                g_mask = gt_labels   == class_id
                p_b = pred_boxes[p_mask]; p_s = pred_scores[p_mask]
                g_b = gt_boxes[g_mask]

                if len(g_b) == 0 and len(p_b) == 0:
                    continue
                if len(g_b) == 0:
                    self.fp[name][iou_thresh] += len(p_b); continue
                if len(p_b) == 0:
                    self.fn[name][iou_thresh] += len(g_b); continue

                iou_mat = np.array([[calculate_iou(p, g) for g in g_b] for p in p_b])
                if iou_thresh == 0.5:
                    self.all_ious.extend(iou_mat.flatten().tolist())

                matched = set()
                for i in np.argsort(-p_s):
                    best_j = -1
                    for j in range(len(g_b)):
                        if j not in matched and iou_mat[i, j] >= iou_thresh:
                            if best_j < 0 or iou_mat[i, j] > iou_mat[i, best_j]:
                                best_j = j
                    if best_j >= 0:
                        matched.add(best_j)
                        self.tp[name][iou_thresh] += 1
                    else:
                        self.fp[name][iou_thresh] += 1
                self.fn[name][iou_thresh] += len(g_b) - len(matched)

    def compute(self):
        results = {'per_class': {}, 'overall': {}}

        for name in self.class_names:
            results['per_class'][name] = {}
            for t in self.iou_thresholds:
                tp = self.tp[name][t]; fp = self.fp[name][t]; fn = self.fn[name][t]
                p  = tp / (tp + fp) if tp + fp > 0 else 0
                r  = tp / (tp + fn) if tp + fn > 0 else 0
                results['per_class'][name][f'iou_{t}'] = {
                    'TP': tp, 'FP': fp, 'FN': fn,
                    'Precision': p, 'Recall': r,
                    'F1': 2*p*r/(p+r) if p+r > 0 else 0
                }

        for t in self.iou_thresholds:
            tp = sum(self.tp[n][t] for n in self.class_names)
            fp = sum(self.fp[n][t] for n in self.class_names)
            fn = sum(self.fn[n][t] for n in self.class_names)
            p  = tp / (tp + fp) if tp + fp > 0 else 0
            r  = tp / (tp + fn) if tp + fn > 0 else 0
            results['overall'][f'iou_{t}'] = {
                'TP': tp, 'FP': fp, 'FN': fn,
                'Precision': p, 'Recall': r,
                'F1': 2*p*r/(p+r) if p+r > 0 else 0
            }

        results['mAP50']    = results['overall']['iou_0.5']['Precision']
        results['mAP50_95'] = float(np.mean([results['overall'][f'iou_{t}']['Precision']
                                             for t in self.iou_thresholds]))
        results['mAP_per_class'] = {
            name: {
                'AP50':    results['per_class'][name]['iou_0.5']['Precision'],
                'AP50_95': float(np.mean([results['per_class'][name][f'iou_{t}']['Precision']
                                          for t in self.iou_thresholds]))
            }
            for name in self.class_names
        }
        if self.all_ious:
            results['iou_stats'] = {
                'mean':   float(np.mean(self.all_ious)),
                'median': float(np.median(self.all_ious))
            }
        return results


def plot_metrics(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    class_names = list(results['mAP_per_class'].keys())
    if not class_names:
        return
    x = np.arange(len(class_names))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(x-0.2, [results['mAP_per_class'][c]['AP50']    for c in class_names], 0.4, label='AP@50')
    axes[0].bar(x+0.2, [results['mAP_per_class'][c]['AP50_95'] for c in class_names], 0.4, label='AP@50:95')
    axes[0].set_xticks(x); axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0].set_title('AP par classe (TEST SET)'); axes[0].legend()
    axes[0].set_ylim(0, 1); axes[0].grid(True, alpha=0.3)

    w = 0.25
    axes[1].bar(x-w, [results['per_class'][c]['iou_0.5']['Precision'] for c in class_names], w, label='Precision')
    axes[1].bar(x,   [results['per_class'][c]['iou_0.5']['Recall']    for c in class_names], w, label='Recall')
    axes[1].bar(x+w, [results['per_class'][c]['iou_0.5']['F1']        for c in class_names], w, label='F1')
    axes[1].set_xticks(x); axes[1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1].set_title('P/R/F1 (IoU=0.5) - TEST SET'); axes[1].legend()
    axes[1].set_ylim(0, 1); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_test_set.png'), dpi=150)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    if not EFFDET_AVAILABLE:
        print("❌ Installez d'abord: pip install effdet timm")
        return

    print("=" * 70)
    print("   ÉVALUATION EfficientDet - TEST SET (10%)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")

    model_path = find_model()
    if model_path is None or not os.path.exists(model_path):
        print("❌ Modèle non trouvé! Spécifiez MODEL_PATH dans .env ou lancez train.py d'abord.")
        return

    model, classes, cat_mapping, model_name, image_size = load_model(model_path, device)

    test_info_path = find_test_info(model_path)
    if test_info_path is None:
        print("❌ test_info.json non trouvé! Lancez generate_test_info.py d'abord.")
        return

    print(f"   test_info: {test_info_path}")
    with open(test_info_path, 'r') as f:
        test_info = json.load(f)

    images_dir       = test_info['images_dir']
    annotations_file = test_info['annotations_file']
    test_image_ids   = test_info['test_image_ids']
    cat_mapping_int  = ({int(k): v for k, v in cat_mapping.items()}
                        if cat_mapping else
                        {int(k): v for k, v in test_info['cat_mapping'].items()})
    eval_image_size  = test_info.get('image_size', image_size)

    print(f"\n📋 Configuration:")
    print(f"   Modèle:   {model_path}")
    print(f"   Test set: {len(test_image_ids)} images")
    print(f"   Classes:  {classes}")
    print(f"\n   ℹ️  Indexation: effdet retourne 0-indexed → converti en 1-indexed pour les métriques")

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    test_dataset = TestDataset(images_dir, annotations_file, test_image_ids,
                               cat_mapping_int, eval_image_size)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    print("\n📊 Évaluation sur le TEST SET...")
    calc = MetricsCalculator(classes, CONFIG["iou_thresholds"])

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Test"):
            images     = images.to(device)
            img_sizes  = targets['img_size'].to(device)
            img_scales = targets['img_scale'].to(device)

            img_info   = {'img_scale': img_scales, 'img_size': img_sizes}
            detections = model(images, img_info)

            if isinstance(detections, torch.Tensor):
                det_list = [detections[i] for i in range(detections.shape[0])]
            else:
                det_list = detections

            for i, det in enumerate(det_list):
                det   = det.detach().cpu().numpy() if hasattr(det, 'detach') else np.array(det)
                valid = det[:, 4] > 0
                det   = det[valid]
                keep  = det[:, 4] >= CONFIG["score_threshold"]
                det   = det[keep]

                pred_boxes  = det[:, :4] if len(det) else np.zeros((0, 4))
                pred_scores = det[:, 4]  if len(det) else np.zeros(0)

                # effdet DetBenchPredict retourne les classes déjà 1-indexed
                # (identique au cat_mapping: 1=panneau_solaire, 2=batiment_peint...)
                pred_labels = det[:, 5].astype(int) if len(det) else np.zeros(0, dtype=int)

                # GT: convertir [y1,x1,y2,x2] -> [x1,y1,x2,y2]
                gt_b = targets['bbox'][i].numpy()
                gt_l = targets['cls'][i].numpy().astype(int)
                if len(gt_b):
                    gt_b = gt_b[:, [1, 0, 3, 2]]

                calc.add_image(pred_boxes, pred_labels, pred_scores, gt_b, gt_l)

    results = calc.compute()
    results['evaluation_info'] = {
        'dataset':    'TEST SET (10%)',
        'num_images': len(test_image_ids),
        'model_path': model_path,
        'model_name': model_name,
        'timestamp':  datetime.now().isoformat()
    }

    print("\n" + "=" * 70)
    print("   📊 RÉSULTATS SUR LE TEST SET")
    print("=" * 70)
    print(f"   Images testées: {len(test_image_ids)}")
    print(f"   mAP@50:    {results['mAP50']:.4f} ({results['mAP50']*100:.2f}%)")
    print(f"   mAP@50:95: {results['mAP50_95']:.4f}")
    print(f"   Precision: {results['overall']['iou_0.5']['Precision']:.4f}")
    print(f"   Recall:    {results['overall']['iou_0.5']['Recall']:.4f}")
    print(f"   F1-Score:  {results['overall']['iou_0.5']['F1']:.4f}")
    print("=" * 70)

    if results['mAP_per_class']:
        print("\n   Par classe (IoU=0.5):")
        for name in results['mAP_per_class']:
            m = results['per_class'][name]['iou_0.5']
            print(f"   {name:<30} P={m['Precision']:.3f} R={m['Recall']:.3f} F1={m['F1']:.3f}")

    with open(os.path.join(CONFIG["output_dir"], "metrics_test_set.json"), 'w') as f:
        json.dump(results, f, indent=2, default=float)

    plot_metrics(results, CONFIG["output_dir"])

    with open(os.path.join(CONFIG["output_dir"], "evaluation_report_test_set.txt"), 'w', encoding='utf-8') as f:
        f.write(f"ÉVALUATION EfficientDet ({model_name}) - TEST SET - {datetime.now()}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Images testées: {len(test_image_ids)}\n")
        f.write(f"Modèle: {model_path}\n\n")
        f.write(f"mAP@50: {results['mAP50']:.4f} ({results['mAP50']*100:.2f}%)\n")
        f.write(f"mAP@50:95: {results['mAP50_95']:.4f}\n")
        f.write(f"Precision: {results['overall']['iou_0.5']['Precision']:.4f}\n")
        f.write(f"Recall: {results['overall']['iou_0.5']['Recall']:.4f}\n")
        f.write(f"F1-Score: {results['overall']['iou_0.5']['F1']:.4f}\n")
        if results['mAP_per_class']:
            f.write("\n\nPAR CLASSE (IoU=0.5)\n" + "-" * 50 + "\n")
            for name in results['mAP_per_class']:
                m = results['per_class'][name]['iou_0.5']
                f.write(f"{name}: P={m['Precision']:.4f} R={m['Recall']:.4f} F1={m['F1']:.4f}\n")

    print(f"\n📁 Résultats sauvegardés: {CONFIG['output_dir']}")


if __name__ == "__main__":
    main()
