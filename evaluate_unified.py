"""
Évaluation EfficientDet unifié

Charge le modèle produit par train_unified.py et calcule les métriques
sur le jeu de test (mAP@50, mAP@50:95, Precision, Recall, F1 par classe).

Usage :
  python evaluate_unified.py
  python evaluate_unified.py --model runs/detect/train/efficientdet_unified_.../best_model.pth
"""

import os
import json
import argparse
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from pycocotools.coco import COCO
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

IOU_THRESHOLDS = [round(t, 2) for t in np.arange(0.5, 1.0, 0.05)]


# =============================================================================
# AUTODÉCOUVERTE
# =============================================================================

def find_unified_model_and_test_info(output_base=None):
    if output_base is None:
        output_base = os.getenv("OUTPUT_DIR", "./runs/detect/train")

    info_path = os.path.join(output_base, "model_info_unified.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        model_path = info.get("best_model", "")
        test_path  = info.get("test_info",  "")
        if model_path and os.path.exists(model_path) and test_path and os.path.exists(test_path):
            print(f"   model_info_unified.json -> {model_path}")
            return model_path, test_path, info.get("classes", [])

    for base in [output_base, "./runs/detect/train"]:
        if not os.path.exists(base):
            continue
        dirs = sorted(
            [d for d in os.listdir(base)
             if os.path.isdir(os.path.join(base, d)) and d.startswith("efficientdet_unified_")],
            reverse=True,
        )
        for d in dirs:
            test_path  = os.path.join(base, d, "test_info.json")
            model_path = os.path.join(base, d, "best_model.pth")
            if os.path.exists(model_path) and os.path.exists(test_path):
                print(f"   Trouve -> {model_path}")
                return model_path, test_path, []

    raise FileNotFoundError(
        "\n[ERREUR] Modele unifie EfficientDet introuvable.\n"
        "Lancez d'abord: python train_unified.py\n"
        "Ou: --model chemin/best_model.pth --test-info chemin/test_info.json"
    )


# =============================================================================
# MODÈLE
# =============================================================================

def load_model(model_path, device):
    print(f"   Chargement: {model_path}")
    ckpt        = torch.load(model_path, map_location=device)
    num_classes = ckpt.get("num_classes", 5)
    classes     = ckpt.get("classes", [])
    model_name  = ckpt.get("model_name", "tf_efficientdet_d0")
    image_size  = ckpt.get("image_size", 512)
    cat_mapping = ckpt.get("cat_mapping", {})

    config = get_efficientdet_config(model_name)
    config.update({'num_classes': num_classes, 'image_size': [image_size, image_size]})
    net = EfficientDet(config, pretrained_backbone=False)
    net.class_net = HeadNet(config, num_outputs=config.num_classes)

    state = ckpt.get('model_state_dict', ckpt)
    state = {k.replace('model.', '', 1) if k.startswith('model.') else k: v
             for k, v in state.items()}
    net.load_state_dict(state, strict=False)

    model = DetBenchPredict(net)
    model.to(device)
    model.eval()

    epoch = ckpt.get("epoch", "?")
    map50 = ckpt.get("map50", 0)
    print(f"   Epoch {epoch} | mAP@50 (val) = {map50:.4f} | {model_name} | {image_size}px")
    return model, classes, model_name, image_size, cat_mapping


# =============================================================================
# DATASET TEST
# =============================================================================

class EfficientDetTestDataset(Dataset):
    def __init__(self, images_dir, coco, image_ids, cat_mapping, image_size):
        self.images_dir  = images_dir
        self.coco        = coco
        self.image_ids   = image_ids
        self.cat_mapping = cat_mapping
        self.image_size  = image_size

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id   = self.image_ids[idx]
        img_info = self.coco.imgs[img_id]
        img_path = os.path.join(self.images_dir, img_info['file_name'])

        image    = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        image    = image.resize((self.image_size, self.image_size))
        scale_x  = self.image_size / orig_w
        scale_y  = self.image_size / orig_h

        tensor = TF.to_tensor(image)
        tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])

        anns   = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
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
            x1 = max(0, x * scale_x)
            y1 = max(0, y * scale_y)
            x2 = min(self.image_size, (x + w) * scale_x)
            y2 = min(self.image_size, (y + h) * scale_y)
            if x2 > x1 and y2 > y1:
                boxes.append([y1, x1, y2, x2])   # format effdet
                labels.append(class_id)

        boxes_t  = torch.tensor(boxes,  dtype=torch.float32) if boxes  else torch.zeros((0, 4), dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.int64)   if labels else torch.zeros((0,),   dtype=torch.int64)

        target = {
            'bbox':      boxes_t,
            'cls':       labels_t.float(),
            'img_size':  torch.tensor([self.image_size, self.image_size], dtype=torch.float32),
            'img_scale': torch.tensor(1.0),
            'image_id':  torch.tensor([img_id]),
        }
        return tensor, target


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
# MÉTRIQUES
# =============================================================================

class MetricsCalculator:
    def __init__(self, class_names, iou_thresholds=None):
        self.class_names    = class_names
        self.iou_thresholds = iou_thresholds or IOU_THRESHOLDS
        self.predictions    = []
        self.ground_truths  = []
        self.inference_times = []

    def add(self, preds, gts, inf_time=0):
        self.predictions.append(preds)
        self.ground_truths.append(gts)
        self.inference_times.append(inf_time)

    @staticmethod
    def _iou(b1, b2):
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        denom = a1 + a2 - inter
        return inter / denom if denom > 0 else 0

    def _ap_at_iou(self, cls_id, iou_thr):
        tps, fps, scores_list = [], [], []
        n_gt = sum((g['labels'] == cls_id).sum() for g in self.ground_truths)
        if n_gt == 0:
            return None
        for pred, gt in zip(self.predictions, self.ground_truths):
            mask_p  = pred['labels'] == cls_id
            mask_g  = gt['labels']   == cls_id
            p_boxes  = pred['boxes'][mask_p]
            p_scores = pred['scores'][mask_p]
            g_boxes  = gt['boxes'][mask_g]
            matched  = set()
            order    = np.argsort(-p_scores) if len(p_scores) else []
            for i in order:
                scores_list.append(p_scores[i])
                if len(g_boxes) == 0:
                    tps.append(0); fps.append(1); continue
                ious   = [self._iou(p_boxes[i], g) for g in g_boxes]
                best_j = int(np.argmax(ious))
                if ious[best_j] >= iou_thr and best_j not in matched:
                    matched.add(best_j); tps.append(1); fps.append(0)
                else:
                    tps.append(0); fps.append(1)
        if not scores_list:
            return 0.0
        order  = np.argsort(-np.array(scores_list))
        tp_cum = np.cumsum(np.array(tps)[order])
        fp_cum = np.cumsum(np.array(fps)[order])
        prec   = tp_cum / (tp_cum + fp_cum + 1e-10)
        rec    = tp_cum / (n_gt + 1e-10)
        return sum(np.max(prec[rec >= t]) if (rec >= t).any() else 0
                   for t in np.arange(0, 1.1, 0.1)) / 11

    def _precision_recall_f1(self, cls_id, iou_thr=0.5, score_thr=0.5):
        tp = fp = fn = 0
        for pred, gt in zip(self.predictions, self.ground_truths):
            mask_p  = (pred['labels'] == cls_id) & (pred['scores'] >= score_thr)
            mask_g  = gt['labels'] == cls_id
            p_boxes = pred['boxes'][mask_p]
            g_boxes = gt['boxes'][mask_g]
            matched = set()
            for pb in p_boxes:
                if len(g_boxes) == 0:
                    fp += 1; continue
                ious   = [self._iou(pb, gb) for gb in g_boxes]
                best_j = int(np.argmax(ious))
                if ious[best_j] >= iou_thr and best_j not in matched:
                    matched.add(best_j); tp += 1
                else:
                    fp += 1
            fn += len(g_boxes) - len(matched)
        prec = tp / (tp + fp + 1e-10)
        rec  = tp / (tp + fn + 1e-10)
        f1   = 2 * prec * rec / (prec + rec + 1e-10)
        return prec, rec, f1

    def compute(self):
        results = {}
        for cls_id, name in enumerate(self.class_names, start=1):
            aps = [self._ap_at_iou(cls_id, t) for t in self.iou_thresholds
                   if self._ap_at_iou(cls_id, t) is not None]
            ap50    = self._ap_at_iou(cls_id, 0.5) or 0.0
            ap50_95 = float(np.mean(aps)) if aps else 0.0
            prec, rec, f1 = self._precision_recall_f1(cls_id)
            results[name] = {
                'ap50': ap50, 'ap50_95': ap50_95,
                'precision': prec, 'recall': rec, 'f1': f1,
            }
        valid    = [v for v in results.values()]
        mAP50    = float(np.mean([v['ap50']    for v in valid])) if valid else 0.0
        mAP50_95 = float(np.mean([v['ap50_95'] for v in valid])) if valid else 0.0
        avg_prec = float(np.mean([v['precision'] for v in valid])) if valid else 0.0
        avg_rec  = float(np.mean([v['recall']    for v in valid])) if valid else 0.0
        avg_f1   = float(np.mean([v['f1']        for v in valid])) if valid else 0.0
        avg_inf  = float(np.mean(self.inference_times)) * 1000 if self.inference_times else 0.0
        return {
            'per_class': results,
            'mAP50': mAP50, 'mAP50_95': mAP50_95,
            'precision': avg_prec, 'recall': avg_rec, 'f1': avg_f1,
            'avg_inference_ms': avg_inf,
            'num_images': len(self.predictions),
        }


# =============================================================================
# INFÉRENCE TEST
# =============================================================================

@torch.no_grad()
def run_evaluation(model, test_loader, class_names, device, score_threshold=0.5):
    calc = MetricsCalculator(class_names)
    n    = len(test_loader.dataset)
    done = 0

    for images, targets in test_loader:
        images    = images.to(device)
        img_info  = {
            'img_scale': targets['img_scale'].to(device),
            'img_size':  targets['img_size'].to(device),
        }
        t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        detections = model(images, img_info)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inf_time = time.time() - t0

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

            # DetBenchPredict retourne déjà [x1,y1,x2,y2,score,class] (XYXY)
            preds = {
                'boxes':  det[:, :4]                        if len(det) else np.zeros((0, 4)),
                'scores': det[:, 4]                        if len(det) else np.zeros(0),
                'labels': det[:, 5].astype(int)             if len(det) else np.zeros(0, dtype=int),
            }
            gt_b = targets['bbox'][i].numpy()
            gt_l = targets['cls'][i].numpy().astype(int)
            if len(gt_b):
                gt_b = gt_b[:, [1, 0, 3, 2]]   # yx -> xy
            gts = {'boxes': gt_b, 'labels': gt_l}

            calc.add(preds, gts, inf_time / max(len(det_list), 1))
            done += 1
            print(f"   [{done}/{n}] {len(preds['boxes'])} detections | {inf_time*1000:.0f}ms")

    return calc.compute()


# =============================================================================
# VISUALISATION
# =============================================================================

def plot_per_class_ap(metrics, output_path, class_names):
    per_class = metrics['per_class']
    names  = [n for n in class_names if n in per_class]
    ap50s  = [per_class[n]['ap50'] for n in names]
    colors = ['green' if v >= 0.5 else 'orange' if v >= 0.3 else 'red' for v in ap50s]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, ap50s, color=colors)
    ax.set_ylim(0, 1)
    ax.set_title('AP@50 par classe — EfficientDet Unifie')
    ax.set_ylabel('AP@50')
    ax.axhline(0.5, color='green', linestyle='--', alpha=0.5, label='0.5')
    ax.axhline(0.3, color='orange', linestyle='--', alpha=0.5, label='0.3')
    for bar, val in zip(bars, ap50s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_global_metrics(metrics, output_path):
    labels = ['mAP@50', 'mAP@50:95', 'Precision', 'Recall', 'F1']
    values = [metrics['mAP50'], metrics['mAP50_95'],
              metrics['precision'], metrics['recall'], metrics['f1']]
    colors = ['#2196F3', '#9C27B0', '#4CAF50', '#FF9800', '#F44336']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1)
    ax.set_title('Metriques globales — EfficientDet Unifie')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    if not EFFDET_AVAILABLE:
        print("Installez d'abord: pip install effdet timm"); return

    parser = argparse.ArgumentParser(
        description="Evaluation EfficientDet unifie",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",           default=None)
    parser.add_argument("--test-info",       default=None)
    parser.add_argument("--output",          default=os.getenv("PREDICTIONS_DIR", "./evaluation_unified"))
    parser.add_argument("--score-threshold", type=float, default=0.5)
    args = parser.parse_args()

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_base = os.getenv("OUTPUT_DIR", "./runs/detect/train")

    print("=" * 65)
    print("   EVALUATION EfficientDet — Modele Unifie")
    print("=" * 65)
    print(f"   Device: {device}")

    if args.model and args.test_info:
        model_path     = args.model
        test_info_path = args.test_info
        extra_classes  = []
    else:
        try:
            model_path, test_info_path, extra_classes = find_unified_model_and_test_info(output_base)
        except FileNotFoundError as e:
            print(e); return

    if not os.path.exists(model_path):
        print(f"   Modele introuvable: {model_path}"); return
    if not os.path.exists(test_info_path):
        print(f"   test_info.json introuvable: {test_info_path}"); return

    with open(test_info_path) as f:
        test_info = json.load(f)

    model, classes, model_name, image_size, cat_mapping_raw = load_model(model_path, device)
    if not classes:
        classes = test_info.get('classes', extra_classes)
    if not classes:
        classes = ["__background__", "panneau_solaire", "batiment_peint",
                   "batiment_non_enduit", "batiment_enduit", "menuiserie_metallique"]

    class_names_no_bg = [c for c in classes if c != '__background__']
    cat_mapping       = {int(k): v for k, v in test_info['cat_mapping'].items()}
    images_dir        = test_info['images_dir']
    annotations_file  = test_info['annotations_file']
    test_image_ids    = test_info['test_image_ids']
    image_size        = test_info.get('image_size', image_size)

    print(f"\n   Classes: {classes}")
    print(f"   Test images: {len(test_image_ids)}")

    coco         = COCO(annotations_file)
    test_dataset = EfficientDetTestDataset(images_dir, coco, test_image_ids, cat_mapping, image_size)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    os.makedirs(args.output, exist_ok=True)
    print(f"\n   Evaluation en cours...")
    metrics = run_evaluation(model, test_loader, class_names_no_bg, device, args.score_threshold)

    print("\n" + "=" * 65)
    print(f"   mAP@50:    {metrics['mAP50']:.4f}")
    print(f"   mAP@50:95: {metrics['mAP50_95']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   F1:        {metrics['f1']:.4f}")
    print(f"   Inference: {metrics['avg_inference_ms']:.1f} ms/image")
    print("\n   Par classe:")
    for name, vals in metrics['per_class'].items():
        print(f"   {name:<30} AP50={vals['ap50']:.3f} | P={vals['precision']:.3f} | R={vals['recall']:.3f} | F1={vals['f1']:.3f}")

    metrics_path = os.path.join(args.output, "metrics_unified.json")
    metrics['model']     = model_path
    metrics['timestamp'] = datetime.now().isoformat()
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    plot_per_class_ap(metrics, os.path.join(args.output, "metrics_unified.png"), class_names_no_bg)
    plot_global_metrics(metrics, os.path.join(args.output, "metrics_unified_global.png"))

    report_path = os.path.join(args.output, "evaluation_report_unified.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"EfficientDet Unifie — Rapport d'evaluation\n{'='*50}\n")
        f.write(f"Date:       {datetime.now().isoformat()}\n")
        f.write(f"Modele:     {model_path}\n")
        f.write(f"Test set:   {len(test_image_ids)} images\n\n")
        f.write(f"mAP@50:     {metrics['mAP50']:.4f}\n")
        f.write(f"mAP@50:95:  {metrics['mAP50_95']:.4f}\n")
        f.write(f"Precision:  {metrics['precision']:.4f}\n")
        f.write(f"Recall:     {metrics['recall']:.4f}\n")
        f.write(f"F1:         {metrics['f1']:.4f}\n")
        f.write(f"Inference:  {metrics['avg_inference_ms']:.1f} ms/image\n\n")
        f.write("Par classe:\n")
        for name, vals in metrics['per_class'].items():
            f.write(f"  {name:<30} AP50={vals['ap50']:.3f} AP50:95={vals['ap50_95']:.3f}"
                    f" P={vals['precision']:.3f} R={vals['recall']:.3f} F1={vals['f1']:.3f}\n")

    print(f"\n   Resultats: {args.output}")


if __name__ == "__main__":
    main()
