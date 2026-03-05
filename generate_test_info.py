"""
Régénère test_info.json sans relancer l'entraînement.
Utilise exactement les mêmes paramètres que train.py (seed=42, split 70/20/10).

Usage:
    python generate_test_info.py
    python generate_test_info.py --output runs/detect/train/efficientdet_XXXXXXXX_XXXXXX
"""

import os
import json
import yaml
import argparse
import numpy as np
from pycocotools.coco import COCO

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def load_classes(yaml_path="classes.yaml"):
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Fichier introuvable: {yaml_path}")
    with open(yaml_path, 'r', encoding='utf-8') as f:
        classes = yaml.safe_load(f).get('classes', [])
    if '__background__' not in classes:
        classes = ['__background__'] + classes
    return classes


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
    return all_image_ids[:n_train], all_image_ids[n_train:n_train+n_val], all_image_ids[n_train+n_val:]


def main():
    parser = argparse.ArgumentParser(description="Régénère test_info.json pour EfficientDet")
    parser.add_argument("--output",      default=None,
                        help="Dossier de destination (défaut: run le plus récent)")
    parser.add_argument("--images",      default=os.getenv("DETECTION_DATASET_IMAGES_DIR",      "../dataset1/images/default"))
    parser.add_argument("--annotations", default=os.getenv("DETECTION_DATASET_ANNOTATIONS_FILE","../dataset1/annotations/instances_default.json"))
    parser.add_argument("--classes",     default=os.getenv("CLASSES_FILE", "classes.yaml"))
    parser.add_argument("--model-name",  default=os.getenv("EFFICIENTDET_MODEL", "tf_efficientdet_d0"))
    parser.add_argument("--image-size",  type=int, default=int(os.getenv("IMAGE_SIZE", "512")))
    parser.add_argument("--train-split", type=float, default=float(os.getenv("TRAIN_SPLIT", "0.70")))
    parser.add_argument("--val-split",   type=float, default=float(os.getenv("VAL_SPLIT",   "0.20")))
    parser.add_argument("--test-split",  type=float, default=float(os.getenv("TEST_SPLIT",  "0.10")))
    args = parser.parse_args()

    output_dir = args.output
    if output_dir is None:
        runs_base = os.path.join("runs", "detect", "train")
        if os.path.exists(runs_base):
            subdirs = sorted(
                [d for d in os.listdir(runs_base) if os.path.isdir(os.path.join(runs_base, d))],
                reverse=True
            )
            if subdirs:
                output_dir = os.path.join(runs_base, subdirs[0])

    if output_dir is None:
        print("❌ Impossible de trouver le dossier du modèle. Spécifiez --output")
        return

    print(f"📂 Destination   : {output_dir}")
    print(f"📂 Images        : {args.images}")
    print(f"📂 Annotations   : {args.annotations}")
    print(f"🧠 Modèle        : {args.model_name} ({args.image_size}px)")
    print(f"📐 Split         : {args.train_split}/{args.val_split}/{args.test_split}")

    if not os.path.exists(args.annotations):
        print(f"❌ Annotations introuvables: {args.annotations}")
        return

    classes     = load_classes(args.classes)
    coco        = COCO(args.annotations)
    cat_ids     = coco.getCatIds()
    cat_mapping = {cat_id: idx + 1 for idx, cat_id in enumerate(cat_ids)}

    train_ids, val_ids, test_ids = stratified_split(
        coco, args.train_split, args.val_split, args.test_split, seed=42
    )
    n_total = len(train_ids) + len(val_ids) + len(test_ids)
    print(f"\n📊 Split reconstruit:")
    print(f"   Total : {n_total} | Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")

    test_info = {
        'test_image_ids':   test_ids,
        'cat_mapping':      {str(k): v for k, v in cat_mapping.items()},
        'images_dir':       os.path.abspath(args.images),
        'annotations_file': os.path.abspath(args.annotations),
        'num_test_images':  len(test_ids),
        'classes':          classes,
        'model_name':       args.model_name,
        'image_size':       args.image_size,
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "test_info.json")
    with open(out_path, 'w') as f:
        json.dump(test_info, f, indent=2)

    print(f"\n✅ test_info.json sauvegardé: {out_path}")
    print(f"   Lance maintenant: python evaluate.py")


if __name__ == "__main__":
    main()
