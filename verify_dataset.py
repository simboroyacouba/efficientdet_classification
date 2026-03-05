"""
Vérification du dataset COCO avant entraînement
"""

import os
import json
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def verify_dataset(images_dir, annotations_file):
    """Vérifier l'intégrité du dataset"""
    
    print("=" * 60)
    print("   VÉRIFICATION DU DATASET")
    print("=" * 60)
    
    if not os.path.exists(annotations_file):
        print(f"❌ Annotations introuvables: {annotations_file}")
        return False
    
    try:
        coco = COCO(annotations_file)
    except Exception as e:
        print(f"❌ Erreur chargement: {e}")
        return False
    
    print(f"\n📊 STATISTIQUES")
    print("-" * 40)
    print(f"   Images: {len(coco.imgs)}")
    print(f"   Annotations: {len(coco.anns)}")
    print(f"   Catégories: {len(coco.cats)}")
    
    print(f"\n📂 CATÉGORIES")
    print("-" * 40)
    for cat_id, cat in coco.cats.items():
        count = len(coco.getAnnIds(catIds=[cat_id]))
        print(f"   [{cat_id}] {cat['name']}: {count} annotations")
    
    print(f"\n🖼️  IMAGES")
    print("-" * 40)
    
    missing = []
    valid = 0
    
    for img_id, img_info in coco.imgs.items():
        path = os.path.join(images_dir, img_info['file_name'])
        if not os.path.exists(path):
            missing.append(img_info['file_name'])
        else:
            valid += 1
    
    print(f"   Valides: {valid}/{len(coco.imgs)}")
    if missing:
        print(f"   ❌ Manquantes: {len(missing)}")
        for m in missing[:5]:
            print(f"      - {m}")
    
    print(f"\n📝 ANNOTATIONS")
    print("-" * 40)
    
    invalid_bbox = 0
    for ann in coco.anns.values():
        if 'bbox' in ann:
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                invalid_bbox += 1
    
    print(f"   Bbox invalides: {invalid_bbox}")
    
    # Distribution
    anns_per_img = [len(coco.getAnnIds(imgIds=[img_id])) for img_id in coco.imgs]
    print(f"   Min/Max par image: {min(anns_per_img)} / {max(anns_per_img)}")
    print(f"   Moyenne: {np.mean(anns_per_img):.1f}")
    
    print("\n" + "=" * 60)
    if len(missing) == 0 and invalid_bbox == 0:
        print("   ✅ DATASET VALIDE")
        return True
    else:
        print("   ⚠️  DATASET AVEC PROBLÈMES")
        return False


def visualize_samples(images_dir, annotations_file, num_samples=3):
    """Visualiser quelques échantillons"""
    
    coco = COCO(annotations_file)
    
    img_ids = [i for i in coco.imgs if len(coco.getAnnIds(imgIds=[i])) > 0]
    np.random.shuffle(img_ids)
    img_ids = img_ids[:num_samples]
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    fig, axes = plt.subplots(1, num_samples, figsize=(6*num_samples, 6))
    if num_samples == 1:
        axes = [axes]
    
    for ax, img_id in zip(axes, img_ids):
        img_info = coco.imgs[img_id]
        path = os.path.join(images_dir, img_info['file_name'])
        
        img = Image.open(path)
        ax.imshow(img)
        
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
        
        for ann in anns:
            color = colors[ann['category_id'] % 10]
            x, y, w, h = ann['bbox']
            rect = patches.Rectangle((x, y), w, h, fill=False, 
                                     edgecolor=color, linewidth=2)
            ax.add_patch(rect)
            
            cat_name = coco.cats[ann['category_id']]['name']
            ax.text(x, y-5, cat_name, fontsize=8, color='white',
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
        
        ax.set_title(f"{img_info['file_name']}\n{len(anns)} annotations")
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig("dataset_samples.png", dpi=150)
    plt.show()
    print("📊 Échantillons sauvegardés: dataset_samples.png")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default=os.getenv("DETECTION_DATASET_IMAGES_DIR","./data/images"))
    parser.add_argument("--annotations", default=os.getenv("DETECTION_DATASET_ANNOTATIONS_FILE", "../dataset1/annotations/instances_default.json"))
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--num-samples", type=int, default=3)
    
    args = parser.parse_args()
    
    valid = verify_dataset(args.images, args.annotations)
    
    if valid and args.visualize:
        visualize_samples(args.images, args.annotations, args.num_samples)
