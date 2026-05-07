"""
Optimisation des hyperparametres EfficientDet unifie avec Optuna.

Algorithme : TPE (Tree-structured Parzen Estimator) + MedianPruner.
Les etudes sont persistees dans une base SQLite (resumable).

Usage :
  python tune_unified.py --n-trials 20
  python tune_unified.py --n-trials 30 --tune-epochs 10
  python tune_unified.py --resume                       # reprendre une etude existante
  python tune_unified.py --tune-fpn                     # inclure le choix du FPN dans la recherche
"""

import os
import gc
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

import torch
import optuna
from optuna.samplers import TPESampler
from optuna.pruners  import MedianPruner
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from effdet import DetBenchPredict
    EFFDET_AVAILABLE = True
except ImportError:
    EFFDET_AVAILABLE = False

from train_unified import (
    load_classes,
    stratified_split,
    EfficientDetDataset,
    collate_fn,
    build_train_model,
    train_one_epoch,
    evaluate_epoch,
)


# =============================================================================
# ESPACE DE RECHERCHE
# =============================================================================

FPN_CHOICES = ["", "bifpn_sum", "bifpn_fa", "bifpn_attn", "pan_fa"]

def _suggest_hparams(trial, tune_fpn=False):
    """Definit l'espace de recherche des hyperparametres."""
    hp = {
        "lr":           trial.suggest_float("lr",           1e-5, 5e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        "beta1":        trial.suggest_float("beta1",        0.85, 0.95),
    }
    if tune_fpn:
        hp["fpn_name"] = trial.suggest_categorical("fpn_name", FPN_CHOICES)
    return hp


# =============================================================================
# OBJECTIF OPTUNA
# =============================================================================

def make_objective(base_config, coco, train_ids, val_ids, cat_mapping):
    """
    Retourne la fonction objectif Optuna.
    Le split dataset est prepare une seule fois (partage entre tous les trials).
    """
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes     = base_config["classes"]
    class_names = [c for c in classes if c != '__background__']
    num_classes = len(class_names)
    tune_fpn    = base_config["tune_fpn"]
    use_amp     = base_config["amp"] and device.type == "cuda"

    def objective(trial):
        hp = _suggest_hparams(trial, tune_fpn=tune_fpn)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- Datasets ---
        train_dataset = EfficientDetDataset(
            base_config["images_dir"], base_config["annotations_file"],
            train_ids, cat_mapping, base_config["image_size"], augment=True,
        )
        val_dataset = EfficientDetDataset(
            base_config["images_dir"], base_config["annotations_file"],
            val_ids, cat_mapping, base_config["image_size"], augment=False,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=base_config["batch_size"],
            shuffle=True, collate_fn=collate_fn, num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False,
            collate_fn=collate_fn, num_workers=0,
        )

        # --- Modele ---
        fpn_name    = hp.get("fpn_name") or base_config.get("fpn_name") or None
        train_model = build_train_model(
            base_config["model_name"], num_classes,
            base_config["image_size"], pretrained=True,
            fpn_name=fpn_name,
        )
        train_model.to(device)
        predict_model = DetBenchPredict(train_model.model)
        predict_model.to(device)

        optimizer = torch.optim.AdamW(
            train_model.parameters(),
            lr=hp["lr"],
            betas=(hp["beta1"], 0.999),
            weight_decay=hp["weight_decay"],
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=base_config["tune_epochs"], eta_min=1e-6,
        )
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        # --- Boucle d'entrainement ---
        best_map50 = 0.0
        try:
            for epoch in range(1, base_config["tune_epochs"] + 1):
                train_one_epoch(
                    train_model, optimizer, train_loader,
                    device, base_config["grad_clip"], scaler=scaler,
                )
                val_map50 = evaluate_epoch(
                    predict_model, val_loader, device,
                    class_names, base_config["score_threshold"],
                )
                lr_scheduler.step()

                best_map50 = max(best_map50, val_map50)
                trial.report(val_map50, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"\n   [Trial {trial.number}] Erreur : {e}")
            best_map50 = 0.0
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                del train_model, predict_model, optimizer, lr_scheduler
            except Exception:
                pass

        return best_map50

    return objective


# =============================================================================
# RAPPORT FINAL
# =============================================================================

def _print_and_save_report(study, base_config):
    best = study.best_trial

    print("\n" + "=" * 70)
    print("   RESULTATS DE L'OPTIMISATION")
    print("=" * 70)
    print(f"   Nombre d'essais termines : {len(study.trials)}")
    print(f"   Meilleur mAP@50          : {best.value:.4f}  (trial #{best.number})")
    print(f"\n   Meilleurs hyperparametres :")
    for k, v in best.params.items():
        print(f"      {k:<20} {v}")

    completed = [t for t in study.trials if t.value is not None]
    completed.sort(key=lambda t: t.value, reverse=True)
    print(f"\n   Top-5 trials :")
    print(f"   {'Trial':>6}  {'mAP@50':>8}  {'lr':>10}  {'wd':>10}  {'beta1':>8}")
    print(f"   {'-'*55}")
    for t in completed[:5]:
        p = t.params
        print(
            f"   {t.number:>6}  {t.value:>8.4f}  "
            f"{p.get('lr', 0):>10.6f}  {p.get('weight_decay', 0):>10.6f}  "
            f"{p.get('beta1', 0):>8.4f}"
        )

    report = {
        "study_name":   study.study_name,
        "n_trials":     len(study.trials),
        "best_trial":   best.number,
        "best_map50":   best.value,
        "best_params":  best.params,
        "top5": [
            {"trial": t.number, "map50": t.value, "params": t.params}
            for t in completed[:5]
        ],
        "optimized_at": datetime.now().isoformat(),
    }

    report_path = os.path.join(base_config["tune_dir"], "best_hparams.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    p = best.params
    fpn_arg = f" --fpn-name {p['fpn_name']}" if "fpn_name" in p else ""
    print(f"\n   Meilleurs hparams sauvegardes : {report_path}")
    print(f"\n   Lancer l'entrainement final :")
    print(
        f"   python train_unified.py"
        f" --lr {p.get('lr', 1e-4):.6f}"
        f" --weight-decay {p.get('weight_decay', 1e-4):.6f}"
        f"{fpn_arg}"
    )
    print("=" * 70)

    return report


# =============================================================================
# MAIN
# =============================================================================

def main():
    if not EFFDET_AVAILABLE:
        print("Installez d'abord: pip install effdet timm")
        return

    parser = argparse.ArgumentParser(
        description="Optimisation hyperparametres EfficientDet unifie — Optuna",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-trials",    type=int, default=20,
                        help="Nombre d'essais Optuna.")
    parser.add_argument("--tune-epochs", type=int, default=8,
                        help="Epochs par essai (moins que l'entrainement final).")
    parser.add_argument("--study-name",  default=None,
                        help="Nom de l'etude Optuna (default : effdet_unified_<date>).")
    parser.add_argument("--resume",      action="store_true",
                        help="Reprendre une etude existante (meme --study-name).")
    parser.add_argument("--tune-fpn",    action="store_true",
                        help="Inclure le choix du mecanisme FPN/attention dans la recherche.")
    parser.add_argument("--fpn-name",    default=os.getenv("FPN_NAME", ""),
                        choices=["", "bifpn_sum", "bifpn_fa", "bifpn_attn", "pan_fa"],
                        help="FPN fixe pour tous les essais (ignore si --tune-fpn).")
    parser.add_argument("--model-name",  default=os.getenv("EFFICIENTDET_MODEL", "tf_efficientdet_d0"),
                        help="Variante EfficientDet (d0..d7).")
    parser.add_argument("--amp",         action="store_true", default=os.getenv("USE_AMP", "0") == "1",
                        help="Mixed precision fp16 (recommande pour D3+).")
    parser.add_argument("--images-dir",       default=None)
    parser.add_argument("--annotations-file", default=None)
    parser.add_argument("--classes-file",     default=None)
    parser.add_argument("--output-dir",       default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(os.getenv("OUTPUT_DIR", "./runs/detect/train"), "unified")
    tune_dir   = os.path.join(output_dir, "tuning")
    os.makedirs(tune_dir, exist_ok=True)

    annotations_file = args.annotations_file or os.getenv(
        "DETECTION_DATASET_ANNOTATIONS_FILE",
        "../dataset1/annotations/instances_default.json",
    )
    classes_file = args.classes_file or os.getenv("CLASSES_FILE", "classes.yaml")

    classes = load_classes(classes_file)
    if not classes:
        print("Erreur : aucune classe chargee.")
        return

    EFFICIENTDET_IMAGE_SIZES = {
        "tf_efficientdet_d0": 512, "tf_efficientdet_d1": 640,
        "tf_efficientdet_d2": 768, "tf_efficientdet_d3": 896,
        "tf_efficientdet_d4": 1024,"tf_efficientdet_d5": 1280,
        "tf_efficientdet_d6": 1280,"tf_efficientdet_d7": 1536,
    }
    image_size = EFFICIENTDET_IMAGE_SIZES.get(args.model_name, 512)

    base_config = {
        "images_dir":       args.images_dir or os.getenv("DETECTION_DATASET_IMAGES_DIR", "../dataset1/images/default"),
        "annotations_file": annotations_file,
        "output_dir":       output_dir,
        "tune_dir":         tune_dir,
        "tune_epochs":      args.tune_epochs,
        "model_name":       args.model_name,
        "image_size":       image_size,
        "batch_size":       int(os.getenv("BATCH_SIZE", "2")),
        "train_split":      float(os.getenv("TRAIN_SPLIT", "0.70")),
        "val_split":        float(os.getenv("VAL_SPLIT",  "0.20")),
        "test_split":       float(os.getenv("TEST_SPLIT", "0.10")),
        "score_threshold":  float(os.getenv("SCORE_THRESHOLD", "0.3")),
        "grad_clip":        float(os.getenv("GRAD_CLIP", "1.0")),
        "classes":          classes,
        "tune_fpn":         args.tune_fpn,
        "fpn_name":         args.fpn_name or None,
        "amp":              args.amp,
    }

    print("=" * 70)
    print(f"   OPTUNA — Optimisation hyperparametres EfficientDet UNIFIE")
    print("=" * 70)
    print(f"   Modele       : {args.model_name} ({image_size}px)")
    print(f"   Classes      : {len(classes) - 1} (hors background)")
    print(f"   FPN/Attention: {'recherche' if args.tune_fpn else (args.fpn_name or 'defaut modele')}")
    print(f"   AMP          : {'OUI' if args.amp else 'non'}")
    print(f"   Essais       : {args.n_trials}  x  {args.tune_epochs} epochs")
    print(f"   Dossier      : {tune_dir}")

    # --- Preparer le split une seule fois ---
    print("\n   Preparation du dataset (une seule fois pour tous les essais)...")
    coco      = COCO(annotations_file)
    cat_ids   = coco.getCatIds()
    coco_cats = {cat['id']: cat['name'] for cat in coco.loadCats(cat_ids)}
    cat_mapping = {}
    for cat_id, cat_name in coco_cats.items():
        if cat_name in classes:
            cat_mapping[cat_id] = classes.index(cat_name)
        else:
            print(f"   Categorie COCO ignoree : '{cat_name}' (id={cat_id})")

    train_ids, val_ids, _, _ = stratified_split(
        coco,
        base_config["train_split"],
        base_config["val_split"],
        base_config["test_split"],
        seed=42,
    )
    print(f"   Train: {len(train_ids)} | Val: {len(val_ids)} images")

    # --- Creer ou reprendre l'etude ---
    study_name   = args.study_name or f"effdet_unified_{datetime.now().strftime('%Y%m%d_%H%M')}"
    storage_path = os.path.join(tune_dir, "optuna_study.db")
    storage_url  = f"sqlite:///{storage_path}"

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=args.resume,
        direction="maximize",
        sampler=TPESampler(seed=42, n_startup_trials=5),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3, interval_steps=1),
    )

    already_done = len([t for t in study.trials if t.value is not None])
    n_remaining  = max(0, args.n_trials - already_done)

    if n_remaining == 0:
        print(f"\n   Etude deja complete ({already_done} essais). Utilisez --resume pour continuer.")
    else:
        if already_done > 0:
            print(f"\n   Reprise : {already_done} essais deja faits, {n_remaining} restants.")
        print(f"\n   Lancement de l'optimisation ({n_remaining} essais)...\n")
        study.optimize(
            make_objective(base_config, coco, train_ids, val_ids, cat_mapping),
            n_trials=n_remaining,
            show_progress_bar=True,
            catch=(Exception,),
        )

    _print_and_save_report(study, base_config)


if __name__ == "__main__":
    main()
