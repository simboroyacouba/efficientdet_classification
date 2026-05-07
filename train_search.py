"""
Recherche d'hyperparamètres EfficientDet
Méthodes: grid search, random search
Chaque trial lance train_unified.py en sous-processus et rapporte mAP@50 + temps.

Usage:
  python train_search.py --method random --n-trials 8 --epochs 15
  python train_search.py --method grid   --epochs 10
"""

import argparse
import itertools
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# =============================================================================
# ESPACE DE RECHERCHE
# Modifiez ici les valeurs à tester.
# =============================================================================

SEARCH_SPACE = {
    "lr":           [1e-4, 5e-4, 1e-3],
    "batch_size":   [2, 4],
    "momentum":     [0.9, 0.95],       # beta1 de AdamW
    "weight_decay": [1e-4, 5e-4],
    "fpn_name":     ["", "bifpn_attn", "bifpn_sum", "pan_fa"],
    "augment":      [False, True],
    "amp":          [False, True],     # mixed precision (GPU uniquement)
}


# =============================================================================
# MÉTHODES DE RECHERCHE
# =============================================================================

def grid_search(space):
    """Toutes les combinaisons possibles."""
    keys   = list(space.keys())
    values = list(space.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def random_search(space, n, seed=42):
    """N combinaisons tirées aléatoirement."""
    rng    = random.Random(seed)
    trials = []
    for _ in range(n):
        trials.append({k: rng.choice(v) for k, v in space.items()})
    return trials


# =============================================================================
# CONSTRUCTION DE LA COMMANDE
# =============================================================================

def build_cmd(params, base_args):
    cmd = [sys.executable, "train_unified.py"] + base_args
    cmd += ["--lr",           str(params["lr"])]
    cmd += ["--batch-size",   str(int(params["batch_size"]))]
    cmd += ["--momentum",     str(params["momentum"])]
    cmd += ["--weight-decay", str(params["weight_decay"])]
    if params.get("fpn_name"):
        cmd += ["--fpn-name", params["fpn_name"]]
    if params.get("augment"):
        cmd += ["--augment"]
    if params.get("amp"):
        cmd += ["--amp"]
    return cmd


def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s" if m else f"{s}s"


# =============================================================================
# LECTURE DES RÉSULTATS D'UN TRIAL
# =============================================================================

def find_latest_run(runs_dir, created_after):
    """Retourne le répertoire de run créé après `created_after` (timestamp)."""
    if not os.path.exists(runs_dir):
        return None
    best_dir, best_t = None, 0
    for d in Path(runs_dir).iterdir():
        if not d.is_dir():
            continue
        t = d.stat().st_ctime
        if t >= created_after and t > best_t:
            best_t, best_dir = t, d
    return best_dir


def read_best_map(run_dir):
    if run_dir is None:
        return None
    h_path = Path(run_dir) / "history.json"
    if not h_path.exists():
        return None
    with open(h_path) as f:
        h = json.load(f)
    return h.get("best_map50")


# =============================================================================
# LANCEMENT D'UN TRIAL
# =============================================================================

def run_trial(params, base_args, trial_idx, total, runs_dir):
    cmd = build_cmd(params, base_args)

    print(f"\n{'─'*60}")
    print(f"  Trial [{trial_idx}/{total}] — {datetime.now().strftime('%H:%M:%S')}")
    for k, v in params.items():
        print(f"    {k:<20} = {v}")
    print(f"  Commande: {' '.join(cmd)}")
    print(f"{'─'*60}")

    created_after = time.time() - 1
    t0   = time.time()
    proc = subprocess.run(cmd, text=True)
    elapsed = time.time() - t0

    run_dir = find_latest_run(runs_dir, created_after)
    map50   = read_best_map(run_dir)

    status = "OK" if proc.returncode == 0 else f"ERREUR (code {proc.returncode})"
    print(f"\n  [{status}] mAP@50 = {map50:.4f if map50 is not None else '?'}"
          f"  |  temps = {format_time(elapsed)}")

    return {
        "trial":      trial_idx,
        "params":     params,
        "map50":      map50,
        "time_s":     round(elapsed, 1),
        "time_fmt":   format_time(elapsed),
        "returncode": proc.returncode,
        "run_dir":    str(run_dir) if run_dir else None,
    }


# =============================================================================
# RAPPORT FINAL
# =============================================================================

def print_report(results, method, total_time):
    valid = [r for r in results if r["map50"] is not None]
    valid.sort(key=lambda r: r["map50"], reverse=True)

    print("\n" + "=" * 70)
    print(f"  RÉSULTATS — {method.upper()} SEARCH ({len(results)} trials)")
    print("=" * 70)
    print(f"  {'#':<4} {'mAP@50':>8} {'temps':>9}  hyperparamètres")
    print(f"  {'─'*65}")

    for rank, r in enumerate(valid, 1):
        p = r["params"]
        hp = (f"lr={p['lr']}  bs={int(p['batch_size'])}"
              f"  mom={p['momentum']}  wd={p['weight_decay']}")
        fpn = p.get("fpn_name") or "bifpn_fa"
        hp += f"  fpn={fpn}"
        flags = []
        if p.get("augment"): flags.append("aug")
        if p.get("amp"):     flags.append("amp")
        if flags:
            hp += "  [" + "+".join(flags) + "]"
        print(f"  {rank:<4} {r['map50']:>8.4f} {r['time_fmt']:>9}  {hp}")

    print(f"\n  Temps total : {format_time(total_time)}")
    print(f"  Temps moyen : {format_time(total_time / len(results))}/trial")

    if valid:
        best = valid[0]
        print(f"\n  Meilleurs hyperparamètres (mAP@50 = {best['map50']:.4f}):")
        for k, v in best["params"].items():
            print(f"    {k:<20} = {v}")
        print(f"\n  Pour entraîner le modèle final avec ces paramètres :")
        final_args = [a for a in build_cmd(best["params"], [])
                      if a not in (sys.executable, "train_unified.py")]
        print(f"    python train_unified.py {' '.join(final_args)}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Recherche d'hyperparamètres EfficientDet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--method",   choices=["grid", "random"], default="random",
                        help="Methode de recherche")
    parser.add_argument("--n-trials", type=int, default=8,
                        help="Nombre d'essais (random search uniquement)")
    parser.add_argument("--epochs",   type=int,
                        default=int(os.getenv("SEARCH_EPOCHS", "15")),
                        help="Epochs par trial")
    parser.add_argument("--output",   default="./search_results",
                        help="Dossier de sortie des résultats")
    parser.add_argument("--runs-dir", default=os.getenv("RUNS_DIR", "./runs/detect/train"),
                        help="Dossier où train_unified.py sauvegarde les runs")
    parser.add_argument("--seed",     type=int, default=42,
                        help="Graine aléatoire (random search)")
    # Transmis directement à train_unified.py
    parser.add_argument("--images-dir",       default=os.getenv("DETECTION_DATASET_IMAGES_DIR",       None))
    parser.add_argument("--annotations-file", default=os.getenv("DETECTION_DATASET_ANNOTATIONS_FILE", None))
    parser.add_argument("--classes-file",     default=os.getenv("CLASSES_FILE",                       None))
    parser.add_argument("--model-name",       default=os.getenv("EFFICIENTDET_MODEL",                 None),
                        help="Variante EfficientDet (tf_efficientdet_d0, d1, ...)")
    parser.add_argument("--image-size",       default=None, type=int,
                        help="Taille d'image (512, 640, ...)")
    parser.add_argument("--mode",             default=None, choices=["nadir", "oblique"],
                        help="Mode de train_unified.py (optionnel)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Construire les trials
    if args.method == "grid":
        trials = grid_search(SEARCH_SPACE)
        print(f"Grid search : {len(trials)} combinaisons × {args.epochs} epochs")
    else:
        trials = random_search(SEARCH_SPACE, args.n_trials, seed=args.seed)
        print(f"Random search : {args.n_trials} essais × {args.epochs} epochs")

    # Arguments de base passés à chaque trial
    base_args = ["--epochs", str(args.epochs)]
    if args.images_dir:
        base_args += ["--images-dir",       args.images_dir]
    if args.annotations_file:
        base_args += ["--annotations-file", args.annotations_file]
    if args.classes_file:
        base_args += ["--classes-file",     args.classes_file]
    if args.model_name:
        base_args += ["--model-name",       args.model_name]
    if args.image_size:
        base_args += ["--image-size",       str(args.image_size)]
    if args.mode:
        base_args += ["--mode",             args.mode]

    print(f"Epochs/trial : {args.epochs} | Méthode : {args.method}")
    print(f"Résultats    : {args.output}")

    results   = []
    start_all = time.time()

    for i, params in enumerate(trials, 1):
        result = run_trial(params, base_args, i, len(trials), args.runs_dir)
        results.append(result)

        # Sauvegarde intermédiaire après chaque trial
        out_path = os.path.join(args.output, "search_results.json")
        with open(out_path, "w") as f:
            json.dump({
                "method":    args.method,
                "epochs":    args.epochs,
                "timestamp": datetime.now().isoformat(),
                "trials":    results,
            }, f, indent=2)

    total_time = time.time() - start_all
    print_report(results, args.method, total_time)

    # Sauvegarde finale
    valid = [r for r in results if r["map50"] is not None]
    valid.sort(key=lambda r: r["map50"], reverse=True)
    out_path = os.path.join(args.output, "search_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "method":       args.method,
            "epochs":       args.epochs,
            "n_trials":     len(results),
            "total_time_s": round(total_time, 1),
            "timestamp":    datetime.now().isoformat(),
            "best":         valid[0] if valid else None,
            "trials":       results,
        }, f, indent=2)

    print(f"\n  Résultats sauvegardés : {out_path}")


if __name__ == "__main__":
    main()
