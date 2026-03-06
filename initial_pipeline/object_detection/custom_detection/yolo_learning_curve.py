"""
Learning curve script for YOLO detection training.

Trains a YOLOv8s model with increasing portions of the training data
(10%, 20%, ..., 100%) while keeping the validation and test sets fixed.
For each portion, the best model is evaluated on the fixed test set and
the results are saved.  A single summary plot is produced at the end.

The val and test splits come directly from data.yaml (images/val and
images/test), so they are never modified.  Only images/train is
sub-sampled.
"""

import torch
import os
import random
import shutil
import gc
import numpy as np
import yaml
import json
import matplotlib
matplotlib.use("Agg")   # headless backend – safe on HPC
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# ──────────────────────────── reproducibility ────────────────────────────── #
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ─────────────────────────── training parameters ─────────────────────────── #
TRAIN_PARAMS = {
    "epochs":   100,
    "patience": 10,
    "imgsz":    512,
    "batch":    16,
    "seed":     RANDOM_SEED,
}

# Portions of training data to evaluate (10 % … 100 %)
PORTIONS = [i / 10 for i in range(1, 11)]   # [0.1, 0.2, …, 1.0]


# ─────────────────────────────── helpers ─────────────────────────────────── #

def cleanup_memory():
    """Free GPU / CPU memory between runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def collect_split_images(dataset_root: Path, split_dir: str) -> list[Path]:
    """Return all .jpg / .png image paths for one split directory."""
    split_path = dataset_root / split_dir
    if not split_path.exists():
        raise FileNotFoundError(f"Split directory not found: {split_path}")
    images = sorted(
        list(split_path.glob("*.jpg")) + list(split_path.glob("*.png"))
    )
    print(f"  Found {len(images)} images in: {split_path}")
    return images


def get_label_path(image_path: Path) -> Path:
    """Derive the .txt label path from an image path (images/ → labels/)."""
    parts = list(image_path.parts)
    for i, part in enumerate(parts):
        if part == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def copy_pair(image_path: Path, dest_images: Path, dest_labels: Path):
    """Copy one image + its label into dest_images / dest_labels."""
    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)

    shutil.copy2(str(image_path), str(dest_images / image_path.name))

    label_path = get_label_path(image_path)
    if label_path.exists():
        shutil.copy2(str(label_path), str(dest_labels / label_path.name))
    else:
        print(f"  WARNING: label not found for {image_path.name}")


def build_portion_dataset(
    portion: float,
    all_train_images: list[Path],
    val_images: list[Path],
    test_images: list[Path],
    portion_dir: Path,
    data_config: dict,
):
    """
    Create a self-contained dataset directory for one training-data portion.

    Directory layout:
        portion_dir/
            images/{train,val,test}/
            labels/{train,val,test}/
            portion.yaml
    """
    # Sample from full training set (fixed seed → reproducible)
    n_train = max(1, int(len(all_train_images) * portion))
    rng = np.random.RandomState(RANDOM_SEED)
    sampled_train = rng.choice(all_train_images, size=n_train, replace=False)

    print(f"\n  Portion {portion*100:.0f}%: using {n_train}/{len(all_train_images)} train images")

    # Copy train subset
    for img in sampled_train:
        copy_pair(
            Path(img),
            portion_dir / "images" / "train",
            portion_dir / "labels" / "train",
        )

    # Copy fixed val
    for img in val_images:
        copy_pair(
            Path(img),
            portion_dir / "images" / "val",
            portion_dir / "labels" / "val",
        )

    # Copy fixed test
    for img in test_images:
        copy_pair(
            Path(img),
            portion_dir / "images" / "test",
            portion_dir / "labels" / "test",
        )

    # Write YAML
    yaml_config = {
        "path": str(portion_dir),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    data_config["nc"],
        "names": data_config["names"],
    }
    yaml_path = portion_dir / "portion.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_config, f, default_flow_style=False)

    return yaml_path, n_train


def train_portion(
    portion: float,
    yaml_path: Path,
    device,
    runs_dir: Path,
    timestamp: str,
) -> tuple:
    """
    Train YOLOv8s on one data portion.

    Returns
    -------
    val_metrics   – ultralytics Metrics object evaluated on val split
    best_pt_path  – path to best.pt weights
    """
    run_name = f"portion_{int(portion*100):03d}pct_{timestamp}"

    model = YOLO("yolov8s.pt")

    params = {
        "data":    str(yaml_path),
        "device":  device,
        "name":    run_name,
        "project": str(runs_dir),
        **TRAIN_PARAMS,
    }

    results = model.train(**params)

    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    return best_pt


def evaluate_on_test(best_pt: Path, yaml_path: Path, device) -> dict:
    """
    Load best.pt and evaluate on the *test* split.

    Returns a dict with scalar metric values.
    """
    model = YOLO(str(best_pt))
    metrics = model.val(
        data=str(yaml_path),
        split="test",
        imgsz=TRAIN_PARAMS["imgsz"],
        device=device,
    )

    result = {
        "map50":     float(metrics.box.map50),
        "map50_95":  float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall":    float(metrics.box.mr),
    }

    print(f"  Test results → mAP@50={result['map50']:.4f}  "
          f"mAP@50-95={result['map50_95']:.4f}  "
          f"P={result['precision']:.4f}  R={result['recall']:.4f}")

    return result


# ──────────────────────────── visualisation ──────────────────────────────── #

METRIC_META = {
    "map50":     ("mAP@50",      "#1f77b4"),
    "map50_95":  ("mAP@50-95",   "#ff7f0e"),
    "precision": ("Precision",   "#2ca02c"),
    "recall":    ("Recall",      "#d62728"),
}


def plot_learning_curves(
    results: list[dict],
    output_path: Path,
):
    """
    Produce a 2×2 grid of learning curves (one panel per metric).

    Parameters
    ----------
    results : list of dicts with keys
              { portion, n_train, map50, map50_95, precision, recall }
    output_path : where to save the PNG
    """
    portions_pct = [r["portion"] * 100 for r in results]
    n_trains     = [r["n_train"]        for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        "YOLO Learning Curve – performance vs. training data fraction",
        fontsize=14, fontweight="bold", y=1.01,
    )

    for ax, (metric_key, (label, color)) in zip(axes.flat, METRIC_META.items()):
        values = [r[metric_key] for r in results]

        ax.plot(portions_pct, values, marker="o", color=color,
                linewidth=2, markersize=6, label=label)
        ax.fill_between(portions_pct, values, alpha=0.08, color=color)

        # Annotate final value
        ax.annotate(
            f"{values[-1]:.3f}",
            xy=(portions_pct[-1], values[-1]),
            xytext=(5, 5), textcoords="offset points",
            fontsize=9, color=color,
        )

        ax.set_title(label, fontsize=12)
        ax.set_xlabel("Training data used (%)", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(portions_pct)
        ax.grid(True, linestyle="--", alpha=0.5)

        # Secondary x-axis with absolute sample counts
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(portions_pct)
        ax2.set_xticklabels(n_trains, rotation=45, fontsize=7)
        ax2.set_xlabel("# train images", fontsize=8)

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nLearning curve plot saved → {output_path}")


def plot_combined_overlay(
    results: list[dict],
    output_path: Path,
):
    """
    Single axes with all four metrics overlaid for direct comparison.
    """
    portions_pct = [r["portion"] * 100 for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(
        "YOLO Learning Curve – all metrics",
        fontsize=14, fontweight="bold",
    )

    for metric_key, (label, color) in METRIC_META.items():
        values = [r[metric_key] for r in results]
        ax.plot(portions_pct, values, marker="o", color=color,
                linewidth=2, markersize=6, label=label)

    ax.set_xlabel("Training data used (%)", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(portions_pct)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Combined overlay plot saved → {output_path}")


# ─────────────────────────────── main ────────────────────────────────────── #

def main():
    DATA_YAML = "data.yaml"

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    data_yaml_path = script_dir / DATA_YAML

    # ── load dataset config ──────────────────────────────────────────────── #
    with open(data_yaml_path, "r") as f:
        data_config = yaml.safe_load(f)

    dataset_root = Path(data_config["path"])
    if not dataset_root.is_absolute():
        dataset_root = (script_dir / dataset_root).resolve()

    print(f"\nDataset root: {dataset_root}")

    # ── collect fixed val / test images ─────────────────────────────────── #
    print("\nCollecting fixed splits...")
    all_train_images = collect_split_images(dataset_root, data_config["train"])
    val_images       = collect_split_images(dataset_root, data_config["val"])
    test_images      = collect_split_images(dataset_root, data_config["test"])

    print(f"\n  Total train pool : {len(all_train_images)}")
    print(f"  Fixed val        : {len(val_images)}")
    print(f"  Fixed test       : {len(test_images)}")

    # ── output directories ───────────────────────────────────────────────── #
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = script_dir / "learning_curve_runs" / f"lc_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_dir      = output_dir / "training_runs"
    datasets_dir  = output_dir / "datasets"
    runs_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")

    # ── run loop ─────────────────────────────────────────────────────────── #
    all_results = []

    for portion in PORTIONS:
        pct_label = f"{int(portion*100):03d}pct"
        print(f"\n{'='*60}")
        print(f"  PORTION {int(portion*100)}%")
        print(f"{'='*60}")

        # 1. Build dataset for this portion
        portion_dir = datasets_dir / pct_label
        yaml_path, n_train = build_portion_dataset(
            portion,
            all_train_images,
            val_images,
            test_images,
            portion_dir,
            data_config,
        )

        # 2. Train
        best_pt = train_portion(portion, yaml_path, device, runs_dir, timestamp)

        # 3. Evaluate on fixed test set
        print(f"\n  Evaluating on fixed test set ({len(test_images)} images)...")
        test_metrics = evaluate_on_test(best_pt, yaml_path, device)

        # 4. Collect result
        row = {
            "portion":   portion,
            "n_train":   n_train,
            **test_metrics,
        }
        all_results.append(row)

        # 5. Save running results to JSON (allows inspection if script is interrupted)
        results_json = output_dir / "learning_curve_results.json"
        with open(results_json, "w") as f:
            json.dump(all_results, f, indent=2)

        cleanup_memory()

    # ── final summary ────────────────────────────────────────────────────── #
    print(f"\n{'='*60}")
    print("LEARNING CURVE RESULTS (test set)")
    print(f"{'='*60}")
    header = f"{'Portion':>8}  {'N_train':>8}  {'mAP@50':>8}  {'mAP@50-95':>10}  {'Precision':>10}  {'Recall':>8}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(
            f"{r['portion']*100:7.0f}%  "
            f"{r['n_train']:8d}  "
            f"{r['map50']:8.4f}  "
            f"{r['map50_95']:10.4f}  "
            f"{r['precision']:10.4f}  "
            f"{r['recall']:8.4f}"
        )

    # ── plots ────────────────────────────────────────────────────────────── #
    plot_learning_curves(
        all_results,
        output_dir / "learning_curve_2x2.png",
    )
    plot_combined_overlay(
        all_results,
        output_dir / "learning_curve_combined.png",
    )

    print(f"\nAll done.  Results in: {output_dir}")


if __name__ == "__main__":
    main()
