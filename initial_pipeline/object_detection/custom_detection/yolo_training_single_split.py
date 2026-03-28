import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO

from yolo_cv_utils import (
    RANDOM_SEED,
    collect_all_data,
    copy_file_pair,
    create_fold_structure,
    create_fold_yaml,
    duplicate_filenames,
    print_yolo_metrics,
    train_fold,
    validate_compatible_data_configs,
)


def parse_args():
    """Parse command-line arguments for single-split YOLO training."""
    parser = argparse.ArgumentParser(description="Run YOLO training with one 80/20 train/val split")
    parser.add_argument(
        "--data-yaml",
        default="data.yaml",
        help="Path to the primary dataset YAML used to build train/val (default: data.yaml)",
    )
    parser.add_argument(
        "--additional-data-yaml",
        default=None,
        help="Optional additional dataset YAML mixed into train only",
    )
    parser.add_argument(
        "--test-data-yaml",
        default=None,
        help="Path to the dataset YAML used entirely for test evaluation",
    )
    return parser.parse_args()


def main():
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.2

    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Random seed: {RANDOM_SEED}")

    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    print(f"Script directory: {script_dir}")

    data_yaml_path = Path(args.data_yaml)
    if not data_yaml_path.is_absolute():
        data_yaml_path = script_dir / data_yaml_path

    additional_data_yaml_path = None
    if args.additional_data_yaml:
        additional_data_yaml_path = Path(args.additional_data_yaml)
        if not additional_data_yaml_path.is_absolute():
            additional_data_yaml_path = script_dir / additional_data_yaml_path

    test_data_yaml_path = Path(args.test_data_yaml)
    if not test_data_yaml_path.is_absolute():
        test_data_yaml_path = script_dir / test_data_yaml_path

    print(f"Primary data YAML: {data_yaml_path}")
    print(f"Additional data YAML: {additional_data_yaml_path}")
    print(f"Test data YAML: {test_data_yaml_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Keep paths short to avoid Windows MAX_PATH issues with long image filenames.
    runs_dir = script_dir / "ss_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = runs_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    split_dir = run_dir / "f1"

    print("\n" + "=" * 60)
    print("SETTING UP SINGLE 80/20 TRAIN-VAL SPLIT")
    print("=" * 60)

    all_images, dataset_root, data_config = collect_all_data(data_yaml_path)
    all_images = np.array(all_images)

    rng = np.random.RandomState(RANDOM_SEED)
    shuffled_images = all_images[rng.permutation(len(all_images))]

    n_train = int(len(shuffled_images) * TRAIN_RATIO)
    train_images = shuffled_images[:n_train]
    val_images = shuffled_images[n_train:]

    print(f"  Train (primary): {len(train_images)} ({len(train_images)/len(shuffled_images)*100:.1f}%)")
    print(f"  Val: {len(val_images)} ({len(val_images)/len(shuffled_images)*100:.1f}%)")

    additional_train_images = []
    if additional_data_yaml_path is not None:
        print(f"Using additional training data YAML: {additional_data_yaml_path}")
        additional_train_images, _, additional_data_config = collect_all_data(additional_data_yaml_path)
        validate_compatible_data_configs(data_config, additional_data_config)
        print(
            "Additional dataset images from train/val/test will be appended to train split: "
            f"{len(additional_train_images)} images"
        )

    additional_duplicates = duplicate_filenames(additional_train_images)
    if additional_duplicates:
        sample = additional_duplicates[:10]
        raise ValueError(
            "Additional dataset contains duplicate image filenames that would overwrite in train dir. "
            f"Examples: {sample}"
        )

    train_name_set = {Path(p).name for p in train_images}
    additional_name_set = {Path(p).name for p in additional_train_images}
    overlap = sorted(train_name_set & additional_name_set)
    if overlap:
        sample = overlap[:10]
        raise ValueError(
            "Primary and additional train data share image filenames. "
            f"These would overwrite in train dir. Examples: {sample}"
        )

    create_fold_structure(split_dir, dataset_root)

    for img in train_images:
        copy_file_pair(Path(img), split_dir / "images" / "train", split_dir / "labels" / "train")

    for img in additional_train_images:
        copy_file_pair(Path(img), split_dir / "images" / "train", split_dir / "labels" / "train")

    for img in val_images:
        copy_file_pair(Path(img), split_dir / "images" / "val", split_dir / "labels" / "val")

    total_train = len(train_images) + len(additional_train_images)
    print(f"  Added additional train images: {len(additional_train_images)}")
    print(f"  Total train images: {total_train}")

    train_yaml_path = create_fold_yaml(split_dir, data_config, 1)

    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    _, best_path = train_fold(
        fold_num=1,
        yaml_path=train_yaml_path,
        device=device,
        runs_dir=run_dir,
        timestamp=timestamp,
        yolo_starting_model=None,
        train_params=None,
    )

    print("\n" + "=" * 60)
    print("PREPARING TEST DATASET (ALL IMAGES FROM TEST YAML)")
    print("=" * 60)

    test_images, _, test_data_config = collect_all_data(test_data_yaml_path)
    validate_compatible_data_configs(data_config, test_data_config)

    test_dir = run_dir / "t"
    (test_dir / "images" / "test").mkdir(parents=True, exist_ok=True)
    (test_dir / "labels" / "test").mkdir(parents=True, exist_ok=True)

    for img in test_images:
        copy_file_pair(Path(img), test_dir / "images" / "test", test_dir / "labels" / "test")

    print(f"Copied {len(test_images)} images into test split")

    test_yaml = {
        "path": str(test_dir),
        "test": "images/test",
        "nc": data_config["nc"],
        "names": data_config["names"],
    }
    test_yaml_path = test_dir / "test_all.yaml"
    with open(test_yaml_path, "w") as f:
        yaml.dump(test_yaml, f, default_flow_style=False)

    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON EXTERNAL TEST DATASET")
    print("=" * 60)

    model = YOLO(best_path)
    test_metrics = model.val(
        data=str(test_yaml_path),
        split="test",
        imgsz=512,
        device=device,
    )
    print_yolo_metrics(test_metrics)

    print("\n" + "=" * 60)
    print("Single-split training complete!")
    print(f"Run directory: {run_dir}")
    print(f"Best model: {best_path}")
    print(f"Test YAML used for final evaluation: {test_data_yaml_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
