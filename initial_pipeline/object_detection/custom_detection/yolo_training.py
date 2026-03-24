import argparse
import torch
import os
from pathlib import Path
from yolo_cv_utils import run_kfold_cv


def parse_args():
    """Parse command-line arguments for YOLO cross-validation training."""
    parser = argparse.ArgumentParser(description="Run YOLO k-fold cross-validation training")
    parser.add_argument(
        "--data-yaml",
        default="data.yaml",
        help="Path to the primary dataset YAML (default: data.yaml)",
    )
    parser.add_argument(
        "--additional-data-yaml",
        default="data_additional.yaml",
        help="Path to the additional dataset YAML mixed into train only (default: data_additional.yaml)",
    )
    return parser.parse_args()


def main():
    N_FOLDS = 5
    args = parse_args()
    
    # Split ratios (must sum to 1.0)
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.1
    TEST_RATIO = 0.2

    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    print(f'Script directory: {script_dir}')

    data_yaml_path = Path(args.data_yaml)
    if not data_yaml_path.is_absolute():
        data_yaml_path = script_dir / data_yaml_path

    additional_data_yaml_path = Path(args.additional_data_yaml)
    if not additional_data_yaml_path.is_absolute():
        additional_data_yaml_path = script_dir / additional_data_yaml_path

    print(f'Primary data YAML: {data_yaml_path}')
    print(f'Additional data YAML: {additional_data_yaml_path}')

    # Create main runs folder with shorter path
    runs_dir = script_dir / "cv_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Run k-fold cross-validation
    all_metrics, best_paths, cv_runs_dir = run_kfold_cv(
        data_yaml_path=data_yaml_path,
        additional_data_yaml_path=additional_data_yaml_path,
        runs_dir=runs_dir,
        device=device,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        n_folds=N_FOLDS,
        yolo_starting_model=None,  # Start from yolov8s.pt
        train_params=None,  # Use default training parameters
        run_name_prefix="cv"
    )


if __name__ == "__main__":
    main()
