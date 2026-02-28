import torch
import os
from pathlib import Path
from yolo_cv_utils import run_kfold_cv


def main():
    DATA_YAML = "data.yaml"
    N_FOLDS = 5
    
    # Split ratios (must sum to 1.0)
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.1
    TEST_RATIO = 0.2

    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    print(f'Script directory: {script_dir}')

    # Create main runs folder with shorter path
    runs_dir = script_dir / "cv_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Run k-fold cross-validation
    all_metrics, best_paths, cv_runs_dir = run_kfold_cv(
        data_yaml_path=script_dir / DATA_YAML,
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
