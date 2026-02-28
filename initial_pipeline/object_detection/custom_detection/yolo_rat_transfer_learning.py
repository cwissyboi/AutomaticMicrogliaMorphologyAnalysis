from ultralytics import YOLO
import torch
import os
from pathlib import Path
from datetime import datetime
from yolo_cv_utils import run_kfold_cv, RANDOM_SEED


def print_yolo_metrics(metrics):
    """Print YOLO evaluation metrics."""
    print("\n=== YOLO Evaluation Metrics ===")

    print(f"mAP@50:      {metrics.box.map50:.5f}")
    print(f"mAP@50–95:   {metrics.box.map:.5f}")
    print(f"Precision:   {metrics.box.mp:.5f}")
    print(f"Recall:      {metrics.box.mr:.5f}")

    print("\nPer-class results:")
    for i, name in metrics.names.items():
        print(
            f"  Class '{name}': "
            f"P={metrics.box.p[i]:.5f}, "
            f"R={metrics.box.r[i]:.5f}, "
            f"AP50={metrics.box.ap50[i]:.5f}, "
            f"AP={metrics.box.ap[i]:.5f}"
        )


def main():
    RAT_DATA_YAML = "rat_data.yaml"
    HUMAN_DATA_YAML = "data.yaml"
    N_FOLDS = 5
    
    # Split ratios for human data cross-validation
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.1
    TEST_RATIO = 0.2

    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    print(f'Random seed: {RANDOM_SEED}')

    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    runs_dir = script_dir / "yolo_runs_transfer"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # STEP 1: Train on rat dataset (NO cross-validation)
    # ============================================================
    print("\n" + "="*60)
    print("STEP 1: TRAINING ON RAT DATASET")
    print("="*60)
    
    model = YOLO("yolov8s.pt")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rat_run_name = f"rat_pretrain_{timestamp}"

    rat_results = model.train(
        data=str(script_dir / RAT_DATA_YAML),
        epochs=100,
        patience=10,
        imgsz=512,
        batch=16,
        device=device,
        name=rat_run_name,
        project=str(runs_dir),
        seed=RANDOM_SEED
    )

    rat_best_path = os.path.join(rat_results.save_dir, "weights", "best.pt")

    print("\nFinished rat pretraining")
    print(f"Best rat weights: {rat_best_path}")

    # ============================================================
    # STEP 2: Fine-tune on human data with 5-fold cross-validation
    # ============================================================
    print("\n" + "="*60)
    print("STEP 2: FINE-TUNING ON HUMAN DATA (5-FOLD CV)")
    print("="*60)

    # Training parameters for fine-tuning
    finetune_params = {
        'epochs': 100,
        'patience': 10,
        'imgsz': 512,
        'batch': 16,
        'optimizer': 'AdamW',
        'lr0': 0.001,  # Lower learning rate for fine-tuning
        'momentum': 0.937  # Only meaningful for SGD
    }

    # Run 5-fold cross-validation on human data starting from rat-pretrained weights
    all_metrics, best_paths, cv_runs_dir = run_kfold_cv(
        data_yaml_path=script_dir / HUMAN_DATA_YAML,
        runs_dir=runs_dir,
        device=device,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        n_folds=N_FOLDS,
        yolo_starting_model=rat_best_path,  # Start from rat-pretrained weights
        train_params=finetune_params,
        run_name_prefix=f"human_finetune_{timestamp}"
    )

    print(f"\n{'='*60}")
    print("Transfer learning with cross-validation complete!")
    print(f"Rat pretraining weights: {rat_best_path}")
    print(f"Human fine-tuning results: {cv_runs_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
