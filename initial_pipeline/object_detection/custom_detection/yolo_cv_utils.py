"""
Shared utilities for YOLO cross-validation training.
"""
from ultralytics import YOLO
import torch
import os
import random
import numpy as np
import yaml
import shutil
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import KFold

# Set random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_yolo_metrics(metrics, fold=None):
    """Print YOLO evaluation metrics."""
    if fold is not None:
        print(f"\n=== YOLO Evaluation Metrics (Fold {fold}) ===")
    else:
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


def collect_all_data(data_yaml_path):
    """Collect all images and labels from train/val/test splits."""
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    dataset_root = Path(data_config['path'])
    all_images = []
    
    # Collect from all splits
    for split in ['train', 'val', 'test']:
        if split in data_config:
            images_path = dataset_root / data_config[split]
            if images_path.exists():
                image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
                all_images.extend(image_files)
    
    total = len(all_images)
    print(f"Collected {total} total images from all splits")
    
    return all_images, dataset_root, data_config


def create_fold_structure(fold_dir, dataset_root):
    """Create the directory structure for a fold."""
    fold_dir = Path(fold_dir)
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (fold_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (fold_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (fold_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (fold_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
    
    return fold_dir


def copy_file_pair(image_path, dest_images_dir, dest_labels_dir):
    """Copy an image and its corresponding label file."""
    image_path = Path(image_path)
    dest_images_dir = Path(dest_images_dir)
    dest_labels_dir = Path(dest_labels_dir)
    
    # Verify source exists
    if not image_path.exists():
        print(f"Warning: Source image does not exist: {image_path}")
        return
    
    # Copy image - use shorter handle for Windows path length limits
    dest_image = dest_images_dir / image_path.name
    
    # Use os.path functions for better Windows compatibility
    try:
        # Ensure parent directory exists
        os.makedirs(str(dest_images_dir), exist_ok=True)
        shutil.copy2(str(image_path), str(dest_image))
    except Exception as e:
        print(f"Error copying image {image_path.name}")
        print(f"  Source: {image_path} (exists: {image_path.exists()})")
        print(f"  Dest: {dest_image}")
        print(f"  Dest dir: {dest_images_dir} (exists: {dest_images_dir.exists()})")
        raise
    
    # Copy label (assuming same name with .txt extension)
    label_path = image_path.parent.parent / 'labels' / image_path.parent.name / f"{image_path.stem}.txt"
    if label_path.exists():
        dest_label = dest_labels_dir / label_path.name
        os.makedirs(str(dest_labels_dir), exist_ok=True)
        shutil.copy2(str(label_path), str(dest_label))


def create_fold_yaml(fold_dir, data_config, fold_num):
    """Create a YAML config file for the fold."""
    fold_yaml = {
        'path': str(fold_dir),
        'train': 'images/train',
        'val': 'images/val',
        'nc': data_config['nc'],
        'names': data_config['names']
    }
    
    yaml_path = fold_dir / f'fold_{fold_num}.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(fold_yaml, f, default_flow_style=False)
    
    return yaml_path


def setup_kfold_data(all_images, dataset_root, data_config, folds_root, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, n_splits=5):
    """Setup K-fold cross-validation data structure with specified split proportions.
    
    Args:
        all_images: List of all image paths
        dataset_root: Root directory of dataset
        data_config: YAML configuration
        folds_root: Root directory for fold data
        train_ratio: Proportion of data for training (default 0.7)
        val_ratio: Proportion of data for validation (default 0.1)
        test_ratio: Proportion of data for testing (default 0.2)
        n_splits: Number of folds (default 5)
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    all_images = np.array(all_images)
    
    total = len(all_images)
    print(f"\nUsing split proportions:")
    print(f"  Train: {train_ratio*100:.1f}%")
    print(f"  Val: {val_ratio*100:.1f}%")
    print(f"  Test: {test_ratio*100:.1f}%")
    
    fold_configs = []
    
    for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(all_images)):
        print(f"\nSetting up Fold {fold_idx + 1}/{n_splits}...")
        
        # K-fold gives us test set (1/n_splits of data)
        # We need to adjust to match our desired test_ratio
        train_val_images = all_images[train_val_idx]
        test_images_kfold = all_images[test_idx]
        
        # Recombine and resplit according to our desired ratios
        all_fold_images = np.concatenate([train_val_images, test_images_kfold])
        np.random.shuffle(all_fold_images)
        
        # Calculate split sizes based on specified ratios
        n_train = int(len(all_fold_images) * train_ratio)
        n_val = int(len(all_fold_images) * val_ratio)
        n_test = len(all_fold_images) - n_train - n_val  # Remaining goes to test
        
        train_images = all_fold_images[:n_train]
        val_images = all_fold_images[n_train:n_train+n_val]
        test_images = all_fold_images[n_train+n_val:]
        
        print(f"  Train: {len(train_images)} ({len(train_images)/len(all_fold_images)*100:.1f}%)")
        print(f"  Val: {len(val_images)} ({len(val_images)/len(all_fold_images)*100:.1f}%)")
        print(f"  Test: {len(test_images)} ({len(test_images)/len(all_fold_images)*100:.1f}%)")
        
        # Create fold directory structure with shorter name
        fold_dir = folds_root / f'f{fold_idx + 1}'  # Shorter fold name
        create_fold_structure(fold_dir, dataset_root)
        
        # Copy files
        for img in train_images:
            copy_file_pair(Path(img), 
                         fold_dir / 'images' / 'train',
                         fold_dir / 'labels' / 'train')
        
        for img in val_images:
            copy_file_pair(Path(img),
                         fold_dir / 'images' / 'val',
                         fold_dir / 'labels' / 'val')
        
        # Create YAML config for this fold
        yaml_path = create_fold_yaml(fold_dir, data_config, fold_idx + 1)
        fold_configs.append((fold_idx + 1, yaml_path, test_images))
    
    return fold_configs


def train_fold(fold_num, yaml_path, device, runs_dir, timestamp, yolo_starting_model=None, train_params=None):
    """Train YOLO model on a single fold.
    
    Args:
        fold_num: Fold number
        yaml_path: Path to fold's YAML config
        device: Device to train on (cpu or cuda)
        runs_dir: Directory to save results
        timestamp: Timestamp for naming
        yolo_starting_model: Optional path to starting weights (for fine-tuning), or None to start from yolov8s.pt
        train_params: Optional dict of training parameters (epochs, patience, imgsz, batch, optimizer, lr0, momentum, etc.)
    
    Returns:
        metrics: Validation metrics
        best_path: Path to best model weights
    """
    print(f"\n{'='*60}")
    print(f"Training Fold {fold_num}")
    print(f"{'='*60}")
    
    # Load model from starting weights or pretrained yolov8s
    if yolo_starting_model is not None:
        print(f"Starting from model: {yolo_starting_model}")
        model = YOLO(yolo_starting_model)
    else:
        print("Starting from yolov8s.pt")
        model = YOLO("yolov8s.pt")
    
    run_name = f"fold_{fold_num}_{timestamp}"
    
    # Default training parameters
    default_params = {
        'data': str(yaml_path),
        'epochs': 100,
        'patience': 10,
        'imgsz': 512,
        'batch': 16,
        'device': device,
        'name': run_name,
        'project': str(runs_dir),
        'seed': RANDOM_SEED
    }
    
    # Merge with user-provided parameters
    if train_params:
        default_params.update(train_params)
    
    results = model.train(**default_params)
    
    # Load best model from this fold
    best_path = os.path.join(results.save_dir, "weights", "best.pt")
    model = YOLO(best_path)
    
    # Evaluate on validation split
    metrics = model.val(
        data=str(yaml_path),
        split="val",
        imgsz=512
    )
    
    print_yolo_metrics(metrics, fold=fold_num)
    
    return metrics, best_path


def aggregate_metrics(all_metrics):
    """Aggregate metrics across all folds."""
    print(f"\n{'='*60}")
    print("AGGREGATED RESULTS ACROSS ALL FOLDS")
    print(f"{'='*60}")
    
    metrics_dict = {
        'map50': [],
        'map': [],
        'precision': [],
        'recall': []
    }
    
    for metrics in all_metrics:
        metrics_dict['map50'].append(metrics.box.map50)
        metrics_dict['map'].append(metrics.box.map)
        metrics_dict['precision'].append(metrics.box.mp)
        metrics_dict['recall'].append(metrics.box.mr)
    
    print(f"\nmAP@50:      {np.mean(metrics_dict['map50']):.5f} ± {np.std(metrics_dict['map50']):.5f}")
    print(f"mAP@50–95:   {np.mean(metrics_dict['map']):.5f} ± {np.std(metrics_dict['map']):.5f}")
    print(f"Precision:   {np.mean(metrics_dict['precision']):.5f} ± {np.std(metrics_dict['precision']):.5f}")
    print(f"Recall:      {np.mean(metrics_dict['recall']):.5f} ± {np.std(metrics_dict['recall']):.5f}")
    
    print("\nPer-fold results:")
    for i, metrics in enumerate(all_metrics):
        print(f"  Fold {i+1}: mAP@50={metrics.box.map50:.5f}, "
              f"mAP@50-95={metrics.box.map:.5f}, "
              f"P={metrics.box.mp:.5f}, "
              f"R={metrics.box.mr:.5f}")
    
    return metrics_dict


def run_kfold_cv(data_yaml_path, runs_dir, device, train_ratio=0.7, val_ratio=0.1, 
                 test_ratio=0.2, n_folds=5, yolo_starting_model=None, train_params=None,
                 run_name_prefix="cv"):
    """Run complete k-fold cross-validation training.
    
    Args:
        data_yaml_path: Path to data YAML file
        runs_dir: Directory to save results
        device: Device to train on
        train_ratio: Train split ratio (default 0.7)
        val_ratio: Validation split ratio (default 0.1)
        test_ratio: Test split ratio (default 0.2)
        n_folds: Number of folds (default 5)
        yolo_starting_model: Optional starting model for fine-tuning
        train_params: Optional dict of training parameters
        run_name_prefix: Prefix for run name (default "cv")
    
    Returns:
        all_metrics: List of metrics from all folds
        best_paths: List of (fold_num, best_path) tuples
        runs_dir: Path to results directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv_runs_dir = runs_dir / f"{run_name_prefix}_{timestamp}"
    cv_runs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create folds directory
    folds_root = cv_runs_dir / "folds"
    folds_root.mkdir(parents=True, exist_ok=True)

    # Collect all data and setup K-fold structure
    print("\n" + "="*60)
    print("SETTING UP K-FOLD CROSS-VALIDATION")
    print("="*60)
    
    all_images, dataset_root, data_config = collect_all_data(data_yaml_path)
    fold_configs = setup_kfold_data(all_images, dataset_root, data_config, folds_root, 
                                    train_ratio=train_ratio, val_ratio=val_ratio, 
                                    test_ratio=test_ratio, n_splits=n_folds)

    # Train each fold
    all_metrics = []
    best_paths = []
    
    for fold_num, yaml_path, test_images in fold_configs:
        metrics, best_path = train_fold(fold_num, yaml_path, device, cv_runs_dir, timestamp,
                                       yolo_starting_model=yolo_starting_model,
                                       train_params=train_params)
        all_metrics.append(metrics)
        best_paths.append((fold_num, best_path))

    # Aggregate and display results
    aggregate_metrics(all_metrics)
    
    print(f"\n{'='*60}")
    print("Cross-validation complete!")
    print(f"Results saved to: {cv_runs_dir}")
    print(f"{'='*60}")
    
    return all_metrics, best_paths, cv_runs_dir
