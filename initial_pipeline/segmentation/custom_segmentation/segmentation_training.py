from data_utils import index_segmentations_df
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold, train_test_split
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import os
import torch
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import torch
import cv2
from segmentation_preprocessing import preprocess_segmentations
from random_disconnections.disconnect_components import disconnect_branches_with_gap
from random_disconnections.find_connection_points import find_random_branch_points
from random_disconnections.random_colour_noise import add_floating_synthetic_fragments
from training.training_utils import set_seed, parse_segmentation_args, save_model_with_timestamp
from training.segmentation_dataset import SegmentationDataset
from training.unet import UNet
from training.cl_dice import soft_cldice_loss, dice_loss

# Setup imports from initial_pipeline (for morphology features)
from setup_imports import setup_initial_pipeline_path
setup_initial_pipeline_path()

from morphology.morphology_features import (
    compute_skeleton_length,
    compute_branch_count,
    compute_skeleton_components,
    compute_mask_area
)

from skimage.morphology import skeletonize


def dice_score(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2 * inter + eps) / (union + eps)

def iou_score(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return (inter + eps) / (union + eps)


def morphology_similarity_score(pred_mask, target_mask, eps=1e-8):
    """
    Compute morphological feature similarity between predicted and target masks.
    
    This compares key morphological features:
    - num_branches: critical for microglia phenotype
    - num_components: detects fragmentation
    - length_pixels: overall skeleton length
    - cell_area: overall cell size
    
    Returns a score in [0, 1] where 1 is perfect similarity.
    """
    # Convert to numpy and binary
    pred_np = (pred_mask > 0.5).cpu().numpy().astype(bool)
    target_np = (target_mask > 0.5).cpu().numpy().astype(bool)
    
    # Check if masks are empty
    if not pred_np.any() or not target_np.any():
        return 0.0
    
    # Compute skeletons
    try:
        pred_skel = skeletonize(pred_np)
        target_skel = skeletonize(target_np)
    except:
        return 0.0
    
    # Extract morphological features
    pred_features = {
        'num_branches': compute_branch_count(pred_skel),
        'num_components': compute_skeleton_components(pred_skel),
        'length_pixels': compute_skeleton_length(pred_skel),
        'cell_area': compute_mask_area(pred_np),
    }
    
    target_features = {
        'num_branches': compute_branch_count(target_skel),
        'num_components': compute_skeleton_components(target_skel),
        'length_pixels': compute_skeleton_length(target_skel),
        'cell_area': compute_mask_area(target_np),
    }
    
    # Compute normalized similarity for each feature
    # Using symmetric relative error: |pred - target| / (pred + target)
    similarities = []
    weights = {
        'num_branches': 2.0,      # Most important for phenotype
        'num_components': 1.5,    # Important for detecting fragmentation
        'length_pixels': 1.0,     # Standard weight
        'cell_area': 1.0,         # Standard weight
    }
    
    for feature_name in pred_features.keys():
        pred_val = pred_features[feature_name]
        target_val = target_features[feature_name]
        weight = weights[feature_name]
        
        # Symmetric relative error
        if pred_val + target_val > 0:
            rel_error = abs(pred_val - target_val) / (pred_val + target_val + eps)
            similarity = 1.0 / (1.0 + rel_error)  # Convert to similarity [0, 1]
        else:
            similarity = 1.0  # Both are 0
        
        similarities.append(weight * similarity)
    
    # Weighted average
    total_weight = sum(weights.values())
    morphology_score = sum(similarities) / total_weight
    
    return morphology_score


class CombinedLoss(nn.Module):
    """
    Flexible loss wrapper supporting multiple loss types:
    - 'bce': Binary Cross Entropy with Logits
    - 'dice': Dice Loss
    - 'cldice': Centerline Dice Loss (topology-aware)
    - 'bce_cldice': Weighted combination of BCE and clDice
    - 'dice_cldice': Weighted combination of Dice and clDice
    """
    def __init__(self, loss_type='bce', alpha=0.5):
        """
        Args:
            loss_type: One of ['bce', 'dice', 'cldice', 'bce_cldice', 'dice_cldice']
            alpha: Weight for clDice in combined losses (0.0-1.0)
                   For combined losses: loss = (1-alpha)*base_loss + alpha*cldice_loss
        """
        super().__init__()
        self.loss_type = loss_type
        self.alpha = alpha
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, logits, target):
        """
        Args:
            logits: Raw model outputs (before sigmoid) - shape (B, C, H, W)
            target: Ground truth binary masks - shape (B, C, H, W)
        
        Returns:
            Scalar loss value
        """
        # Get probabilities for losses that need them
        probs = torch.sigmoid(logits)
        
        if self.loss_type == 'bce':
            # Standard Binary Cross Entropy
            return self.bce_loss(logits, target)
        
        elif self.loss_type == 'dice':
            # Dice Loss (negative dice coefficient)
            # dice_loss from cl_dice.py expects probs, not logits
            return dice_loss(probs, target).mean()
        
        elif self.loss_type == 'cldice':
            # Pure Centerline Dice Loss
            return soft_cldice_loss(probs, target).mean()
        
        elif self.loss_type == 'bce_cldice':
            # Combined: BCE + clDice
            bce = self.bce_loss(logits, target)
            cldice = soft_cldice_loss(probs, target).mean()
            return (1 - self.alpha) * bce + self.alpha * cldice
        
        elif self.loss_type == 'dice_cldice':
            # Combined: Dice + clDice
            dice = dice_loss(probs, target).mean()
            cldice = soft_cldice_loss(probs, target).mean()
            return (1 - self.alpha) * dice + self.alpha * cldice
        
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


def train_epoch(model, loader, device, optimizer, criterion):
    model.train()
    total_loss = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)



@torch.no_grad()
def evaluate(model, loader, device, calculate_morphology_metric=False):
    """
    Evaluate model on a dataset.
    
    Args:
        model: The segmentation model
        loader: DataLoader for the dataset
        device: Device to run evaluation on
        calculate_morphology_metric: If True, compute morphology similarity score.
                                     Default False since it's computationally expensive.
    
    Returns:
        Dictionary with dice, iou, and optionally morphology scores
    """
    model.eval()
    dice_list, iou_list = [], []
    morphology_list = [] if calculate_morphology_metric else None

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits)

        # Pixel-level metrics (always computed)
        dice_list.append(dice_score(probs, y).item())
        iou_list.append(iou_score(probs, y).item())
        
        # Morphology-level metric (only if requested)
        if calculate_morphology_metric:
            batch_size = probs.shape[0]
            batch_morphology_scores = []
            for i in range(batch_size):
                morph_score = morphology_similarity_score(probs[i, 0], y[i, 0])
                batch_morphology_scores.append(morph_score)
            
            morphology_list.extend(batch_morphology_scores)

    results = {
        "dice": sum(dice_list) / len(dice_list),
        "iou": sum(iou_list) / len(iou_list),
    }
    
    if calculate_morphology_metric:
        results["morphology"] = sum(morphology_list) / len(morphology_list)
    
    return results


# Visualize predictions

def main(): 
    args = parse_segmentation_args()

    print('arg value')
    disconnect_components = args.disconnect_components
    add_new_components = args.add_new_components
    print(args)

    set_seed(42)
    print('preparing segmentations')
    preprocess_segmentations()

    print('segmentations ready')

    SEGMENTATIONS_DIR = Path.cwd().parents[2] / "AnnotationsData" / "Segmentations"

    
    path_df = index_segmentations_df(
        SEGMENTATIONS_DIR,
        mask_name = 'calculated_masks'
    )

    mask_quality_df = pd.read_csv('mask_quality_summary.csv')
    mask_quality_df = mask_quality_df[['image_path', 'mask_quality']]


    mask_quality_df["image_path"] = mask_quality_df["image_path"].apply(Path)

    path_df = path_df.merge(mask_quality_df, on = 'image_path', how = 'inner')
    print(path_df.columns)

    # Only segment the entire cell ie. not just the SOMA
    path_df = path_df[path_df['class'] == 'MG_whole']
    # Remove all bad annotations
    path_df = path_df[~path_df['mask_quality'].isin(['bad', 'bad_image_quality', 'disagree'])]

    # path_df = path_df.merg(mask_quality_df, on = )
    # Only take first 160 for now to train fast
    # path_df = path_df.head(80)
    # print(len(path_df), "annotation pairs found")

    k_folds = 5
    kf = KFold(
        n_splits=k_folds,
        shuffle=True,
        random_state=42
    )
    all_fold_metrics = []

    train_tfms = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        ToTensorV2()
    ])

    test_tfms = A.Compose([
        A.Resize(256, 256),
        ToTensorV2()
    ])



    for fold, (trainval_idx, test_idx) in enumerate(kf.split(path_df)):
        print(f'Currently doing fold {fold}')
        trainval_df = path_df.iloc[trainval_idx]
        test_df     = path_df.iloc[test_idx]

        # Split train → train / val
        train_df, val_df = train_test_split(
            trainval_df,
            test_size=0.1,
            random_state=42,
            shuffle=True
        )

        print(
            f"Sizes | Train: {len(train_df)}, "
            f"Val: {len(val_df)}, "
            f"Test: {len(test_df)}"
        )

        train_ds = SegmentationDataset(
            train_df, train_tfms,
            disconnect_components=disconnect_components,
            add_new_components=add_new_components
        )

        val_ds = SegmentationDataset(
            val_df, test_tfms,
            disconnect_components=False,
            add_new_components=False
        )

        test_ds = SegmentationDataset(
            test_df, test_tfms,
            disconnect_components=False,
            add_new_components=False
        )

        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False)
        test_loader  = DataLoader(test_ds,  batch_size=16, shuffle=False)

        # Setup loss function based on arguments
        criterion = CombinedLoss(loss_type=args.loss_type, alpha=args.cldice_alpha)
        print(f'Using loss function: {args.loss_type}')
        if args.loss_type in ['bce_cldice', 'dice_cldice']:
            print(f'  clDice weight (alpha): {args.cldice_alpha}')
    
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f'currently using device: {device}')

        model = UNet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        max_epochs = args.epochs
        patience = args.patience

        best_dice = -float("inf")
        best_epoch = -1
        epochs_no_improve = 0
        best_state_dict = None


        for epoch in trange(max_epochs, desc="Training", unit="epoch"):
            train_loss = train_epoch(model, train_loader, device = device, optimizer = optimizer, criterion = criterion)
            # Don't calculate morphology during training (expensive)
            test_metrics = evaluate(model, test_loader, device = device, calculate_morphology_metric=False)
            val_metrics = evaluate(model, val_loader, device = device, calculate_morphology_metric=False)

            current_dice = val_metrics["dice"]

            # Check improvement
            if current_dice > best_dice:
                best_dice = current_dice
                best_epoch = epoch
                epochs_no_improve = 0

                # store BEST model weights in memory
                best_state_dict = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

                save_model_with_timestamp(model)
                status = "NEW BEST"

            else:
                epochs_no_improve += 1
                status = f"no improve ({epochs_no_improve}/{patience})"

            tqdm.write(
                f"Epoch {epoch:03d} | "
                f"Loss: {train_loss:.4f} | "
                f"Validation Dice: {current_dice:.4f} | "
                f"Validation IoU: {val_metrics['iou']:.4f} | "
                f"Test Dice: {test_metrics['dice']:.4f} | "
                f"Test IoU: {test_metrics['iou']:.4f} | "
                f"{status}"
            )

            # Early stopping
            if epochs_no_improve >= patience:
                tqdm.write(
                    f"Early stopping at epoch {epoch}. "
                    f"Best Dice {best_dice:.4f} at epoch {best_epoch}."
                )
                break

        # restore best model before testing
        model.load_state_dict(best_state_dict)
        model.to(device)

        # Final test evaluation - NOW calculate morphology metric
        final_test_metrics = evaluate(model, test_loader, device, calculate_morphology_metric=True)
        all_fold_metrics.append(final_test_metrics)


        print(
            f"Fold {fold + 1} final test | "
            f"Dice: {final_test_metrics['dice']:.4f}, "
            f"IoU: {final_test_metrics['iou']:.4f}, "
            f"Morphology: {final_test_metrics['morphology']:.4f}"
        )

    dice_scores = [m["dice"] for m in all_fold_metrics]
    iou_scores  = [m["iou"]  for m in all_fold_metrics]
    morphology_scores = [m["morphology"] for m in all_fold_metrics]

    print("\n========== Cross-validation results ==========")
    print(f"Dice:       {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
    print(f"IoU:        {np.mean(iou_scores):.4f} ± {np.std(iou_scores):.4f}")
    print(f"Morphology: {np.mean(morphology_scores):.4f} ± {np.std(morphology_scores):.4f}")


if __name__ == "__main__":
    main()