"""
Evaluation utilities for segmentation models with region-based analysis.

This module provides functions to evaluate segmentation predictions with
separate metrics for different anatomical regions (soma vs branches).
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from skimage.morphology import skeletonize
import sys
from pathlib import Path

# Setup imports from initial_pipeline for morphology features
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from morphology.morphology_features import (
    compute_skeleton_length,
    compute_branch_count,
    compute_skeleton_components,
    compute_mask_area
)


@dataclass
class RegionMetrics:
    """Metrics for a specific region (soma or branches)."""
    dice: float
    iou: float
    precision: float
    recall: float
    pixel_count: int  # Number of ground truth pixels in this region


@dataclass
class SegmentationMetrics:
    """Complete segmentation metrics including region-specific results."""
    # Overall metrics (entire cell)
    overall_dice: float
    overall_iou: float
    overall_precision: float
    overall_recall: float
    
    # Region-specific metrics
    soma_metrics: Optional[RegionMetrics] = None
    branches_metrics: Optional[RegionMetrics] = None


def dice_score(pred, target, eps=1e-6):
    """
    Compute Dice score between prediction and target.
    
    Args:
        pred: Predicted mask (will be binarized at 0.5)
        target: Ground truth mask (can have fractional values from interpolation)
        eps: Small constant to avoid division by zero
        
    Returns:
        Dice score as tensor
    """
    pred = (pred > 0.5).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2 * inter + eps) / (union + eps)


def iou_score(pred, target, eps=1e-6):
    """
    Compute IoU (Jaccard) score between prediction and target.
    
    Args:
        pred: Predicted mask (will be binarized at 0.5)
        target: Ground truth mask (can have fractional values from interpolation)
        eps: Small constant to avoid division by zero
        
    Returns:
        IoU score as tensor
    """
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


def compute_binary_metrics(pred: torch.Tensor, target: torch.Tensor, 
                           epsilon: float = 1e-7) -> Tuple[float, float, float, float]:
    """
    Compute binary segmentation metrics.
    
    NOTE: This function is kept for backward compatibility but is NOT used by evaluate_regions()
    which now uses dice_score() and iou_score() from segmentation_training.py directly.
    
    Args:
        pred: Predicted binary mask [H, W] or [B, H, W]
        target: Ground truth binary mask [H, W] or [B, H, W]
        epsilon: Small constant to avoid division by zero
        
    Returns:
        Tuple of (dice, iou, precision, recall)
    """
    pred = pred.flatten()
    target = target.flatten()
    
    # True positives, false positives, false negatives
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    
    # Dice score
    dice = (2 * tp + epsilon) / (2 * tp + fp + fn + epsilon)
    
    # IoU (Jaccard)
    iou = (tp + epsilon) / (tp + fp + fn + epsilon)
    
    # Precision and Recall
    precision = (tp + epsilon) / (tp + fp + epsilon)
    recall = (tp + epsilon) / (tp + fn + epsilon)
    
    return dice, iou, precision, recall


def evaluate_regions(pred: torch.Tensor, 
                     whole_cell_mask: torch.Tensor,
                     soma_mask: Optional[torch.Tensor] = None,
                     threshold: float = 0.5) -> SegmentationMetrics:
    """
    Evaluate segmentation prediction with separate metrics for soma and branches.
    
    Uses dice_score() and iou_score() from segmentation_training.py for consistency:
    - Prediction is binarized at threshold
    - Target mask can have fractional values (from interpolation during resize)
    - This allows partial credit at boundaries
    
    Args:
        pred: Predicted mask [H, W] (can be logits or probabilities)
        whole_cell_mask: Ground truth mask for entire cell [H, W] (can have fractional values)
        soma_mask: Ground truth mask for soma/cell body [H, W]. If None, only overall metrics computed.
        threshold: Threshold to binarize prediction
        
    Returns:
        SegmentationMetrics object with overall and region-specific metrics
    """
    # Ensure tensors are on CPU
    pred = pred.cpu()
    whole_cell_mask = whole_cell_mask.cpu()
    
    # Keep whole_cell_mask as-is (may have interpolated values between 0 and 1)
    whole_cell_target = whole_cell_mask
    
    # Compute overall metrics using EXACT SAME functions as evaluate() in segmentation_training.py
    # These functions expect probabilities and will binarize at 0.5 internally
    overall_dice = dice_score(pred, whole_cell_target).item()
    overall_iou = iou_score(pred, whole_cell_target).item()
    
    # For precision/recall, use soft approximations
    pred_binary = (pred > threshold).float()
    inter = (pred_binary * whole_cell_target).sum().item()
    pred_sum = pred_binary.sum().item()
    target_sum = whole_cell_target.sum().item()
    overall_precision = (inter + 1e-6) / (pred_sum + 1e-6)  # Approximation
    overall_recall = (inter + 1e-6) / (target_sum + 1e-6)  # Approximation
    
    # Initialize region metrics as None
    soma_metrics = None
    branches_metrics = None
    
    # If soma mask is provided, compute region-specific metrics
    if soma_mask is not None:
        soma_mask = soma_mask.cpu()
        # Keep soma_mask as-is (may have interpolated values)
        soma_target = soma_mask
        
        # Create branches mask: whole cell - soma (keep fractional values)
        branches_target = whole_cell_target * (1 - soma_target)
        
        # ========================================================================
        # SOMA METRICS: Recall only (using soft formulation)
        # ========================================================================
        # For soma, we only compute recall because:
        # - We don't know which predictions were "intended" for soma vs branches
        # - Recall = what proportion of GT soma was captured by prediction
        
        soma_pixel_count = soma_target.sum().item()  # Can be fractional
        if soma_pixel_count > 0:
            # Intersection: how many soma pixels were predicted correctly (can be fractional)
            pred_binary = (pred > threshold).float()
            soma_intersection = (pred_binary * soma_target).sum().item()
            soma_recall = soma_intersection / soma_pixel_count
            
            soma_metrics = RegionMetrics(
                dice=soma_recall,      # Store as recall for all fields (for compatibility)
                iou=soma_recall,
                precision=soma_recall,
                recall=soma_recall,
                pixel_count=int(soma_pixel_count)  # Store as int for reporting
            )
        
        # ========================================================================
        # BRANCHES METRICS: Full metrics using dice_score/iou_score from segmentation_training.py
        # ========================================================================
        # For branches, we compute soft Dice/IoU because:
        # - We define "branch predictions" as all predictions OUTSIDE the soma region
        # - GT branches = whole cell - soma (can have fractional values)
        # - pred_branches = all predictions outside soma region
        
        branches_pixel_count = branches_target.sum().item()  # Can be fractional
        if branches_pixel_count > 0:
            # Create branches prediction: predictions outside soma region
            # We mask by (1 - soma_target) to exclude the soma region
            pred_binary = (pred > threshold).float()
            pred_branches = pred_binary * (1 - soma_target)
            
            # Use the EXACT SAME functions from segmentation_training.py
            # These expect the pred to be probabilities (will binarize internally)
            # So we pass pred_branches which is already binary, and branches_target
            branches_dice = dice_score(pred_branches, branches_target).item()
            branches_iou = iou_score(pred_branches, branches_target).item()
            
            # Soft precision/recall approximations
            branches_inter = (pred_branches * branches_target).sum().item()
            branches_pred_sum = pred_branches.sum().item()
            branches_precision = (branches_inter + 1e-6) / (branches_pred_sum + 1e-6)
            branches_recall = (branches_inter + 1e-6) / (branches_pixel_count + 1e-6)
            
            branches_metrics = RegionMetrics(
                dice=branches_dice,
                iou=branches_iou,
                precision=branches_precision,
                recall=branches_recall,
                pixel_count=int(branches_pixel_count)  # Store as int for reporting
            )
    
    return SegmentationMetrics(
        overall_dice=overall_dice,
        overall_iou=overall_iou,
        overall_precision=overall_precision,
        overall_recall=overall_recall,
        soma_metrics=soma_metrics,
        branches_metrics=branches_metrics
    )


def aggregate_metrics(metrics_list: list[SegmentationMetrics]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics across multiple samples.
    
    Args:
        metrics_list: List of SegmentationMetrics objects
        
    Returns:
        Dictionary with mean and std for each metric category
    """
    # Overall metrics
    overall_dice = [m.overall_dice for m in metrics_list]
    overall_iou = [m.overall_iou for m in metrics_list]
    overall_precision = [m.overall_precision for m in metrics_list]
    overall_recall = [m.overall_recall for m in metrics_list]
    
    results = {
        'overall': {
            'dice_mean': np.mean(overall_dice),
            'dice_std': np.std(overall_dice),
            'iou_mean': np.mean(overall_iou),
            'iou_std': np.std(overall_iou),
            'precision_mean': np.mean(overall_precision),
            'precision_std': np.std(overall_precision),
            'recall_mean': np.mean(overall_recall),
            'recall_std': np.std(overall_recall),
        }
    }
    
    # Soma metrics (only from samples that have soma masks)
    soma_samples = [m.soma_metrics for m in metrics_list if m.soma_metrics is not None]
    if soma_samples:
        results['soma'] = {
            'dice_mean': np.mean([s.dice for s in soma_samples]),
            'dice_std': np.std([s.dice for s in soma_samples]),
            'iou_mean': np.mean([s.iou for s in soma_samples]),
            'iou_std': np.std([s.iou for s in soma_samples]),
            'precision_mean': np.mean([s.precision for s in soma_samples]),
            'precision_std': np.std([s.precision for s in soma_samples]),
            'recall_mean': np.mean([s.recall for s in soma_samples]),
            'recall_std': np.std([s.recall for s in soma_samples]),
            'avg_pixel_count': np.mean([s.pixel_count for s in soma_samples]),
            'sample_count': len(soma_samples),
        }
    
    # Branches metrics (only from samples that have branches)
    branches_samples = [m.branches_metrics for m in metrics_list if m.branches_metrics is not None]
    if branches_samples:
        results['branches'] = {
            'dice_mean': np.mean([b.dice for b in branches_samples]),
            'dice_std': np.std([b.dice for b in branches_samples]),
            'iou_mean': np.mean([b.iou for b in branches_samples]),
            'iou_std': np.std([b.iou for b in branches_samples]),
            'precision_mean': np.mean([b.precision for b in branches_samples]),
            'precision_std': np.std([b.precision for b in branches_samples]),
            'recall_mean': np.mean([b.recall for b in branches_samples]),
            'recall_std': np.std([b.recall for b in branches_samples]),
            'avg_pixel_count': np.mean([b.pixel_count for b in branches_samples]),
            'sample_count': len(branches_samples),
        }
    
    return results


def print_metrics_summary(results: Dict[str, Dict[str, float]]):
    """
    Print a formatted summary of aggregated metrics.
    
    Args:
        results: Dictionary from aggregate_metrics()
    """
    print("\n" + "="*70)
    print("SEGMENTATION EVALUATION RESULTS")
    print("="*70)
    
    # Overall metrics
    print("\nOVERALL (Entire Cell):")
    print(f"  Dice:      {results['overall']['dice_mean']:.4f} ± {results['overall']['dice_std']:.4f}")
    print(f"  IoU:       {results['overall']['iou_mean']:.4f} ± {results['overall']['iou_std']:.4f}")
    print(f"  Precision: {results['overall']['precision_mean']:.4f} ± {results['overall']['precision_std']:.4f}")
    print(f"  Recall:    {results['overall']['recall_mean']:.4f} ± {results['overall']['recall_std']:.4f}")
    
    # Soma metrics
    if 'soma' in results:
        print("\nSOMA (Cell Body) - RECALL ONLY:")
        print(f"  Samples:    {results['soma']['sample_count']}")
        print(f"  Avg Pixels: {results['soma']['avg_pixel_count']:.0f}")
        print(f"  Recall:     {results['soma']['recall_mean']:.4f} ± {results['soma']['recall_std']:.4f}")
        print(f"  (Proportion of GT soma pixels captured by prediction)")
    
    # Branches metrics
    if 'branches' in results:
        print("\nBRANCHES (Arms/Processes) - FULL METRICS:")
        print(f"  Samples:    {results['branches']['sample_count']}")
        print(f"  Avg Pixels: {results['branches']['avg_pixel_count']:.0f}")
        print(f"  Dice:       {results['branches']['dice_mean']:.4f} ± {results['branches']['dice_std']:.4f}")
        print(f"  IoU:        {results['branches']['iou_mean']:.4f} ± {results['branches']['iou_std']:.4f}")
        print(f"  Precision:  {results['branches']['precision_mean']:.4f} ± {results['branches']['precision_std']:.4f}")
        print(f"  Recall:     {results['branches']['recall_mean']:.4f} ± {results['branches']['recall_std']:.4f}")
        print(f"  (Predictions outside soma region vs GT branches)")
    
    print("\n" + "="*70 + "\n")


def evaluate_model_with_regions(model, data_df, device='cuda', threshold=0.5, 
                                has_soma_masks=True, img_size=256):
    """
    Evaluate a trained segmentation model on a test set with region-based analysis.
    
    This function loads paired whole cell and soma masks, runs model inference,
    and computes metrics separately for soma and branches.
    
    Args:
        model: Trained PyTorch model (should output [B, 1, H, W])
        data_df: DataFrame from pair_whole_and_soma_masks() with columns:
                 ['scan', 'image_path', 'whole_mask_path', 'soma_mask_path']
        device: Device to run inference on ('cuda' or 'cpu')
        threshold: Threshold for binarizing predictions (default: 0.5)
        has_soma_masks: Whether to compute region-specific metrics (default: True)
        img_size: Input size for model (default: 256)
        
    Returns:
        Dictionary with aggregated metrics from aggregate_metrics()
    """
    from PIL import Image
    import torchvision.transforms.functional as TF
    
    model.eval()
    model.to(device)
    
    all_metrics = []
    
    with torch.no_grad():
        for idx, row in data_df.iterrows():
            # Load image
            img = Image.open(row['image_path']).convert('RGB')
            img_tensor = TF.to_tensor(img).unsqueeze(0).to(device)  # [1, 3, H, W]
            
            # Resize if needed
            if img_tensor.shape[2] != img_size or img_tensor.shape[3] != img_size:
                img_tensor = TF.resize(img_tensor, [img_size, img_size])
            
            # Run inference
            pred = model(img_tensor)  # [1, 1, H, W]
            
            # Apply sigmoid if output is logits
            if pred.min() < 0 or pred.max() > 1:
                pred = torch.sigmoid(pred)
            
            pred = pred.squeeze(0).squeeze(0)  # [H, W]
            
            # Load ground truth masks
            whole_mask = Image.open(row['whole_mask_path']).convert('L')
            whole_mask_tensor = TF.to_tensor(whole_mask).squeeze(0)  # [H, W]
            
            # Resize if needed
            if whole_mask_tensor.shape[0] != img_size or whole_mask_tensor.shape[1] != img_size:
                whole_mask_tensor = TF.resize(whole_mask_tensor.unsqueeze(0), [img_size, img_size]).squeeze(0)
            
            # Load soma mask if available
            soma_mask_tensor = None
            if has_soma_masks and row['soma_mask_path'] is not None:
                soma_mask = Image.open(row['soma_mask_path']).convert('L')
                soma_mask_tensor = TF.to_tensor(soma_mask).squeeze(0)  # [H, W]
                
                # Resize if needed
                if soma_mask_tensor.shape[0] != img_size or soma_mask_tensor.shape[1] != img_size:
                    soma_mask_tensor = TF.resize(soma_mask_tensor.unsqueeze(0), [img_size, img_size]).squeeze(0)
            
            # Evaluate with region-based metrics
            metrics = evaluate_regions(
                pred=pred,
                whole_cell_mask=whole_mask_tensor,
                soma_mask=soma_mask_tensor,
                threshold=threshold
            )
            
            all_metrics.append(metrics)
            
            # Print progress every 50 samples
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{len(data_df)} samples...")
    
    # Aggregate metrics
    results = aggregate_metrics(all_metrics)
    
    return results