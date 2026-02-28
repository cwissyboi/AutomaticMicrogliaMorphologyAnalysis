"""
Debug script for region-based evaluation - UPDATED VERSION.

This script loads a trained model and evaluates it on a few samples,
visualizing all intermediate masks and metrics to verify the fix.
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

# Import local modules
from data_utils import pair_whole_and_soma_masks
from training.unet import UNet
from training.evaluation import evaluate_regions


def visualize_masks_and_metrics(
    image_path,
    whole_mask_path,
    soma_mask_path,
    model,
    device,
    img_size=256,
    threshold=0.5,
    save_dir=None
):
    """
    Visualize all masks and compute metrics for a single sample with detailed debugging.
    
    Args:
        image_path: Path to input image
        whole_mask_path: Path to MG_whole mask
        soma_mask_path: Path to MG_cell_body mask
        model: Trained model
        device: Device to run on
        img_size: Image size for model
        threshold: Prediction threshold
        save_dir: Directory to save visualizations
    """
    print("\n" + "="*80)
    print(f"DEBUGGING SAMPLE: {Path(image_path).name}")
    print("="*80)
    
    # ============================================================================
    # STEP 1: Load and preprocess image
    # ============================================================================
    print("\n[STEP 1] Loading and preprocessing image...")
    img = Image.open(image_path).convert('RGB')
    print(f"  - Original image size: {img.size}")
    
    img_tensor = TF.to_tensor(img).unsqueeze(0).to(device)  # [1, 3, H, W]
    print(f"  - Image tensor shape: {img_tensor.shape}")
    
    # Resize if needed
    if img_tensor.shape[2] != img_size or img_tensor.shape[3] != img_size:
        img_tensor = TF.resize(img_tensor, [img_size, img_size])
        print(f"  - Resized to: {img_tensor.shape}")
    
    # ============================================================================
    # STEP 2: Run model inference
    # ============================================================================
    print("\n[STEP 2] Running model inference...")
    with torch.no_grad():
        pred = model(img_tensor)  # [1, 1, H, W]
    
    print(f"  - Raw prediction shape: {pred.shape}")
    print(f"  - Raw prediction range: [{pred.min().item():.4f}, {pred.max().item():.4f}]")
    
    # Apply sigmoid if output is logits
    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)
        print(f"  - Applied sigmoid")
        print(f"  - After sigmoid range: [{pred.min().item():.4f}, {pred.max().item():.4f}]")
    
    pred = pred.squeeze(0).squeeze(0)  # [H, W]
    print(f"  - Final prediction shape: {pred.shape}")
    
    # Binarize prediction
    pred_binary = (pred > threshold).float()
    print(f"  - Threshold: {threshold}")
    print(f"  - Predicted pixels (after threshold): {pred_binary.sum().item():.0f}")
    
    # ============================================================================
    # STEP 3: Load ground truth masks
    # ============================================================================
    print("\n[STEP 3] Loading ground truth masks...")
    
    # Load whole cell mask
    whole_mask = Image.open(whole_mask_path).convert('L')
    print(f"  - Whole mask original size: {whole_mask.size}")
    whole_mask_tensor = TF.to_tensor(whole_mask).squeeze(0)  # [H, W]
    
    # Resize if needed
    if whole_mask_tensor.shape[0] != img_size or whole_mask_tensor.shape[1] != img_size:
        whole_mask_tensor = TF.resize(whole_mask_tensor.unsqueeze(0), [img_size, img_size]).squeeze(0)
        print(f"  - Resized whole mask to: {whole_mask_tensor.shape}")
    
    whole_mask_binary = (whole_mask_tensor > 0.5).float()
    print(f"  - Whole mask pixels: {whole_mask_binary.sum().item():.0f}")
    
    # Load soma mask
    soma_mask_tensor = None
    soma_mask_binary = None
    branches_mask_binary = None
    
    if soma_mask_path is not None:
        soma_mask = Image.open(soma_mask_path).convert('L')
        print(f"  - Soma mask original size: {soma_mask.size}")
        soma_mask_tensor = TF.to_tensor(soma_mask).squeeze(0)  # [H, W]
        
        # Resize if needed
        if soma_mask_tensor.shape[0] != img_size or soma_mask_tensor.shape[1] != img_size:
            soma_mask_tensor = TF.resize(soma_mask_tensor.unsqueeze(0), [img_size, img_size]).squeeze(0)
            print(f"  - Resized soma mask to: {soma_mask_tensor.shape}")
        
        soma_mask_binary = (soma_mask_tensor > 0.5).float()
        print(f"  - Soma mask pixels: {soma_mask_binary.sum().item():.0f}")
        
        # ============================================================================
        # STEP 4: Create branches mask
        # ============================================================================
        print("\n[STEP 4] Creating branches mask...")
        print("  - Formula: branches = whole_cell * (1 - soma)")
        branches_mask_binary = whole_mask_binary * (1 - soma_mask_binary)
        print(f"  - Branches mask pixels: {branches_mask_binary.sum().item():.0f}")
        
        # Verify partition
        overlap = (soma_mask_binary * branches_mask_binary).sum().item()
        total = soma_mask_binary.sum().item() + branches_mask_binary.sum().item()
        whole_total = whole_mask_binary.sum().item()
        print(f"\n  PARTITION VERIFICATION:")
        print(f"  - Soma pixels: {soma_mask_binary.sum().item():.0f}")
        print(f"  - Branches pixels: {branches_mask_binary.sum().item():.0f}")
        print(f"  - Soma + Branches: {total:.0f}")
        print(f"  - Whole cell pixels: {whole_total:.0f}")
        print(f"  - Difference: {abs(total - whole_total):.0f}")
        print(f"  - Overlap (should be 0): {overlap:.0f}")
        
        # ============================================================================
        # STEP 5: Compute region metrics (NEW LOGIC)
        # ============================================================================
        print("\n[STEP 5] Computing region metrics (NEW CORRECTED LOGIC)...")
        
        # --- SOMA: Recall only ---
        print("\n  === SOMA METRICS (Recall Only) ===")
        print("  - We compute: (pred ∩ soma) / soma")
        print("  - This tells us: what % of GT soma was captured")
        
        soma_intersection = (pred_binary * soma_mask_binary).sum().item()
        soma_recall = soma_intersection / soma_mask_binary.sum().item()
        
        print(f"  - GT soma pixels: {soma_mask_binary.sum().item():.0f}")
        print(f"  - Predicted pixels in soma region: {soma_intersection:.0f}")
        print(f"  - Soma Recall: {soma_recall:.4f}")
        
        # --- BRANCHES: Full metrics ---
        print("\n  === BRANCHES METRICS (Full: Dice/IoU/Precision/Recall) ===")
        print("  - Step 1: Get predictions outside soma region")
        print("  - Formula: pred_branches = pred * (1 - soma)")
        print("  - NOTE: We DON'T mask by whole cell - predictions outside cell count as FP!")
        
        pred_branches = pred_binary * (1 - soma_mask_binary)
        
        print(f"  - Total predictions: {pred_binary.sum().item():.0f}")
        print(f"  - Predictions in branches region (outside soma): {pred_branches.sum().item():.0f}")
        print(f"  - GT branches pixels: {branches_mask_binary.sum().item():.0f}")
        
        # Compute full metrics for branches
        print("\n  - Step 2: Compute metrics comparing pred_branches vs GT branches")
        
        pred_branches_flat = pred_branches.flatten()
        branches_gt_flat = branches_mask_binary.flatten()
        
        tp = (pred_branches_flat * branches_gt_flat).sum().item()
        fp = (pred_branches_flat * (1 - branches_gt_flat)).sum().item()
        fn = ((1 - pred_branches_flat) * branches_gt_flat).sum().item()
        tn = ((1 - pred_branches_flat) * (1 - branches_gt_flat)).sum().item()
        
        epsilon = 1e-7
        branches_dice = (2 * tp + epsilon) / (2 * tp + fp + fn + epsilon)
        branches_iou = (tp + epsilon) / (tp + fp + fn + epsilon)
        branches_precision = (tp + epsilon) / (tp + fp + epsilon)
        branches_recall = (tp + epsilon) / (tp + fn + epsilon)
        
        print(f"\n  Confusion Matrix:")
        print(f"    TP (correct branches): {tp:.0f}")
        print(f"    FP (pred branches, actually not): {fp:.0f}")
        print(f"    FN (missed branches): {fn:.0f}")
        print(f"    TN (correctly not branches): {tn:.0f}")
        
        print(f"\n  Metrics:")
        print(f"    Dice:      {branches_dice:.4f}")
        print(f"    IoU:       {branches_iou:.4f}")
        print(f"    Precision: {branches_precision:.4f}")
        print(f"    Recall:    {branches_recall:.4f}")
    
    # ============================================================================
    # STEP 6: Compare with evaluate_regions function
    # ============================================================================
    print("\n[STEP 6] Comparing with evaluate_regions() function...")
    metrics = evaluate_regions(
        pred=pred,
        whole_cell_mask=whole_mask_tensor,
        soma_mask=soma_mask_tensor,
        threshold=threshold
    )
    
    print(f"\n  evaluate_regions() results:")
    print(f"    Overall Dice: {metrics.overall_dice:.4f}")
    print(f"    Overall IoU: {metrics.overall_iou:.4f}")
    if metrics.soma_metrics:
        print(f"    Soma Recall: {metrics.soma_metrics.recall:.4f}")
    if metrics.branches_metrics:
        print(f"    Branches Dice: {metrics.branches_metrics.dice:.4f}")
        print(f"    Branches IoU: {metrics.branches_metrics.iou:.4f}")
        print(f"    Branches Precision: {metrics.branches_metrics.precision:.4f}")
        print(f"    Branches Recall: {metrics.branches_metrics.recall:.4f}")
    
    print(f"\n  VERIFICATION:")
    if metrics.soma_metrics:
        match = abs(soma_recall - metrics.soma_metrics.recall) < 0.001
        print(f"    Soma Recall match: {match} (diff: {abs(soma_recall - metrics.soma_metrics.recall):.6f})")
    if metrics.branches_metrics:
        dice_match = abs(branches_dice - metrics.branches_metrics.dice) < 0.001
        print(f"    Branches Dice match: {dice_match} (diff: {abs(branches_dice - metrics.branches_metrics.dice):.6f})")
    
    # ============================================================================
    # STEP 7: Visualize everything
    # ============================================================================
    print("\n[STEP 7] Creating visualizations...")
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    # Row 1: Input and predictions
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Input Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(pred.cpu().numpy(), cmap='hot', vmin=0, vmax=1)
    axes[0, 1].set_title(f'Prediction (continuous)\nRange: [{pred.min():.3f}, {pred.max():.3f}]')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(pred_binary.cpu().numpy(), cmap='gray')
    axes[0, 2].set_title(f'Prediction (binary, t={threshold})\nPixels: {pred_binary.sum():.0f}')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(whole_mask_binary.cpu().numpy(), cmap='gray')
    axes[0, 3].set_title(f'GT Whole Cell\nPixels: {whole_mask_binary.sum():.0f}')
    axes[0, 3].axis('off')
    
    # Row 2: Ground truth masks and partition
    if soma_mask_binary is not None:
        axes[1, 0].imshow(soma_mask_binary.cpu().numpy(), cmap='Blues')
        axes[1, 0].set_title(f'GT Soma\nPixels: {soma_mask_binary.sum():.0f}')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(branches_mask_binary.cpu().numpy(), cmap='Greens')
        axes[1, 1].set_title(f'GT Branches\nPixels: {branches_mask_binary.sum():.0f}')
        axes[1, 1].axis('off')
        
        # Verify partition visually
        combined = np.zeros((*soma_mask_binary.shape, 3))
        combined[:, :, 0] = branches_mask_binary.cpu().numpy()  # Red = branches
        combined[:, :, 2] = soma_mask_binary.cpu().numpy()      # Blue = soma
        axes[1, 2].imshow(combined)
        axes[1, 2].set_title('Partition Check\nRed=Branches, Blue=Soma')
        axes[1, 2].axis('off')
        
        axes[1, 3].imshow(whole_mask_binary.cpu().numpy(), cmap='gray')
        axes[1, 3].set_title(f'GT Whole (reference)\nPixels: {whole_mask_binary.sum():.0f}')
        axes[1, 3].axis('off')
    
    # Row 3: Region-specific predictions and overlays
    if soma_mask_binary is not None:
        # Soma intersection (for recall)
        soma_intersection_mask = pred_binary * soma_mask_binary
        axes[2, 0].imshow(soma_intersection_mask.cpu().numpy(), cmap='Blues')
        axes[2, 0].set_title(f'Pred ∩ Soma\nPixels: {soma_intersection_mask.sum():.0f}\nRecall: {soma_recall:.3f}')
        axes[2, 0].axis('off')
        
        # Branches prediction (pred outside soma)
        axes[2, 1].imshow(pred_branches.cpu().numpy(), cmap='Greens')
        axes[2, 1].set_title(f'Pred Branches\n(pred - soma)\nPixels: {pred_branches.sum():.0f}\nDice: {branches_dice:.3f}')
        axes[2, 1].axis('off')
        
        # Show overlap for soma
        overlay_soma = np.zeros((*soma_mask_binary.shape, 3))
        overlay_soma[:, :, 0] = soma_mask_binary.cpu().numpy()           # GT in red
        overlay_soma[:, :, 1] = soma_intersection_mask.cpu().numpy()     # Pred ∩ soma in green
        axes[2, 2].imshow(overlay_soma)
        axes[2, 2].set_title('Soma: GT(red) vs Captured(green)\nYellow=Overlap')
        axes[2, 2].axis('off')
        
        # Show overlap for branches
        overlay_branches = np.zeros((*branches_mask_binary.shape, 3))
        overlay_branches[:, :, 0] = branches_mask_binary.cpu().numpy()  # GT in red
        overlay_branches[:, :, 1] = pred_branches.cpu().numpy()         # Pred in green
        axes[2, 3].imshow(overlay_branches)
        axes[2, 3].set_title(f'Branches: GT(red) vs Pred(green)\nYellow=Overlap\nFP visible as green-only')
        axes[2, 3].axis('off')
    
    plt.suptitle(f'Debug: {Path(image_path).name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / f'debug_{Path(image_path).stem}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  - Saved visualization to: {save_path}")
    
    plt.show()
    
    return metrics


def main():
    """Main debug script."""
    
    # ============================================================================
    # Configuration
    # ============================================================================
    MODEL_PATH = "checkpoints/best_run_25_1.pth"
    DATA_DIR = Path.cwd().parents[2] / "AnnotationsData_Adjusted_WithSoma" / "Segmentations"
    NUM_SAMPLES = 5  # Number of samples to debug
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    IMG_SIZE = 512
    THRESHOLD = 0.5
    SAVE_DIR = Path("debug_output")
    
    print("="*80)
    print("REGION-BASED EVALUATION DEBUG SCRIPT - UPDATED")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Data directory: {DATA_DIR}")
    print(f"  Number of samples: {NUM_SAMPLES}")
    print(f"  Device: {DEVICE}")
    print(f"  Image size: {IMG_SIZE}")
    print(f"  Threshold: {THRESHOLD}")
    
    # Create output directory
    SAVE_DIR.mkdir(exist_ok=True)
    print(f"  Output directory: {SAVE_DIR}")
    
    # ============================================================================
    # Load model
    # ============================================================================
    print("\n" + "="*80)
    print("LOADING MODEL")
    print("="*80)
    
    model = UNet()
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("  - Loaded from checkpoint dict")
    else:
        model.load_state_dict(checkpoint)
        print("  - Loaded state dict directly")
    
    model.to(DEVICE)
    model.eval()
    print(f"  - Model loaded successfully on {DEVICE}")
    
    # ============================================================================
    # Load data
    # ============================================================================
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    paired_df = pair_whole_and_soma_masks(DATA_DIR)
    print(f"  - Total samples: {len(paired_df)}")
    print(f"  - Samples with soma masks: {paired_df['soma_mask_path'].notna().sum()}")
    
    # Select samples for debugging
    debug_samples = paired_df.head(NUM_SAMPLES)
    print(f"  - Selected {len(debug_samples)} samples for debugging")
    
    # ============================================================================
    # Debug each sample
    # ============================================================================
    all_metrics = []
    
    for idx, row in debug_samples.iterrows():
        metrics = visualize_masks_and_metrics(
            image_path=row['image_path'],
            whole_mask_path=row['whole_mask_path'],
            soma_mask_path=row['soma_mask_path'],
            model=model,
            device=DEVICE,
            img_size=IMG_SIZE,
            threshold=THRESHOLD,
            save_dir=SAVE_DIR
        )
        all_metrics.append(metrics)
    
    # ============================================================================
    # Aggregate results
    # ============================================================================
    print("\n" + "="*80)
    print("AGGREGATE RESULTS ACROSS DEBUG SAMPLES")
    print("="*80)
    
    overall_dice = [m.overall_dice for m in all_metrics]
    soma_recall = [m.soma_metrics.recall for m in all_metrics if m.soma_metrics]
    branches_dice = [m.branches_metrics.dice for m in all_metrics if m.branches_metrics]
    branches_recall = [m.branches_metrics.recall for m in all_metrics if m.branches_metrics]
    branches_precision = [m.branches_metrics.precision for m in all_metrics if m.branches_metrics]
    
    print(f"\nOverall Dice:        {np.mean(overall_dice):.4f} ± {np.std(overall_dice):.4f}")
    print(f"Soma Recall:         {np.mean(soma_recall):.4f} ± {np.std(soma_recall):.4f}")
    print(f"Branches Dice:       {np.mean(branches_dice):.4f} ± {np.std(branches_dice):.4f}")
    print(f"Branches Precision:  {np.mean(branches_precision):.4f} ± {np.std(branches_precision):.4f}")
    print(f"Branches Recall:     {np.mean(branches_recall):.4f} ± {np.std(branches_recall):.4f}")
    
    print(f"\n{'='*80}")
    print("DEBUG COMPLETE!")
    print(f"Check the '{SAVE_DIR}' directory for visualizations")
    print("Look for green-only pixels in the bottom-right plot - these are FPs!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
