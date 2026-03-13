from data_utils import index_segmentations_df, pair_whole_and_soma_masks
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold, train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import os
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms.functional as TF
from segmentation_preprocessing import preprocess_segmentations
from random_disconnections.disconnect_components import disconnect_branches_with_gap
from random_disconnections.find_connection_points import find_random_branch_points
from random_disconnections.random_colour_noise import add_floating_synthetic_fragments
from training.training_utils import set_seed, parse_segmentation_args, save_model_with_timestamp
from training.segmentation_dataset import SegmentationDataset
from training.unet import UNet
from training.cl_dice import soft_cldice_loss, dice_loss
from training.betti_loss import BettiMatchingLoss
from training.evaluation import (
    evaluate_regions, 
    aggregate_metrics, 
    print_metrics_summary,
    dice_score,
    iou_score,
    morphology_similarity_score,
    morphology_similarity_score_detailed
)
from crf import connect_components_adaptive, connect_components_probability_based
from training.printing_utils import print_all_cross_validation_results, print_morphology_feature_summary

# Setup imports from initial_pipeline (for morphology features)
from setup_imports import setup_initial_pipeline_path
setup_initial_pipeline_path()


def soma_collate_fn(batch):
    """Custom collate for SegmentationDataset with include_soma=True.

    Each item is a 3-tuple (image, mask, soma_tensor) where soma_tensor may
    be None for cells that have no paired soma mask.  PyTorch's default collate
    cannot stack a mix of tensors and None, so we handle the third element
    manually: tensors are stacked into a batch tensor; Nones are left as a
    plain Python list so the training/evaluation loop can detect them.
    """
    from torch.utils.data.dataloader import default_collate

    if len(batch[0]) == 3:
        images  = default_collate([item[0] for item in batch])
        masks   = default_collate([item[1] for item in batch])
        # soma tensors: stack if all present, else keep as list
        soma_items = [item[2] for item in batch]
        if all(s is not None for s in soma_items):
            somas = default_collate(soma_items)
        else:
            somas = soma_items  # list of (tensor | None)
        return images, masks, somas
    else:
        return default_collate(batch)


class CombinedLoss(nn.Module):
    """
    Flexible loss wrapper supporting multiple loss types:
    - 'bce': Binary Cross Entropy with Logits
    - 'dice': Dice Loss
    - 'cldice': Centerline Dice Loss (topology-aware)
    - 'betti': Betti Matching Loss (component/hole counting)
    - 'bce_cldice': Weighted combination of BCE and clDice
    - 'dice_cldice': Weighted combination of Dice and clDice
    - 'bce_cldice_betti': Weighted combination of BCE, clDice, and Betti
    - 'dice_cldice_betti': Weighted combination of Dice, clDice, and Betti
    """
    def __init__(self, loss_type='bce', alpha=0.5, beta=0.3, 
                 betti_b0_weight=1.0, betti_b1_weight=0.5):
        """
        Args:
            loss_type: One of ['bce', 'dice', 'cldice', 'betti', 
                               'bce_cldice', 'dice_cldice',
                               'bce_cldice_betti', 'dice_cldice_betti']
            alpha: Weight for clDice in combined losses (0.0-1.0)
            beta: Weight for Betti loss in triple combinations (0.0-1.0)
            betti_b0_weight: Weight for component count (β0) in Betti loss
            betti_b1_weight: Weight for hole count (β1) in Betti loss
            
        For triple combinations (bce_cldice_betti, dice_cldice_betti):
            loss = (1-alpha-beta)*base_loss + alpha*cldice_loss + beta*betti_loss
        """
        super().__init__()
        self.loss_type = loss_type
        self.alpha = alpha
        self.beta = beta
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.betti_loss = BettiMatchingLoss(
            beta_0_weight=betti_b0_weight,
            beta_1_weight=betti_b1_weight,
            soft=False  # Use hard version for now
        )
        
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
            return dice_loss(probs, target).mean()
        
        elif self.loss_type == 'cldice':
            # Pure Centerline Dice Loss
            return soft_cldice_loss(probs, target).mean()
        
        elif self.loss_type == 'betti':
            # Pure Betti Matching Loss
            return self.betti_loss(probs, target)
        
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
        
        elif self.loss_type == 'bce_cldice_betti':
            # Triple combination: BCE + clDice + Betti
            bce = self.bce_loss(logits, target)
            cldice = soft_cldice_loss(probs, target).mean()
            betti = self.betti_loss(probs, target)
            return (1 - self.alpha - self.beta) * bce + self.alpha * cldice + self.beta * betti
        
        elif self.loss_type == 'dice_cldice_betti':
            # Triple combination: Dice + clDice + Betti
            dice = dice_loss(probs, target).mean()
            cldice = soft_cldice_loss(probs, target).mean()
            betti = self.betti_loss(probs, target)
            return (1 - self.alpha - self.beta) * dice + self.alpha * cldice + self.beta * betti
        
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
def evaluate(model, loader, device, calculate_morphology_metric=False, apply_postprocessing=False, postprocessing_type='adaptive'):
    """
    Evaluate model on a dataset.
    
    Args:
        model: The segmentation model
        loader: DataLoader for the dataset.  Each batch may be either:
                  - (image, mask)                  – standard 2-tuple
                  - (image, mask, soma_mask)        – 3-tuple with soma masks
                When calculate_morphology_metric=True a 3-tuple loader is expected
                so that the ground-truth soma can be passed to morphology_similarity_score
                and the predicted soma can be derived via the Gaussian filter from the image.
        device: Device to run evaluation on
        calculate_morphology_metric: If True, compute morphology similarity score.
                                     Default False since it's computationally expensive.
        apply_postprocessing: If True, apply postprocessing to predictions
                             before computing metrics (mimics inference pipeline).
                             Default False.
        postprocessing_type: Type of postprocessing to apply ('adaptive' or 'probability').
                            Only used if apply_postprocessing=True.
                            - 'adaptive': Uses connect_components_adaptive (color/texture based)
                            - 'probability': Uses connect_components_probability_based (UNet confidence based)
                            Default 'adaptive'.
    
    Returns:
        Dictionary with dice, iou, and optionally morphology scores
    """
    model.eval()
    dice_list, iou_list = [], []
    morphology_list = [] if calculate_morphology_metric else None

    for batch in loader:
        # Auto-detect whether the loader yields soma masks as a 3rd element
        if len(batch) == 3:
            x, y, soma_batch = batch
        else:
            x, y = batch
            soma_batch = None

        x, y = x.to(device), y.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits)
        
        batch_size = probs.shape[0]
        
        # Process each sample in the batch
        for i in range(batch_size):
            pred = probs[i, 0]  # [H, W]
            target = y[i, 0]    # [H, W]

            # Always extract the image as uint8 numpy (needed for Gaussian soma)
            img_np = (x[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            
            # Apply postprocessing if requested
            if apply_postprocessing:
                if postprocessing_type == 'adaptive':
                    # Traditional: color/texture based
                    pred_np = (pred > 0.5).cpu().numpy().astype(np.uint8)
                    connected_mask = connect_components_adaptive(
                        pred_np, 
                        img_np, 
                        min_component_frac=0.0
                    )
                elif postprocessing_type == 'probability':
                    # Probability-based: uses UNet confidence
                    prob_map = pred.cpu().numpy().astype(np.float32)
                    connected_mask = connect_components_probability_based(
                        prob_map=prob_map,
                        image=img_np,
                        threshold=0.5,
                        min_component_frac=0.0,
                    )
                else:
                    raise ValueError(f"Unknown postprocessing_type: {postprocessing_type}. Must be 'adaptive' or 'probability'")
                
                # Convert back to tensor
                pred = torch.from_numpy(connected_mask).float().to(device)
            
            # Compute metrics
            dice_list.append(dice_score(pred.unsqueeze(0), target.unsqueeze(0)).item())
            iou_list.append(iou_score(pred.unsqueeze(0), target.unsqueeze(0)).item())
            
            # Morphology-level metric (only if requested)
            if calculate_morphology_metric:
                # Extract per-sample soma mask (may be None if unavailable)
                soma_mask_i = None
                if soma_batch is not None:
                    s = soma_batch[i]
                    if s is not None:
                        # soma_batch items are (1, H, W) tensors or None
                        soma_mask_i = s[0] if s.dim() == 3 else s
                morph_score = morphology_similarity_score(
                    pred, target,
                    soma_mask=soma_mask_i,
                    image_rgb=img_np,
                )
                morphology_list.append(morph_score)

    results = {
        "dice": sum(dice_list) / len(dice_list),
        "iou": sum(iou_list) / len(iou_list),
    }
    
    if calculate_morphology_metric:
        results["morphology"] = sum(morphology_list) / len(morphology_list)
    
    return results


@torch.no_grad()
def evaluate_test_with_regions(model, test_df, device, training_data_dir, img_size=256, threshold=0.5, apply_postprocessing=False, postprocessing_type='adaptive'):
    """
    Evaluate model on test set with region-based metrics (soma vs branches).
    
    This function loads the test data with paired soma masks from the WithSoma dataset,
    runs inference, and computes metrics separately for soma and branches regions.
    
    Args:
        model: Trained segmentation model
        test_df: Test DataFrame with image paths to evaluate
        device: Device to run inference on
        training_data_dir: The directory used for training (to auto-detect WithSoma usage)
        img_size: Image size for model input (default: 256)
        threshold: Threshold for binarizing predictions (default: 0.5)
        apply_postprocessing: If True, apply postprocessing to predictions
                             before computing metrics (mimics inference pipeline).
                             Default False.
        postprocessing_type: Type of postprocessing to apply ('adaptive' or 'probability').
                            Only used if apply_postprocessing=True.
                            - 'adaptive': Uses connect_components_adaptive (color/texture based)
                            - 'probability': Uses connect_components_probability_based (UNet confidence based)
                            Default 'adaptive'.
        
    Returns:
        Dictionary with aggregated metrics from aggregate_metrics()
    """
    model.eval()
    
    # Get unique image paths from test set
    test_image_paths = set(test_df['image_path'].tolist())
    
    # Load paired whole and soma masks
    paired_df = pair_whole_and_soma_masks(training_data_dir)
    
    # Filter to only test images (by exact image path match)
    test_paired_df = paired_df[paired_df['image_path'].isin(test_image_paths)].reset_index(drop=True)
    
    if len(test_paired_df) == 0:
        print(f"\nWarning: No samples found in dataset for test images")
        print(f"Test set has {len(test_df)} images")
        print(f"Paired dataset has {len(paired_df)} images")
        print("Skipping region-based evaluation.")
        return None
    
    soma_count = test_paired_df['soma_mask_path'].notna().sum()
    postprocessing_label = f" (WITH POSTPROCESSING: {postprocessing_type})" if apply_postprocessing else ""
    print(f"\nREGION-BASED EVALUATION ON TEST SET{postprocessing_label}")
    print(f"Test samples: {len(test_paired_df)} (expected: {len(test_df)})")
    print(f"Samples with soma masks: {soma_count}/{len(test_paired_df)}")
    
    all_metrics = []
    
    for idx, row in test_paired_df.iterrows():
        # Load image
        img = Image.open(row['image_path']).convert('RGB')
        img_tensor = TF.to_tensor(img).unsqueeze(0).to(device)  # [1, 3, H, W]
        
        # Verify size (should already be 512x512)
        if img_tensor.shape[2] != img_size or img_tensor.shape[3] != img_size:
            img_tensor = TF.resize(img_tensor, [img_size, img_size])
        
        # Run inference
        pred = model(img_tensor)  # [1, 1, H, W]
        
        # Apply sigmoid if output is logits
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)
        
        pred = pred.squeeze(0).squeeze(0)  # [H, W]
        
        # Apply postprocessing if requested
        if apply_postprocessing:
            img_np = (img_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            
            if postprocessing_type == 'adaptive':
                # Traditional: color/texture based
                pred_np = (pred > threshold).cpu().numpy().astype(np.uint8)
                connected_mask = connect_components_adaptive(
                    pred_np,
                    img_np,
                    min_component_frac=0.0
                )
            elif postprocessing_type == 'probability':
                # Probability-based: uses UNet confidence
                prob_map = pred.cpu().numpy().astype(np.float32)
                connected_mask = connect_components_probability_based(
                    prob_map=prob_map,
                    image=img_np,
                    threshold=threshold,
                    min_component_frac=0.0,
                )
            else:
                raise ValueError(f"Unknown postprocessing_type: {postprocessing_type}. Must be 'adaptive' or 'probability'")
            
            # Convert back to tensor
            pred = torch.from_numpy(connected_mask).float()
        
        # Load ground truth masks
        whole_mask = Image.open(row['whole_mask_path']).convert('L')
        whole_mask_tensor = TF.to_tensor(whole_mask).squeeze(0)  # [H, W]
        
        # Verify size (should already be 512x512)
        if whole_mask_tensor.shape[0] != img_size or whole_mask_tensor.shape[1] != img_size:
            whole_mask_tensor = TF.resize(whole_mask_tensor.unsqueeze(0), [img_size, img_size]).squeeze(0)
        
        # Load soma mask if available
        soma_mask_tensor = None
        if pd.notna(row['soma_mask_path']):
            soma_mask = Image.open(row['soma_mask_path']).convert('L')
            soma_mask_tensor = TF.to_tensor(soma_mask).squeeze(0)  # [H, W]
            
            # Verify size (should already be 512x512)
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
    
    # Aggregate metrics
    results = aggregate_metrics(all_metrics)
    
    return results


# Visualize predictions

@torch.no_grad()
def evaluate_morphology_breakdown(model, loader, device):
    """Run a morphology-only evaluation pass that collects per-cell feature breakdowns.

    This is called once at the very end of training (after all folds) to produce
    the detailed morphology feature summary.  It does NOT affect any training logic.

    Args:
        model: The trained segmentation model (should already have best weights loaded).
        loader: DataLoader for the test set.  Expected to yield 3-tuples
                (image, mask, soma_mask) as produced when ``include_soma=True``.
        device: Torch device.

    Returns:
        List of per-cell breakdown dicts (one per cell in the loader), each as
        returned by ``morphology_similarity_score_detailed``.
    """
    model.eval()
    cell_breakdowns = []

    for batch in loader:
        if len(batch) == 3:
            x, y, soma_batch = batch
        else:
            x, y = batch
            soma_batch = None

        x, y = x.to(device), y.to(device)
        logits = model(x)
        probs  = torch.sigmoid(logits)

        for i in range(probs.shape[0]):
            pred   = probs[i, 0]
            target = y[i, 0]
            img_np = (x[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            soma_mask_i = None
            if soma_batch is not None:
                s = soma_batch[i]
                if s is not None:
                    soma_mask_i = s[0] if s.dim() == 3 else s

            _, breakdown = morphology_similarity_score_detailed(
                pred, target,
                soma_mask=soma_mask_i,
                image_rgb=img_np,
            )
            if breakdown:
                cell_breakdowns.append(breakdown)

    return cell_breakdowns


def main():
    args = parse_segmentation_args()

    disconnect_components = args.disconnect_components
    add_new_components = args.add_new_components

    set_seed(42)
    # print('preparing segmentations')
    # preprocess_segmentations()
    # print('segmentations ready')

    SEGMENTATIONS_DIR = Path.cwd().parents[3] / "SegmentationDatasets" / "AnnotationsData_Adjusted_WithSoma" / "Segmentations"

    
    path_df = index_segmentations_df(
        SEGMENTATIONS_DIR,
        mask_name = 'masks'
    )
    # Only segment the entire cell ie. not just the SOMA
    path_df = path_df[path_df['class'] == 'MG_whole']

    # Merge soma_mask_path so that val/test datasets can load GT soma masks
    # for morphology metric evaluation.
    paired_df = pair_whole_and_soma_masks(SEGMENTATIONS_DIR, mask_name='masks')
    # paired_df uses 'whole_mask_path' for the cell mask; path_df uses 'mask_path'.
    # They refer to the same file, so join on image_path.
    path_df = path_df.merge(
        paired_df[['image_path', 'soma_mask_path']],
        on='image_path',
        how='left'
    )

    # Only take first 160 for now to train fast
    # path_df = path_df.head(256)
    print(len(path_df), "annotation pairs found")
    soma_count = path_df['soma_mask_path'].notna().sum()
    print(f"  ({soma_count}/{len(path_df)} have paired soma masks)")

    k_folds = 5
    kf = KFold(
        n_splits=k_folds,
        shuffle=True,
        random_state=42
    )
    all_fold_metrics = []
    all_fold_region_metrics = []
    all_fold_postprocessed_metrics = []
    all_fold_postprocessed_region_metrics = []
    all_fold_postprocessed_probability_metrics = []
    all_fold_postprocessed_probability_region_metrics = []
    all_fold_morphology_breakdowns = []  # per-fold list of per-cell feature breakdowns

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
            add_new_components=False,
            include_soma=True,
        )

        test_ds = SegmentationDataset(
            test_df, test_tfms,
            disconnect_components=False,
            add_new_components=False,
            include_soma=True,
        )

        batch_size = 16
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=soma_collate_fn)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=soma_collate_fn)

        # Setup loss function based on arguments
        criterion = CombinedLoss(loss_type=args.loss_type, alpha=args.cldice_alpha)
        print(f'Using loss function: {args.loss_type}')
        if args.loss_type in ['bce_cldice', 'dice_cldice']:
            print(f'  clDice weight (alpha): {args.cldice_alpha}')
    
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f'currently using device: {device}')

        model = UNet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        max_epochs = 100
        patience = 10

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

        # Collect per-feature morphology breakdown for the post-training summary
        try:
            fold_breakdowns = evaluate_morphology_breakdown(model, test_loader, device)
            all_fold_morphology_breakdowns.append(fold_breakdowns)
        except Exception as e:
            print(f"\n!!! WARNING: Could not collect morphology feature breakdown for fold {fold + 1}: {e}")


        print(
            f"Fold {fold + 1} final test | "
            f"Dice: {final_test_metrics['dice']:.4f}, "
            f"IoU: {final_test_metrics['iou']:.4f}, "
            f"Morphology: {final_test_metrics['morphology']:.4f}"
        )
        
        # Region-based evaluation (soma vs branches) on test set
        try:
            region_results = evaluate_test_with_regions(
                model=model,
                test_df=test_df,
                device=device,
                training_data_dir=SEGMENTATIONS_DIR,
                img_size=256,
                threshold=0.5
            )
            
            if region_results is not None:
                print_metrics_summary(region_results)
                all_fold_region_metrics.append(region_results)
            else:
                print("\nRegion-based evaluation returned None (no soma masks found)")
        except Exception as e:
            print(f"\n!!! ERROR in region-based evaluation: {e}")
            print(f"!!! Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            print("!!! Continuing to next fold...\n")
        
        # ========================================================================
        # POSTPROCESSED EVALUATION (ADAPTIVE - Color/Texture Based)
        # ========================================================================
        print("\n" + "="*70)
        print("EVALUATING WITH POSTPROCESSING: ADAPTIVE (Color/Texture Based)")
        print("="*70)
        
        # Standard metrics with postprocessing
        try:
            postprocessed_metrics = evaluate(
                model, test_loader, device, 
                calculate_morphology_metric=True,
                apply_postprocessing=True,
                postprocessing_type='adaptive'
            )
            all_fold_postprocessed_metrics.append(postprocessed_metrics)
            
            print(
                f"Fold {fold + 1} postprocessed (adaptive) test | "
                f"Dice: {postprocessed_metrics['dice']:.4f}, "
                f"IoU: {postprocessed_metrics['iou']:.4f}, "
                f"Morphology: {postprocessed_metrics['morphology']:.4f}"
            )
        except Exception as e:
            print(f"\n!!! ERROR in postprocessed evaluation: {e}")
            print(f"!!! Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            print("!!! Continuing to next fold...\n")
        
        # Region-based evaluation with postprocessing
        try:
            postprocessed_region_results = evaluate_test_with_regions(
                model=model,
                test_df=test_df,
                device=device,
                training_data_dir=SEGMENTATIONS_DIR,
                img_size=256,
                threshold=0.5,
                apply_postprocessing=True,
                postprocessing_type='adaptive'
            )
            
            if postprocessed_region_results is not None:
                print_metrics_summary(postprocessed_region_results)
                all_fold_postprocessed_region_metrics.append(postprocessed_region_results)
            else:
                print("\nPostprocessed region-based evaluation returned None (no soma masks found)")
        except Exception as e:
            print(f"\n!!! ERROR in postprocessed region-based evaluation: {e}")
            print(f"!!! Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            print("!!! Continuing to next fold...\n")
        
        # ========================================================================
        # POSTPROCESSED EVALUATION (PROBABILITY - UNet Confidence Based)
        # ========================================================================
        print("\n" + "="*70)
        print("EVALUATING WITH POSTPROCESSING: PROBABILITY (UNet Confidence Based)")
        print("="*70)
        
        # Standard metrics with probability-based postprocessing
        try:
            postprocessed_prob_metrics = evaluate(
                model, test_loader, device, 
                calculate_morphology_metric=True,
                apply_postprocessing=True,
                postprocessing_type='probability'
            )
            all_fold_postprocessed_probability_metrics.append(postprocessed_prob_metrics)
            
            print(
                f"Fold {fold + 1} postprocessed (probability) test | "
                f"Dice: {postprocessed_prob_metrics['dice']:.4f}, "
                f"IoU: {postprocessed_prob_metrics['iou']:.4f}, "
                f"Morphology: {postprocessed_prob_metrics['morphology']:.4f}"
            )
        except Exception as e:
            print(f"\n!!! ERROR in probability-based postprocessed evaluation: {e}")
            print(f"!!! Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            print("!!! Continuing to next fold...\n")
        
        # Region-based evaluation with probability-based postprocessing
        try:
            postprocessed_prob_region_results = evaluate_test_with_regions(
                model=model,
                test_df=test_df,
                device=device,
                training_data_dir=SEGMENTATIONS_DIR,
                img_size=256,
                threshold=0.5,
                apply_postprocessing=True,
                postprocessing_type='probability'
            )
            
            if postprocessed_prob_region_results is not None:
                print_metrics_summary(postprocessed_prob_region_results)
                all_fold_postprocessed_probability_region_metrics.append(postprocessed_prob_region_results)
            else:
                print("\nProbability-based postprocessed region-based evaluation returned None (no soma masks found)")
        except Exception as e:
            print(f"\n!!! ERROR in probability-based postprocessed region-based evaluation: {e}")
            print(f"!!! Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            print("!!! Continuing to next fold...\n")
        
        print("="*70 + "\n")


    # Print all cross-validation results using the extracted printing utilities
    print_all_cross_validation_results(
        all_fold_metrics=all_fold_metrics,
        all_fold_region_metrics=all_fold_region_metrics,
        all_fold_postprocessed_metrics=all_fold_postprocessed_metrics,
        all_fold_postprocessed_region_metrics=all_fold_postprocessed_region_metrics,
        all_fold_postprocessed_probability_metrics=all_fold_postprocessed_probability_metrics,
        all_fold_postprocessed_probability_region_metrics=all_fold_postprocessed_probability_region_metrics
    )

    # Print morphology feature summary: per-feature % performance, weight, and contribution
    if all_fold_morphology_breakdowns:
        print_morphology_feature_summary(
            all_fold_feature_breakdowns=all_fold_morphology_breakdowns,
            label="NO POSTPROCESSING",
        )



if __name__ == "__main__":
    main()