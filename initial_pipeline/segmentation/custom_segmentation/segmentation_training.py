from data_utils import index_segmentations_df, pair_whole_and_soma_masks
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
from training.betti_loss import BettiMatchingLoss
from training.evaluation import (
    evaluate_regions, 
    aggregate_metrics, 
    print_metrics_summary,
    dice_score,
    iou_score,
    morphology_similarity_score
)

# Setup imports from initial_pipeline (for morphology features)
from setup_imports import setup_initial_pipeline_path
setup_initial_pipeline_path()



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


@torch.no_grad()
def evaluate_test_with_regions(model, test_df, device, training_data_dir, img_size=256, threshold=0.5):
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
        
    Returns:
        Dictionary with aggregated metrics from aggregate_metrics()
    """
    import torchvision.transforms.functional as TF
    
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
    print("\nREGION-BASED EVALUATION ON TEST SET")
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

def main():
    args = parse_segmentation_args()

    disconnect_components = args.disconnect_components
    add_new_components = args.add_new_components

    set_seed(42)
    # print('preparing segmentations')
    # preprocess_segmentations()
    # print('segmentations ready')

    SEGMENTATIONS_DIR = Path.cwd().parents[2] / "AnnotationsData_Adjusted_WithSoma" / "Segmentations"

    
    path_df = index_segmentations_df(
        SEGMENTATIONS_DIR,
        mask_name = 'masks'
    )

    # mask_quality_df = pd.read_csv('mask_quality_summary.csv')
    # mask_quality_df = mask_quality_df[['image_path', 'mask_quality']]


    # mask_quality_df["image_path"] = mask_quality_df["image_path"].apply(Path)

    # path_df = path_df.merge(mask_quality_df, on = 'image_path', how = 'inner')
    # print(path_df.columns)

    # Only segment the entire cell ie. not just the SOMA
    path_df = path_df[path_df['class'] == 'MG_whole']
    # # Remove all bad annotations
    # path_df = path_df[~path_df['mask_quality'].isin(['bad', 'bad_image_quality', 'disagree'])]



    # path_df = path_df.merg(mask_quality_df, on = )
    # Only take first 160 for now to train fast
    # path_df = path_df.head(80)
    print(len(path_df), "annotation pairs found")

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

        batch_size = 16
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

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
            else:
                print("\nRegion-based evaluation returned None (no soma masks found)")
        except Exception as e:
            print(f"\n!!! ERROR in region-based evaluation: {e}")
            print(f"!!! Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            print("!!! Continuing to next fold...\n")


    dice_scores = [m["dice"] for m in all_fold_metrics]
    iou_scores  = [m["iou"]  for m in all_fold_metrics]
    morphology_scores = [m["morphology"] for m in all_fold_metrics]

    print("\nCross-validation results")
    print(f"Dice:       {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
    print(f"IoU:        {np.mean(iou_scores):.4f} ± {np.std(iou_scores):.4f}")
    print(f"Morphology: {np.mean(morphology_scores):.4f} ± {np.std(morphology_scores):.4f}")


if __name__ == "__main__":
    main()