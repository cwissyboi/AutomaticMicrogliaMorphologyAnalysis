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
def evaluate(model, loader, device):
    model.eval()
    dice_list, iou_list = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits)

        dice_list.append(dice_score(probs, y).item())
        iou_list.append(iou_score(probs, y).item())

    return {
        "dice": sum(dice_list) / len(dice_list),
        "iou": sum(iou_list) / len(iou_list),
    }


# Visualize predictions

def main(): 

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
            disconnect_components=False,
            add_new_components=False
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

    
        criterion = nn.BCEWithLogitsLoss()
    
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
            test_metrics = evaluate(model, test_loader, device = device)
            val_metrics = evaluate(model, val_loader, device = device)

            current_dice = val_metrics["dice"]

            # Check improvement
            if current_dice > best_dice:
                best_dice = current_dice
                best_epoch = epoch
                epochs_no_improve = 0

                # 🔹 store BEST model weights in memory
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

        final_test_metrics = evaluate(model, test_loader, device)
        all_fold_metrics.append(final_test_metrics)


        print(
            f"Fold {fold + 1} final test | "
            f"Dice: {final_test_metrics['dice']:.4f}, "
            f"IoU: {final_test_metrics['iou']:.4f}"
        )

    dice_scores = [m["dice"] for m in all_fold_metrics]
    iou_scores  = [m["iou"]  for m in all_fold_metrics]

    print("\n========== Cross-validation results ==========")
    print(f"Dice: {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
    print(f"IoU:  {np.mean(iou_scores):.4f} ± {np.std(iou_scores):.4f}")


if __name__ == "__main__":
    main()