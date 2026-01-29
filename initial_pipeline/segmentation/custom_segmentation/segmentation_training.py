from data_utils import index_segmentations_df
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
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
from segmentation_preprocessing import preprocess_segmentations


class SegmentationDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "image_path"]
        mask_path = self.df.loc[idx, "mask_path"]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = np.array(image)
        mask = np.array(mask)

        # Normalize mask to {0,1}
        mask = (mask > 0).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image = image.float() / 255.0
        mask = mask.unsqueeze(0)  # (1, H, W)

        return image, mask
    

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = DoubleConv(3, 64)
        self.d2 = DoubleConv(64, 128)
        self.d3 = DoubleConv(128, 256)

        self.u2 = DoubleConv(256 + 128, 128)
        self.u1 = DoubleConv(128 + 64, 64)

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(F.max_pool2d(c1, 2))
        c3 = self.d3(F.max_pool2d(c2, 2))

        u2 = F.interpolate(c3, scale_factor=2, mode="bilinear", align_corners=False)
        u2 = self.u2(torch.cat([u2, c2], dim=1))

        u1 = F.interpolate(u2, scale_factor=2, mode="bilinear", align_corners=False)
        u1 = self.u1(torch.cat([u1, c1], dim=1))

        return self.out(u1)

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




def save_model_with_timestamp(model):

    # Create output directory
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # Timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Filename
    model_path = os.path.join(
        save_dir,
        f"unet_MG_whole_{timestamp}.pth"
    )

    # Save weights
    torch.save(model.state_dict(), model_path)

    print(f"Model saved to: {model_path}")





# Visualize predictions

def main(): 

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
    # Only take first 100 for now to train fast
    # path_df = path_df.head(160)
    print(len(path_df), "annotation pairs found")



    # train_df, test_df = train_test_split(
    #     path_df,
    #     test_size=0.2,
    #     random_state=42,
    #     shuffle=True
    # )

        # First split: train + temp (val+test)
    train_df, temp_df = train_test_split(
        path_df,
        test_size=0.3,        # 70% train, 30% temp
        random_state=42,
        shuffle=True
    )

    # Second split: val + test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,        # 15% val, 15% test
        random_state=42,
        shuffle=True
    )

    print('Training, validatoin and testing sizes')
    print(len(train_df), len(val_df), len(test_df))


    
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



    train_ds = SegmentationDataset(train_df, train_tfms)
    test_ds  = SegmentationDataset(test_df, test_tfms)
    val_ds = SegmentationDataset(val_df, test_tfms)

    # train_ds = SegmentationDataset(train_df)
    # test_ds  = SegmentationDataset(test_df)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=0)

    
    criterion = nn.BCEWithLogitsLoss()
    # criterion = DiceLoss()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'currently using device: {device}')

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    

        
    max_epochs = 40
    patience = 10

    best_dice = -float("inf")
    best_epoch = -1
    epochs_no_improve = 0

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

            # Save best model
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


if __name__ == "__main__":
    main()