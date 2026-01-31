import random
import argparse
import numpy as np
import torch
import os
from datetime import datetime


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



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



def parse_segmentation_args():
    parser = argparse.ArgumentParser(
        description="Train UNet segmentation model"
    )

    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    parser.add_argument("--folds", type=int, default=5,
                        help="Number of CV folds")

    parser.add_argument("--epochs", type=int, default=100,
                        help="Max training epochs")

    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")

    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")

    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size")

    return parser.parse_args()
