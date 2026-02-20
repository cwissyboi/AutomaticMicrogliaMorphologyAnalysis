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

    parser.add_argument("--epochs", type=int, default=100,
                        help="Max training epochs")

    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    
    parser.add_argument("--disconnect_components", type=bool, default=False,
                    help="Whether to apply random disconnections augmentation")
    
    parser.add_argument("--add_new_components", type=bool, default=False,
                    help="Whether to add synthetic fragments augmentation")
    
    parser.add_argument("--loss_type", type=str, default="bce",
                        choices=["bce", "dice", "cldice", "bce_cldice", "dice_cldice"],
                        help="Loss function type: 'bce' (Binary Cross Entropy), "
                             "'dice' (Dice loss), 'cldice' (Centerline Dice), "
                             "'bce_cldice' (BCE + clDice), 'dice_cldice' (Dice + clDice)")
    
    parser.add_argument("--cldice_alpha", type=float, default=0.5,
                        help="Weight for clDice in combined losses (0.0-1.0). "
                             "For 'bce_cldice': loss = (1-alpha)*BCE + alpha*clDice. "
                             "For 'dice_cldice': loss = (1-alpha)*Dice + alpha*clDice")
    
    return parser.parse_args()