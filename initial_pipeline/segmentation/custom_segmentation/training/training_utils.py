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
                        choices=["bce", "dice", "cldice", "betti", 
                                "bce_cldice", "dice_cldice",
                                "bce_cldice_betti", "dice_cldice_betti"],
                        help="Loss function type: 'bce' (Binary Cross Entropy), "
                             "'dice' (Dice loss), 'cldice' (Centerline Dice), "
                             "'betti' (Betti Matching - topology), "
                             "'bce_cldice' (BCE + clDice), 'dice_cldice' (Dice + clDice), "
                             "'bce_cldice_betti' (BCE + clDice + Betti), "
                             "'dice_cldice_betti' (Dice + clDice + Betti)")
    
    parser.add_argument("--cldice_alpha", type=float, default=0.5,
                        help="Weight for clDice in combined losses (0.0-1.0). "
                             "For dual combinations: loss = (1-alpha)*base + alpha*clDice. "
                             "For triple combinations: loss = (1-alpha-beta)*base + alpha*clDice + beta*Betti")
    
    parser.add_argument("--betti_beta", type=float, default=0.3,
                        help="Weight for Betti loss in triple combinations (0.0-1.0). "
                             "Only used for 'bce_cldice_betti' and 'dice_cldice_betti'")
    
    parser.add_argument("--betti_b0_weight", type=float, default=1.0,
                        help="Weight for component count (β0) in Betti loss. "
                             "Higher values penalize wrong number of components more")
    
    parser.add_argument("--betti_b1_weight", type=float, default=0.5,
                        help="Weight for hole count (β1) in Betti loss. "
                             "Higher values penalize wrong number of holes more")
    
    return parser.parse_args()