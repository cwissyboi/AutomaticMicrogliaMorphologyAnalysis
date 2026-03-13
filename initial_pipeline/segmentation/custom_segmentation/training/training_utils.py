import random
import argparse
import numpy as np
import torch
import copy
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


# ---------------------------------------------------------------------------
# Mean Teacher utilities
# ---------------------------------------------------------------------------

def create_ema_model(student_model):
    """
    Create a teacher model as a deep copy of the student.

    The teacher's parameters are updated only via EMA, never via gradients,
    so we detach everything and freeze requires_grad.

    Args:
        student_model: Initialised student UNet (on any device).

    Returns:
        teacher_model: Identical architecture, same initial weights, no grads.
    """
    teacher_model = copy.deepcopy(student_model)
    for param in teacher_model.parameters():
        param.requires_grad_(False)
    return teacher_model


@torch.no_grad()
def update_ema_weights(teacher_model, student_model, alpha):
    """
    Update teacher model weights using exponential moving average.

    θ_teacher  ←  alpha * θ_teacher  +  (1 - alpha) * θ_student

    A higher alpha means the teacher changes more slowly (more momentum).
    Typical range: 0.99 – 0.999.

    Args:
        teacher_model: EMA model whose parameters are updated in-place.
        student_model: Student model after the latest gradient step.
        alpha:         EMA decay coefficient.
    """
    for t_param, s_param in zip(teacher_model.parameters(), student_model.parameters()):
        t_param.data.mul_(alpha).add_(s_param.data, alpha=1.0 - alpha)


def get_consistency_weight(epoch, rampup_epochs):
    """
    Sigmoid ramp-up schedule for the unsupervised consistency loss weight.

    Returns a value in [0, 1]:
      - 0.0 at epoch 0  (pure supervised training at the start)
      - ~1.0 after rampup_epochs

    This prevents the consistency loss from dominating before the student has
    learned meaningful representations.

    Args:
        epoch:          Current training epoch (0-indexed).
        rampup_epochs:  Number of epochs over which to ramp up.

    Returns:
        float in [0, 1]
    """
    if rampup_epochs == 0:
        return 1.0
    rampup_ratio = min(epoch / rampup_epochs, 1.0)
    # Gaussian ramp-up as used in the original Mean Teacher paper
    return float(np.exp(-5.0 * (1.0 - rampup_ratio) ** 2))


def parse_mean_teacher_args():
    """
    Argument parser for Mean Teacher semi-supervised training.

    Extends the base segmentation args with Mean Teacher-specific options.
    """
    parser = argparse.ArgumentParser(
        description="Mean Teacher semi-supervised UNet segmentation training"
    )

    # ---- inherited supervised args ----------------------------------------
    parser.add_argument("--epochs", type=int, default=100,
                        help="Max training epochs")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--disconnect_components", type=bool, default=False,
                        help="Apply random disconnections augmentation to labelled data")
    parser.add_argument("--add_new_components", type=bool, default=False,
                        help="Add synthetic fragment augmentation to labelled data")
    parser.add_argument("--loss_type", type=str, default="bce",
                        choices=["bce", "dice", "cldice", "betti",
                                 "bce_cldice", "dice_cldice",
                                 "bce_cldice_betti", "dice_cldice_betti"],
                        help="Supervised loss function type")
    parser.add_argument("--cldice_alpha", type=float, default=0.5,
                        help="clDice weight in combined supervised losses")
    parser.add_argument("--betti_beta", type=float, default=0.3,
                        help="Betti loss weight in triple supervised losses")
    parser.add_argument("--betti_b0_weight", type=float, default=1.0,
                        help="β0 (component count) weight inside Betti loss")
    parser.add_argument("--betti_b1_weight", type=float, default=0.5,
                        help="β1 (hole count) weight inside Betti loss")

    # ---- Mean Teacher-specific args ---------------------------------------
    parser.add_argument("--unlabelled_dir", type=str,
                        default=None,
                        help="Root directory of unlabelled cell crops "
                             "(sub-folders per scan). If None, falls back to "
                             "pure supervised training.")
    parser.add_argument("--ema_alpha", type=float, default=0.999,
                        help="EMA decay for teacher weight update. "
                             "Higher = slower teacher update. Typical: 0.99–0.999")
    parser.add_argument("--consistency_weight", type=float, default=1.0,
                        help="Maximum weight of the unsupervised consistency loss "
                             "relative to the supervised loss")
    parser.add_argument("--consistency_rampup", type=int, default=20,
                        help="Number of epochs over which to ramp up the "
                             "consistency loss weight from 0 to consistency_weight")
    parser.add_argument("--unlabelled_batch_size", type=int, default=16,
                        help="Batch size for unlabelled data loader")

    # ---- Two-phase training args ------------------------------------------
    parser.add_argument("--finetune_epochs", type=int, default=100,
                        help="Max epochs for phase 2 semi-supervised fine-tuning. "
                             "Only used when --unlabelled_dir is provided.")
    parser.add_argument("--finetune_patience", type=int, default=10,
                        help="Early stopping patience for phase 2 fine-tuning.")
    parser.add_argument("--finetune_lr", type=float, default=1e-5,
                        help="Learning rate for phase 2 fine-tuning. Should be "
                             "lower than phase 1 LR since the model is already "
                             "converged. Default: 1e-5 (10x lower than phase 1).")

    # ---- Exp 4: photometric augmentation for unlabelled data -----------------
    parser.add_argument("--photometric_augmentation", action="store_true",
                        default=False,
                        help="Add colour jitter, Gaussian blur and Gaussian noise "
                             "to the unlabelled augmentation pipeline. These are NOT "
                             "inverted before the consistency loss, forcing the student "
                             "and teacher to produce invariant predictions across "
                             "photometric perturbations.")

    # ---- Exp 2: joint training from scratch (no separate Phase 1) -----------
    parser.add_argument("--joint_training", action="store_true",
                        default=False,
                        help="Skip the pure supervised Phase 1 pre-training and run "
                             "supervised + Mean Teacher consistency jointly from epoch 0. "
                             "Uses --finetune_lr as the learning rate throughout and "
                             "--consistency_rampup to gradually introduce the consistency "
                             "term. Requires --unlabelled_dir.")

    # ---- Exp 6: pseudo-label mode -------------------------------------------
    parser.add_argument("--pseudo_label_threshold", type=float, default=None,
                        help="If set, replace the soft MSE consistency loss with a "
                             "hard pseudo-label supervised loss on unlabelled crops. "
                             "Teacher predictions with mean confidence above this "
                             "threshold (e.g. 0.8) are binarised and used as pseudo "
                             "ground-truth masks for the student supervised loss. "
                             "Crops below the threshold are skipped. "
                             "Mutually exclusive with the default soft MSE consistency.")

    # ---- Labelled data fraction (for semi-supervised ablation) --------------
    parser.add_argument("--labelled_fraction", type=float, default=1.0,
                        help="Fraction of the labelled training set to use (0.0, 1.0]. "
                             "The subset is drawn with a fixed random seed so results "
                             "are reproducible. The val and test sets are always kept "
                             "at full size. Use this to study how performance scales "
                             "with the amount of labelled data, e.g. 0.1, 0.25, 0.5, "
                             "0.75, 1.0.")

    return parser.parse_args()