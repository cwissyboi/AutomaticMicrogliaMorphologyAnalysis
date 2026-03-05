"""
Mean Teacher Semi-Supervised Training for Microglia Segmentation
================================================================

This script extends the supervised segmentation training in segmentation_training.py
with a Mean Teacher semi-supervised objective (Tarvainen & Valpola, NeurIPS 2017).

How it works
------------
1. Two UNets share the same architecture:
   - Student  — trained with gradient descent on:
       (a) supervised loss on labelled crops, and
       (b) consistency loss on unlabelled crops
   - Teacher  — updated only via Exponential Moving Average (EMA) of student weights;
                never receives gradient updates directly.

2. For each training step:
   a. Labelled batch  → student forward → supervised loss (same CombinedLoss as before)
   b. Unlabelled batch:
      - Student  receives one augmented view of each crop
      - Teacher  receives a DIFFERENT augmented view of the same crop
      - Consistency loss = MSE( σ(student_logits), σ(teacher_logits) )
      - Teacher logits are computed under torch.no_grad()
   c. Total loss = supervised_loss + w(epoch) * consistency_loss
      where w(epoch) is a sigmoid ramp-up from 0 → consistency_weight

3. After each gradient step, teacher weights are updated:
       θ_teacher ← alpha * θ_teacher + (1 - alpha) * θ_student

4. Validation and early stopping use the student model on the labelled val set,
   identical to the original supervised training.

5. Final evaluation mirrors segmentation_training.py exactly:
   - Standard pixel metrics (Dice, IoU, Morphology)
   - Region-based metrics (soma vs branches)
   - Both adaptive and probability-based postprocessing variants

Usage
-----
python mean_teacher_training.py \\
    --unlabelled_dir /path/to/yolo_for_semi_supervised \\
    --loss_type bce_cldice_betti \\
    --cldice_alpha 0.4 \\
    --betti_beta 0.2 \\
    --ema_alpha 0.999 \\
    --consistency_weight 1.0 \\
    --consistency_rampup 20

If --unlabelled_dir is not provided the script falls back to pure supervised
training, making it a drop-in replacement for segmentation_training.py.
"""

from data_utils import index_segmentations_df, pair_whole_and_soma_masks
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold, train_test_split
import torch
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import os
from tqdm import trange, tqdm
import torchvision.transforms.functional as TF

from segmentation_preprocessing import preprocess_segmentations
from training.training_utils import (
    set_seed,
    save_model_with_timestamp,
    create_ema_model,
    update_ema_weights,
    get_consistency_weight,
    parse_mean_teacher_args,
)
from training.segmentation_dataset import SegmentationDataset, UnlabelledCellDataset
from training.unet import UNet
from training.cl_dice import soft_cldice_loss, dice_loss, consistency_loss
from training.betti_loss import BettiMatchingLoss
from training.evaluation import (
    evaluate_regions,
    aggregate_metrics,
    print_metrics_summary,
    dice_score,
    iou_score,
    morphology_similarity_score,
)
from crf import connect_components_adaptive, connect_components_probability_based
from training.printing_utils import print_all_cross_validation_results

from setup_imports import setup_initial_pipeline_path
setup_initial_pipeline_path()


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class CombinedLoss(nn.Module):
    """
    Identical to the CombinedLoss in segmentation_training.py.
    Kept here to make this script self-contained.
    """
    def __init__(self, loss_type='bce', alpha=0.5, beta=0.3,
                 betti_b0_weight=1.0, betti_b1_weight=0.5):
        super().__init__()
        self.loss_type = loss_type
        self.alpha = alpha
        self.beta = beta
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.betti_loss = BettiMatchingLoss(
            beta_0_weight=betti_b0_weight,
            beta_1_weight=betti_b1_weight,
            soft=False
        )

    def forward(self, logits, target):
        probs = torch.sigmoid(logits)

        if self.loss_type == 'bce':
            return self.bce_loss(logits, target)
        elif self.loss_type == 'dice':
            return dice_loss(probs, target).mean()
        elif self.loss_type == 'cldice':
            return soft_cldice_loss(probs, target).mean()
        elif self.loss_type == 'betti':
            return self.betti_loss(probs, target)
        elif self.loss_type == 'bce_cldice':
            bce = self.bce_loss(logits, target)
            cldice = soft_cldice_loss(probs, target).mean()
            return (1 - self.alpha) * bce + self.alpha * cldice
        elif self.loss_type == 'dice_cldice':
            dice = dice_loss(probs, target).mean()
            cldice = soft_cldice_loss(probs, target).mean()
            return (1 - self.alpha) * dice + self.alpha * cldice
        elif self.loss_type == 'bce_cldice_betti':
            bce = self.bce_loss(logits, target)
            cldice = soft_cldice_loss(probs, target).mean()
            betti = self.betti_loss(probs, target)
            return (1 - self.alpha - self.beta) * bce + self.alpha * cldice + self.beta * betti
        elif self.loss_type == 'dice_cldice_betti':
            dice = dice_loss(probs, target).mean()
            cldice = soft_cldice_loss(probs, target).mean()
            betti = self.betti_loss(probs, target)
            return (1 - self.alpha - self.beta) * dice + self.alpha * cldice + self.beta * betti
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

DEBUG_OUTPUT_DIR = Path("debug_output/mean_teacher")


def invert_spatial_replay(pred_tensor, replay):
    """
    Invert the spatial augmentations recorded in an Albumentations ReplayCompose
    replay dict, applied to a prediction tensor.

    Only HorizontalFlip, VerticalFlip and RandomRotate90 are handled — these are
    the only spatial transforms used in the unlabelled augmentation pipeline.
    All three are self-inverse (flip) or have a simple inverse (rotate90 k times
    → rotate90 (4-k) times), so no approximation is needed.

    Args:
        pred_tensor: (1, 1, H, W) or (1, H, W) float tensor — teacher prediction
                     in augmented space.
        replay:      The 'replay' dict returned by albumentations.ReplayCompose.
                     May be None (no-op).

    Returns:
        Tensor of the same shape with spatial transforms inverted.
    """
    if replay is None:
        return pred_tensor

    squeeze = pred_tensor.dim() == 3
    if squeeze:
        pred_tensor = pred_tensor.unsqueeze(0)  # → (1, 1, H, W)

    # Work through the recorded transforms in REVERSE order
    transforms_applied = replay.get("transforms", [])

    for t in reversed(transforms_applied):
        if not t.get("applied", False):
            continue

        name = t.get("__class_fullname__", "")

        if "HorizontalFlip" in name:
            pred_tensor = torch.flip(pred_tensor, dims=[-1])

        elif "VerticalFlip" in name:
            pred_tensor = torch.flip(pred_tensor, dims=[-2])

        elif "RandomRotate90" in name:
            # Albumentations records the number of 90° CCW rotations applied.
            # The inverse is (4 - k) % 4 CCW rotations = k CW rotations.
            k = t.get("params", {}).get("factor", 0)
            k_inv = (4 - k) % 4
            if k_inv > 0:
                # torch.rot90 with k>0 rotates CCW; we want CW so negate k
                pred_tensor = torch.rot90(pred_tensor, k=k_inv, dims=[-2, -1])

    if squeeze:
        pred_tensor = pred_tensor.squeeze(0)

    return pred_tensor


def _tensor_to_pil_rgb(t):
    """Convert a (3, H, W) float [0,1] tensor to a PIL RGB image."""
    arr = (t.cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _prob_map_to_pil(logits_hw):
    """Convert a (H, W) raw logit tensor to a grayscale PIL image (0=bg, 255=fg)."""
    probs = torch.sigmoid(logits_hw).cpu().numpy()
    arr = (probs * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def _label_image(img, text, font_size=14):
    """Draw a text label at the top-left of a PIL image (returns a copy)."""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    # Use default bitmap font — always available, no font file needed
    draw.rectangle([0, 0, img.width, font_size + 4], fill=(0, 0, 0))
    draw.text((2, 2), text, fill=(255, 255, 255))
    return img


def save_debug_visuals(
    x_orig,
    x_student,
    x_teacher,
    student_logits_u,
    teacher_logits_u,
    student_logits_inv,
    teacher_logits_inv,
    epoch, fold,
    n_samples=4,
):
    """
    Save side-by-side debug images to debug_output/mean_teacher/.

    For each of the first n_samples crops in the batch this writes one PNG
    with eight panels:

        original | student aug | teacher aug |
        student pred raw | teacher pred raw |
        student pred inv | teacher pred inv | abs difference

    Panels explained:
      - original:              Unaugmented 256x256 crop (canonical orientation).
      - student aug:           Student's independently augmented view.
      - teacher aug:           Teacher's independently augmented view (different
                               random flip/rotation than student's).
      - student pred (raw):    Student's probability map in student-augmented space.
      - teacher pred (raw):    Teacher's probability map in teacher-augmented space.
      - student pred (inv):    Student pred after inverting student augmentation
                               → canonical orientation.
      - teacher pred (inv):    Teacher pred after inverting teacher augmentation
                               → canonical orientation.
      - abs difference:        |student pred (inv) - teacher pred (inv)|
                               Should be small and unstructured if both inversions
                               are correct.

    Args:
        x_orig:             (B, 3, H, W) float [0,1] — original unaugmented crops
        x_student:          (B, 3, H, W) float [0,1] — student's augmented view
        x_teacher:          (B, 3, H, W) float [0,1] — teacher's augmented view
        student_logits_u:   (B, 1, H, W) raw student logits (student-aug space)
        teacher_logits_u:   (B, 1, H, W) raw teacher logits (teacher-aug space)
        student_logits_inv: (B, 1, H, W) student logits inverted to canonical space
        teacher_logits_inv: (B, 1, H, W) teacher logits inverted to canonical space
        epoch:              Current epoch index (0-based)
        fold:               Current CV fold index (0-based)
        n_samples:          How many crops from the batch to save
    """
    DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    b = min(n_samples, x_orig.shape[0])

    for i in range(b):
        orig_img        = _label_image(_tensor_to_pil_rgb(x_orig[i]),                  "original")
        student_aug_img = _label_image(_tensor_to_pil_rgb(x_student[i]),               "student aug")
        teacher_aug_img = _label_image(_tensor_to_pil_rgb(x_teacher[i]),               "teacher aug")
        student_raw     = _label_image(_prob_map_to_pil(student_logits_u[i, 0]),        "student pred (raw)")
        teacher_raw     = _label_image(_prob_map_to_pil(teacher_logits_u[i, 0]),        "teacher pred (raw)")
        student_inv     = _label_image(_prob_map_to_pil(student_logits_inv[i, 0]),      "student pred (inv)")
        teacher_inv     = _label_image(_prob_map_to_pil(teacher_logits_inv[i, 0]),      "teacher pred (inv)")

        # Absolute difference between the two canonically aligned predictions
        diff = (
            torch.sigmoid(student_logits_inv[i, 0]) -
            torch.sigmoid(teacher_logits_inv[i, 0])
        ).abs()
        diff_arr = (diff.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        diff_img = _label_image(Image.fromarray(diff_arr).convert("RGB"), "abs difference")

        W, H = orig_img.size
        strip = Image.new("RGB", (W * 8, H))
        strip.paste(orig_img,        (0,     0))
        strip.paste(student_aug_img, (W,     0))
        strip.paste(teacher_aug_img, (W * 2, 0))
        strip.paste(student_raw,     (W * 3, 0))
        strip.paste(teacher_raw,     (W * 4, 0))
        strip.paste(student_inv,     (W * 5, 0))
        strip.paste(teacher_inv,     (W * 6, 0))
        strip.paste(diff_img,        (W * 7, 0))

        fname = DEBUG_OUTPUT_DIR / f"fold{fold}_epoch{epoch:03d}_sample{i:02d}.png"
        strip.save(fname)

    tqdm.write(f"[debug] Saved {b} Mean Teacher debug images to {DEBUG_OUTPUT_DIR}/")


def train_epoch_mean_teacher(
    student, teacher, labelled_loader, unlabelled_iter,
    device, optimizer, criterion,
    ema_alpha, consistency_w,
    epoch=0, fold=0, debug=True,
):
    """
    One epoch of Mean Teacher training.

    Args:
        student:           Student UNet (receives gradient updates).
        teacher:           Teacher UNet (EMA copy, no gradients).
        labelled_loader:   DataLoader yielding (image, mask) pairs.
        unlabelled_iter:   Infinite iterator over UnlabelledCellDataset
                           yielding (student_view, teacher_view) pairs.
                           Pass None to run pure supervised.
        device:            Torch device.
        optimizer:         Student optimizer.
        criterion:         CombinedLoss for the supervised term.
        ema_alpha:         EMA decay coefficient for teacher update.
        consistency_w:     Current (ramped) consistency loss weight.
        epoch:             Current epoch index (used for debug filenames).
        fold:              Current CV fold index (used for debug filenames).
        debug:             If True, save debug visualisations on the first batch
                           of every 10th epoch (epochs 0, 10, 20, ...).

    Returns:
        dict with keys 'supervised', 'consistency', 'total' — average losses.
    """
    student.train()
    if teacher is not None:
        teacher.train()  # BatchNorm in eval mode would use running stats;
                         # keeping train mode ensures BN uses batch stats for
                         # the teacher's prediction, matching original MT paper.

    total_sup = 0.0
    total_con = 0.0
    total_all = 0.0
    # Debug visuals fire once per debug epoch, on the first unlabelled batch
    _debug_saved_this_epoch = False
    _save_debug_this_epoch  = debug and (epoch % 10 == 0)

    for x_lab, y_lab in labelled_loader:
        x_lab = x_lab.to(device)
        y_lab = y_lab.to(device)

        # ---- supervised term -----------------------------------------------
        student_logits_lab = student(x_lab)
        sup_loss = criterion(student_logits_lab, y_lab)

        # ---- consistency term (skip if no unlabelled data) -----------------
        con_loss = torch.tensor(0.0, device=device)

        if unlabelled_iter is not None and consistency_w > 0.0:
            try:
                x_u_orig, x_student, student_replays, x_teacher, teacher_replays = next(unlabelled_iter)
            except StopIteration:
                con_loss = torch.tensor(0.0, device=device)
            else:
                x_student = x_student.to(device)
                x_teacher = x_teacher.to(device)

                # Student forward on its own augmented view
                student_logits_u = student(x_student)

                # Teacher forward on its own independently augmented view
                with torch.no_grad():
                    teacher_logits_u = teacher(x_teacher)

                # Invert each prediction back to canonical (original) space so
                # the pixel-wise MSE compares spatially aligned outputs.
                student_logits_inv = torch.stack([
                    invert_spatial_replay(
                        student_logits_u[i],   # (1, H, W)
                        student_replays[i] if student_replays is not None else None,
                    )
                    for i in range(student_logits_u.shape[0])
                ])  # → (B, 1, H, W)

                teacher_logits_inv = torch.stack([
                    invert_spatial_replay(
                        teacher_logits_u[i],   # (1, H, W)
                        teacher_replays[i] if teacher_replays is not None else None,
                    )
                    for i in range(teacher_logits_u.shape[0])
                ])  # → (B, 1, H, W)

                # Both inverted preds are now in canonical orientation → MSE
                con_loss = consistency_loss(student_logits_inv, teacher_logits_inv)

                # Debug: save all 8 panels side-by-side
                if _save_debug_this_epoch and not _debug_saved_this_epoch:
                    save_debug_visuals(
                        x_orig=x_u_orig.detach(),
                        x_student=x_student.detach(),
                        x_teacher=x_teacher.detach(),
                        student_logits_u=student_logits_u.detach(),
                        teacher_logits_u=teacher_logits_u.detach(),
                        student_logits_inv=student_logits_inv.detach(),
                        teacher_logits_inv=teacher_logits_inv.detach(),
                        epoch=epoch,
                        fold=fold,
                    )
                    _debug_saved_this_epoch = True

        # ---- total loss & optimisation -------------------------------------
        total_loss = sup_loss + consistency_w * con_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # ---- EMA teacher update --------------------------------------------
        if teacher is not None:
            update_ema_weights(teacher, student, ema_alpha)

        total_sup += sup_loss.item()
        total_con += con_loss.item()
        total_all += total_loss.item()

    n = len(labelled_loader)
    return {
        "supervised":   total_sup / n,
        "consistency":  total_con / n,
        "total":        total_all / n,
    }


# ---------------------------------------------------------------------------
# Evaluation  (identical to segmentation_training.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device, calculate_morphology_metric=False,
             apply_postprocessing=False, postprocessing_type='adaptive'):
    model.eval()
    dice_list, iou_list = [], []
    morphology_list = [] if calculate_morphology_metric else None

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits)

        for i in range(probs.shape[0]):
            pred   = probs[i, 0]
            target = y[i, 0]

            if apply_postprocessing:
                img_np = (x[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

                if postprocessing_type == 'adaptive':
                    pred_np = (pred > 0.5).cpu().numpy().astype(np.uint8)
                    pred = torch.from_numpy(
                        connect_components_adaptive(pred_np, img_np, min_component_frac=0.0)
                    ).float().to(device)

                elif postprocessing_type == 'probability':
                    prob_map = pred.cpu().numpy().astype(np.float32)
                    pred = torch.from_numpy(
                        connect_components_probability_based(
                            prob_map=prob_map, image=img_np,
                            threshold=0.5, min_component_frac=0.0,
                        )
                    ).float().to(device)
                else:
                    raise ValueError(f"Unknown postprocessing_type: {postprocessing_type}")

            dice_list.append(dice_score(pred.unsqueeze(0), target.unsqueeze(0)).item())
            iou_list.append(iou_score(pred.unsqueeze(0), target.unsqueeze(0)).item())

            if calculate_morphology_metric:
                morphology_list.append(morphology_similarity_score(pred, target))

    results = {
        "dice": sum(dice_list) / len(dice_list),
        "iou":  sum(iou_list)  / len(iou_list),
    }
    if calculate_morphology_metric:
        results["morphology"] = sum(morphology_list) / len(morphology_list)
    return results


@torch.no_grad()
def evaluate_test_with_regions(model, test_df, device, training_data_dir,
                                img_size=256, threshold=0.5,
                                apply_postprocessing=False,
                                postprocessing_type='adaptive'):
    model.eval()

    test_image_paths = set(test_df['image_path'].tolist())
    paired_df = pair_whole_and_soma_masks(training_data_dir)
    test_paired_df = paired_df[
        paired_df['image_path'].isin(test_image_paths)
    ].reset_index(drop=True)

    if len(test_paired_df) == 0:
        print(f"\nWarning: No samples found for region-based evaluation. Skipping.")
        return None

    soma_count = test_paired_df['soma_mask_path'].notna().sum()
    pp_label = f" (WITH POSTPROCESSING: {postprocessing_type})" if apply_postprocessing else ""
    print(f"\nREGION-BASED EVALUATION ON TEST SET{pp_label}")
    print(f"Test samples: {len(test_paired_df)} | Samples with soma: {soma_count}")

    all_metrics = []

    for _, row in test_paired_df.iterrows():
        img = Image.open(row['image_path']).convert('RGB')
        img_tensor = TF.to_tensor(img).unsqueeze(0).to(device)

        if img_tensor.shape[2] != img_size or img_tensor.shape[3] != img_size:
            img_tensor = TF.resize(img_tensor, [img_size, img_size])

        pred = model(img_tensor)
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)
        pred = pred.squeeze(0).squeeze(0)

        if apply_postprocessing:
            img_np = (img_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            if postprocessing_type == 'adaptive':
                pred_np = (pred > threshold).cpu().numpy().astype(np.uint8)
                pred = torch.from_numpy(
                    connect_components_adaptive(pred_np, img_np, min_component_frac=0.0)
                ).float()
            elif postprocessing_type == 'probability':
                prob_map = pred.cpu().numpy().astype(np.float32)
                pred = torch.from_numpy(
                    connect_components_probability_based(
                        prob_map=prob_map, image=img_np,
                        threshold=threshold, min_component_frac=0.0,
                    )
                ).float()
            else:
                raise ValueError(f"Unknown postprocessing_type: {postprocessing_type}")

        whole_mask = TF.to_tensor(
            Image.open(row['whole_mask_path']).convert('L')
        ).squeeze(0)
        if whole_mask.shape[0] != img_size or whole_mask.shape[1] != img_size:
            whole_mask = TF.resize(whole_mask.unsqueeze(0), [img_size, img_size]).squeeze(0)

        soma_mask_tensor = None
        if pd.notna(row['soma_mask_path']):
            soma_mask = TF.to_tensor(
                Image.open(row['soma_mask_path']).convert('L')
            ).squeeze(0)
            if soma_mask.shape[0] != img_size or soma_mask.shape[1] != img_size:
                soma_mask = TF.resize(soma_mask.unsqueeze(0), [img_size, img_size]).squeeze(0)
            soma_mask_tensor = soma_mask

        all_metrics.append(
            evaluate_regions(
                pred=pred,
                whole_cell_mask=whole_mask,
                soma_mask=soma_mask_tensor,
                threshold=threshold,
            )
        )

    return aggregate_metrics(all_metrics)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_mean_teacher_args()
    set_seed(42)

    SEGMENTATIONS_DIR = Path.cwd().parents[3] / "SegmentationDatasets" / "AnnotationsData_Adjusted_WithSoma" / "Segmentations"

    path_df = index_segmentations_df(SEGMENTATIONS_DIR, mask_name='masks')
    path_df = path_df[path_df['class'] == 'MG_whole']

    path_df = path_df.head(40)
    print(f"{len(path_df)} annotation pairs found")

    # ------------------------------------------------------------------
    # Unlabelled data setup
    # ------------------------------------------------------------------
    use_mean_teacher = args.unlabelled_dir is not None
    if use_mean_teacher:
        print(f"\nMean Teacher mode ENABLED")
        print(f"  Unlabelled data dir : {args.unlabelled_dir}")
        print(f"  EMA alpha           : {args.ema_alpha}")
        print(f"  Consistency weight  : {args.consistency_weight}")
        print(f"  Consistency rampup  : {args.consistency_rampup} epochs")
        print(f"  Unlabelled batch    : {args.unlabelled_batch_size}\n")
    else:
        print("\nNo --unlabelled_dir provided. Running pure supervised training.\n")

    # Augmentation transforms
    # Labelled training — plain Compose (mask must be transformed together with image)
    train_tfms = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        ToTensorV2()
    ])

    # Unlabelled — ReplayCompose records which flips/rotations fired so the
    # training loop can invert them on the teacher's output before MSE.
    unlabelled_tfms = A.ReplayCompose([
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

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    all_fold_metrics                          = []
    all_fold_region_metrics                   = []
    all_fold_postprocessed_metrics            = []
    all_fold_postprocessed_region_metrics     = []
    all_fold_postprocessed_probability_metrics        = []
    all_fold_postprocessed_probability_region_metrics = []

    for fold, (trainval_idx, test_idx) in enumerate(kf.split(path_df)):
        print(f"\n{'='*70}")
        print(f"FOLD {fold + 1} / {k_folds}")
        print(f"{'='*70}")

        trainval_df = path_df.iloc[trainval_idx]
        test_df     = path_df.iloc[test_idx]

        train_df, val_df = train_test_split(
            trainval_df, test_size=0.1, random_state=42, shuffle=True
        )

        print(
            f"Sizes | Train: {len(train_df)}, "
            f"Val: {len(val_df)}, Test: {len(test_df)}"
        )

        # Labelled datasets
        train_ds = SegmentationDataset(
            train_df, train_tfms,
            disconnect_components=args.disconnect_components,
            add_new_components=args.add_new_components,
        )
        val_ds  = SegmentationDataset(val_df,  test_tfms, disconnect_components=False, add_new_components=False)
        test_ds = SegmentationDataset(test_df, test_tfms, disconnect_components=False, add_new_components=False)

        batch_size = 16
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

        # Unlabelled dataloader (infinite cycling iterator)
        unlabelled_iter = None
        if use_mean_teacher:
            unlabelled_ds = UnlabelledCellDataset(
                root_dir=args.unlabelled_dir,
                transform=unlabelled_tfms,
            )
            print(f"Unlabelled dataset size: {len(unlabelled_ds)} crops")

            def unlabelled_collate(batch):
                """
                Custom collate for UnlabelledCellDataset.

                Each item is (original, student_view, student_replay,
                              teacher_view, teacher_replay).
                Image tensors are stacked; replay dicts cannot be stacked by
                PyTorch's default collate so they are kept as plain Python lists.
                """
                originals       = torch.stack([item[0] for item in batch])
                student_views   = torch.stack([item[1] for item in batch])
                student_replays = [item[2] for item in batch]   # list of dicts
                teacher_views   = torch.stack([item[3] for item in batch])
                teacher_replays = [item[4] for item in batch]   # list of dicts
                return originals, student_views, student_replays, teacher_views, teacher_replays

            unlabelled_loader = DataLoader(
                unlabelled_ds,
                batch_size=args.unlabelled_batch_size,
                shuffle=True,
                drop_last=True,
                collate_fn=unlabelled_collate,
            )

            # Infinite iterator — cycles through unlabelled data continuously
            # so we always have an unlabelled batch for every labelled batch.
            def infinite_loader(loader):
                while True:
                    yield from loader

            unlabelled_iter = infinite_loader(unlabelled_loader)

        # Device + models
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")

        student = UNet().to(device)
        teacher = create_ema_model(student).to(device) if use_mean_teacher else None

        criterion = CombinedLoss(
            loss_type=args.loss_type,
            alpha=args.cldice_alpha,
            beta=args.betti_beta,
            betti_b0_weight=args.betti_b0_weight,
            betti_b1_weight=args.betti_b1_weight,
        )
        print(f"Supervised loss: {args.loss_type}")

        # ==================================================================
        # PHASE 1 — Pure supervised pre-training
        # ==================================================================
        print(f"\n{'─'*70}")
        print(f"PHASE 1 — Supervised pre-training  (lr=1e-4, patience={args.patience})")
        print(f"{'─'*70}")

        optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)

        best_dice          = -float("inf")
        best_epoch         = -1
        epochs_no_improve  = 0
        best_student_state = None

        for epoch in trange(args.epochs, desc="Phase 1", unit="epoch"):
            loss_dict = train_epoch_mean_teacher(
                student=student,
                teacher=None,           # no teacher in phase 1
                labelled_loader=train_loader,
                unlabelled_iter=None,   # no unlabelled data in phase 1
                device=device,
                optimizer=optimizer,
                criterion=criterion,
                ema_alpha=args.ema_alpha,
                consistency_w=0.0,      # consistency always off in phase 1
                epoch=epoch,
                fold=fold,
                debug=False,            # no debug visuals needed in phase 1
            )

            val_metrics  = evaluate(student, val_loader,  device)
            test_metrics = evaluate(student, test_loader, device)
            current_dice = val_metrics["dice"]

            if current_dice > best_dice:
                best_dice          = current_dice
                best_epoch         = epoch
                epochs_no_improve  = 0
                best_student_state = {
                    k: v.detach().cpu().clone()
                    for k, v in student.state_dict().items()
                }
                save_model_with_timestamp(student)
                status = "NEW BEST"
            else:
                epochs_no_improve += 1
                status = f"no improve ({epochs_no_improve}/{args.patience})"

            tqdm.write(
                f"[P1] Epoch {epoch:03d} | "
                f"Sup Loss: {loss_dict['supervised']:.4f} | "
                f"Val Dice: {current_dice:.4f} | "
                f"Val IoU: {val_metrics['iou']:.4f} | "
                f"Test Dice: {test_metrics['dice']:.4f} | "
                f"Test IoU: {test_metrics['iou']:.4f} | "
                f"{status}"
            )

            if epochs_no_improve >= args.patience:
                tqdm.write(
                    f"[P1] Early stopping at epoch {epoch}. "
                    f"Best Val Dice {best_dice:.4f} at epoch {best_epoch}."
                )
                break

        # Restore best supervised checkpoint
        student.load_state_dict(best_student_state)
        student.to(device)

        # Evaluate supervised-only performance (recorded for comparison)
        supervised_only_metrics = evaluate(
            student, test_loader, device, calculate_morphology_metric=True
        )
        print(
            f"\n[P1] Supervised-only test | "
            f"Dice: {supervised_only_metrics['dice']:.4f}, "
            f"IoU: {supervised_only_metrics['iou']:.4f}, "
            f"Morphology: {supervised_only_metrics['morphology']:.4f}"
        )

        # If no unlabelled data, skip phase 2 entirely
        if not use_mean_teacher:
            all_fold_metrics.append(supervised_only_metrics)
            # still run the full region / postprocessed evaluations below
            final_student_state = best_student_state
        else:
            # ==============================================================
            # PHASE 2 — Semi-supervised fine-tuning
            # ==============================================================
            print(f"\n{'─'*70}")
            print(
                f"PHASE 2 — Semi-supervised fine-tuning  "
                f"(lr={args.finetune_lr}, patience={args.finetune_patience})"
            )
            print(f"{'─'*70}")

            # Initialise teacher from the converged supervised checkpoint
            # so its predictions on unlabelled data are immediately useful.
            teacher.load_state_dict(best_student_state)
            teacher.to(device)

            # Lower learning rate — model is already in a good basin.
            optimizer = torch.optim.Adam(student.parameters(), lr=args.finetune_lr)

            # Reset early stopping for phase 2
            best_dice_ft         = best_dice   # must beat phase-1 to count as improvement
            best_epoch_ft        = -1
            epochs_no_improve_ft = 0
            best_student_state_ft = best_student_state  # fallback: keep phase-1 weights

            for epoch in trange(args.finetune_epochs, desc="Phase 2", unit="epoch"):
                c_weight = args.consistency_weight * get_consistency_weight(
                    epoch, args.consistency_rampup
                )

                loss_dict = train_epoch_mean_teacher(
                    student=student,
                    teacher=teacher,
                    labelled_loader=train_loader,
                    unlabelled_iter=unlabelled_iter,
                    device=device,
                    optimizer=optimizer,
                    criterion=criterion,
                    ema_alpha=args.ema_alpha,
                    consistency_w=c_weight,
                    epoch=epoch,
                    fold=fold,
                    debug=True,
                )

                val_metrics  = evaluate(student, val_loader,  device)
                test_metrics = evaluate(student, test_loader, device)
                current_dice = val_metrics["dice"]

                if current_dice > best_dice_ft:
                    best_dice_ft          = current_dice
                    best_epoch_ft         = epoch
                    epochs_no_improve_ft  = 0
                    best_student_state_ft = {
                        k: v.detach().cpu().clone()
                        for k, v in student.state_dict().items()
                    }
                    save_model_with_timestamp(student)
                    status = "NEW BEST"
                else:
                    epochs_no_improve_ft += 1
                    status = f"no improve ({epochs_no_improve_ft}/{args.finetune_patience})"

                tqdm.write(
                    f"[P2] Epoch {epoch:03d} | "
                    f"Sup Loss: {loss_dict['supervised']:.4f} | "
                    f"Consistency: {loss_dict['consistency']:.4f} (w={c_weight:.3f}) | "
                    f"Val Dice: {current_dice:.4f} | "
                    f"Val IoU: {val_metrics['iou']:.4f} | "
                    f"Test Dice: {test_metrics['dice']:.4f} | "
                    f"Test IoU: {test_metrics['iou']:.4f} | "
                    f"{status}"
                )

                if epochs_no_improve_ft >= args.finetune_patience:
                    tqdm.write(
                        f"[P2] Early stopping at epoch {epoch}. "
                        f"Best Val Dice {best_dice_ft:.4f} at epoch {best_epoch_ft}."
                    )
                    break

            # Restore best fine-tuned weights (or phase-1 weights if no improvement)
            student.load_state_dict(best_student_state_ft)
            student.to(device)

            if best_epoch_ft == -1:
                tqdm.write(
                    "[P2] Semi-supervised fine-tuning did not improve over "
                    "supervised pre-training. Using phase-1 weights for evaluation."
                )
            else:
                tqdm.write(
                    f"[P2] Fine-tuning improved Val Dice: "
                    f"{supervised_only_metrics['dice']:.4f} → {best_dice_ft:.4f}"
                )

            final_student_state = best_student_state_ft

        # Restore final weights before evaluation (no-op if already loaded above)
        student.load_state_dict(final_student_state)
        student.to(device)

        # ------------------------------------------------------------------
        # Final evaluation (mirrors segmentation_training.py exactly)
        # ------------------------------------------------------------------

        final_test_metrics = evaluate(student, test_loader, device, calculate_morphology_metric=True)
        all_fold_metrics.append(final_test_metrics)
        print(
            f"Fold {fold + 1} final test | "
            f"Dice: {final_test_metrics['dice']:.4f}, "
            f"IoU: {final_test_metrics['iou']:.4f}, "
            f"Morphology: {final_test_metrics['morphology']:.4f}"
        )

        # Region-based evaluation
        try:
            region_results = evaluate_test_with_regions(
                model=student, test_df=test_df, device=device,
                training_data_dir=SEGMENTATIONS_DIR, img_size=256, threshold=0.5,
            )
            if region_results is not None:
                print_metrics_summary(region_results)
                all_fold_region_metrics.append(region_results)
        except Exception as e:
            import traceback
            print(f"\n!!! ERROR in region-based evaluation: {e}")
            traceback.print_exc()

        # Postprocessed — adaptive
        print("\n" + "="*70)
        print("EVALUATING WITH POSTPROCESSING: ADAPTIVE")
        print("="*70)
        try:
            pp_adaptive = evaluate(
                student, test_loader, device,
                calculate_morphology_metric=True,
                apply_postprocessing=True, postprocessing_type='adaptive',
            )
            all_fold_postprocessed_metrics.append(pp_adaptive)
            print(
                f"Fold {fold + 1} postprocessed (adaptive) | "
                f"Dice: {pp_adaptive['dice']:.4f}, "
                f"IoU: {pp_adaptive['iou']:.4f}, "
                f"Morphology: {pp_adaptive['morphology']:.4f}"
            )
        except Exception as e:
            import traceback
            print(f"!!! ERROR in adaptive postprocessed evaluation: {e}")
            traceback.print_exc()

        try:
            pp_adaptive_region = evaluate_test_with_regions(
                model=student, test_df=test_df, device=device,
                training_data_dir=SEGMENTATIONS_DIR, img_size=256, threshold=0.5,
                apply_postprocessing=True, postprocessing_type='adaptive',
            )
            if pp_adaptive_region is not None:
                print_metrics_summary(pp_adaptive_region)
                all_fold_postprocessed_region_metrics.append(pp_adaptive_region)
        except Exception as e:
            import traceback
            print(f"!!! ERROR in adaptive postprocessed region evaluation: {e}")
            traceback.print_exc()

        # Postprocessed — probability
        print("\n" + "="*70)
        print("EVALUATING WITH POSTPROCESSING: PROBABILITY")
        print("="*70)
        try:
            pp_prob = evaluate(
                student, test_loader, device,
                calculate_morphology_metric=True,
                apply_postprocessing=True, postprocessing_type='probability',
            )
            all_fold_postprocessed_probability_metrics.append(pp_prob)
            print(
                f"Fold {fold + 1} postprocessed (probability) | "
                f"Dice: {pp_prob['dice']:.4f}, "
                f"IoU: {pp_prob['iou']:.4f}, "
                f"Morphology: {pp_prob['morphology']:.4f}"
            )
        except Exception as e:
            import traceback
            print(f"!!! ERROR in probability postprocessed evaluation: {e}")
            traceback.print_exc()

        try:
            pp_prob_region = evaluate_test_with_regions(
                model=student, test_df=test_df, device=device,
                training_data_dir=SEGMENTATIONS_DIR, img_size=256, threshold=0.5,
                apply_postprocessing=True, postprocessing_type='probability',
            )
            if pp_prob_region is not None:
                print_metrics_summary(pp_prob_region)
                all_fold_postprocessed_probability_region_metrics.append(pp_prob_region)
        except Exception as e:
            import traceback
            print(f"!!! ERROR in probability postprocessed region evaluation: {e}")
            traceback.print_exc()

        print("="*70 + "\n")

    # ------------------------------------------------------------------
    # Cross-validation summary
    # ------------------------------------------------------------------
    print_all_cross_validation_results(
        all_fold_metrics=all_fold_metrics,
        all_fold_region_metrics=all_fold_region_metrics,
        all_fold_postprocessed_metrics=all_fold_postprocessed_metrics,
        all_fold_postprocessed_region_metrics=all_fold_postprocessed_region_metrics,
        all_fold_postprocessed_probability_metrics=all_fold_postprocessed_probability_metrics,
        all_fold_postprocessed_probability_region_metrics=all_fold_postprocessed_probability_region_metrics,
    )


if __name__ == "__main__":
    main()
