from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from random_disconnections.disconnect_components import disconnect_branches_with_gap
from random_disconnections.find_connection_points import find_random_branch_points
from random_disconnections.random_colour_noise import add_floating_synthetic_fragments



class SegmentationDataset(Dataset):
    def __init__(self, df, transform=None, disconnect_components=True,
                 add_new_components=True, include_soma=False):
        """
        Args:
            df                  : DataFrame with at least columns image_path and mask_path.
                                  When include_soma=True, a soma_mask_path column is also
                                  expected (may contain None/NaN for cells without soma masks).
            transform           : Albumentations transform applied to image and whole-cell mask.
            disconnect_components: Apply random branch-disconnection augmentation.
            add_new_components  : Add synthetic floating fragments augmentation.
            include_soma        : If True, each item also returns the ground-truth soma mask
                                  as a third tensor (1, H, W), or None if unavailable.
                                  Only enable at evaluation time (no random augmentation is
                                  applied to the soma mask beyond resizing).
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.disconnect_components = disconnect_components
        self.add_new_components = add_new_components
        self.include_soma = include_soma

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

        # Override the image variables above if you want random changes
        if (self.disconnect_components): 
            
            points = find_random_branch_points(
                image_path=img_path, 
                mask_path = mask_path, 
                points_per_branch=5, 
                min_branch_length=10
            )

            H, W = image.shape[:2]
            box_size = int(0.1 * H)   # 10% of image height

            # --- cut branches locally ---
            image, _ = disconnect_branches_with_gap(
                img_path,
                mask_path,
                points, 
                box_size = box_size,
                blur_output = True, 
                blur_feather_radius=1
            )
            
            # swap order of colours to make it consitent with the code below
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.add_new_components: 
            image = add_floating_synthetic_fragments(
                    image,
                    mask
                )


        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image = image.float() / 255.0
        mask = mask.unsqueeze(0)  # (1, H, W)

        if not self.include_soma:
            return image, mask

        # ---- Soma mask (only loaded when include_soma=True) ----------------
        soma_mask_path = self.df.loc[idx, "soma_mask_path"] if "soma_mask_path" in self.df.columns else None

        soma_tensor = None
        if soma_mask_path is not None and str(soma_mask_path) not in ("nan", "None", ""):
            try:
                soma_img = Image.open(soma_mask_path).convert("L")
                soma_arr = (np.array(soma_img) > 0).astype(np.float32)

                # Resize soma to match the transformed image size using the
                # same deterministic resize the main transform already applied.
                # We apply only a plain resize (no random augmentation) so that
                # the soma mask stays geometrically aligned with the cell mask.
                target_h, target_w = mask.shape[1], mask.shape[2]
                soma_arr_resized = np.array(
                    Image.fromarray((soma_arr * 255).astype(np.uint8)).resize(
                        (target_w, target_h), Image.NEAREST
                    )
                ) / 255.0
                soma_tensor = torch.from_numpy(soma_arr_resized).float().unsqueeze(0)  # (1, H, W)
            except Exception:
                soma_tensor = None

        return image, mask, soma_tensor


class UnlabelledCellDataset(Dataset):
    """
    Dataset for unlabelled cell crops used in Mean Teacher semi-supervised training.

    Scans a root directory that contains one subfolder per scan, where each
    subfolder holds individual JPEG/PNG cell crops produced by YOLO detection.

    Both the student and teacher receive the SAME randomly-augmented view of
    each image (same flip/rotation applied via a shared ReplayCompose).  The
    replay data is returned alongside the tensors so the training loop can
    invert the spatial transform on the teacher's output prediction before
    computing the pixel-wise MSE consistency loss.

    This ensures the consistency loss compares spatially-aligned predictions:
      - Student pred (in augmented space)
      - Teacher pred inverse-transformed back to augmented space  ← aligned

    Args:
        root_dir:   Path to the directory that contains one subfolder per scan.
        transform:  albumentations.ReplayCompose transform. Must use ReplayCompose
                    (not plain Compose) so replay data can be captured.
                    If None, only resize to 256x256 is applied.
    """

    _EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    def __init__(self, root_dir, transform=None):
        self.transform = transform

        root = Path(root_dir)
        self.image_paths = sorted([
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in self._EXTENSIONS
        ])

        if len(self.image_paths) == 0:
            raise ValueError(
                f"No images found under {root_dir}. "
                f"Expected sub-folders with {self._EXTENSIONS} files."
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = np.array(Image.open(img_path).convert("RGB"))

        _fallback_tfm = A.ReplayCompose([A.Resize(256, 256), ToTensorV2()])

        if self.transform:
            student_result = self.transform(image=image)
            student_view   = student_result["image"].float() / 255.0
            student_replay = student_result["replay"]

            teacher_result = self.transform(image=image)
            teacher_view   = teacher_result["image"].float() / 255.0
            teacher_replay = teacher_result["replay"]
        else:
            student_result = _fallback_tfm(image=image)
            student_view   = student_result["image"].float() / 255.0
            student_replay = None

            teacher_result = _fallback_tfm(image=image)
            teacher_view   = teacher_result["image"].float() / 255.0
            teacher_replay = None

        # Original: resize only, no augmentation — used by the debug visualiser.
        original = A.Compose([A.Resize(256, 256), ToTensorV2()])(image=image)["image"]
        original = original.float() / 255.0

        return original, student_view, student_replay, teacher_view, teacher_replay