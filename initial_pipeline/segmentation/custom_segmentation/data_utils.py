from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path


# Read the annotations outputted from QuPath script into a df that has columns that point to the image path and the mask path
def index_segmentations_df(root_dir, mask_name = 'masks'):
    root_dir = Path(root_dir)
    images_root = root_dir / "images"
    masks_root = root_dir / mask_name

    records = []

    for scan_dir in images_root.iterdir():
        if not scan_dir.is_dir():
            continue

        scan_name = scan_dir.name

        for class_dir in scan_dir.iterdir():
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name

            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() not in [".png", ".jpg", ".tif", ".tiff"]:
                    continue

                mask_path = masks_root / scan_name / class_name / img_path.name

                if not mask_path.exists():
                    mask_path = None

                records.append({
                    "scan": scan_name,
                    "class": class_name,
                    "image_path": img_path,
                    "mask_path": mask_path,
                })

    df = pd.DataFrame(records)

    # Optional: sort for reproducibility
    df = df.sort_values(
        by=["scan", "class", "image_path"],
        ignore_index=True
    )

    return df


def export_calculated_masks(
    df,
    root_dir,
    mask_column="calculated_mask",
    output_root_name="calculated_masks",
    suffix=None,
):
    """
    Export calculated masks to the same folder structure as original masks.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: scan, class, image_path, calculated_mask
    root_dir : str or Path
        Root dataset directory (same as used for indexing)
    mask_column : str
        Column name containing numpy mask arrays
    output_root_name : str
        Folder name under Segmentations/ where masks are written
    suffix : str or None
        Optional suffix added before file extension (e.g. '_calc')
    """

    root_dir = Path(root_dir)
    out_root = root_dir / output_root_name

    for _, row in df.iterrows():
        scan = row["scan"]
        class_name = row["class"]
        image_path = Path(row["image_path"])

        mask = row[mask_column]
        if mask is None:
            continue

        # Ensure uint8 [0,255]
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        if mask.max() <= 1:
            mask = mask * 255

        # Build output path
        out_dir = out_root / scan / class_name
        out_dir.mkdir(parents=True, exist_ok=True)

        name = image_path.stem
        ext = image_path.suffix

        if suffix is not None:
            out_name = f"{name}{suffix}{ext}"
        else:
            out_name = image_path.name

        out_path = out_dir / out_name

        Image.fromarray(mask).save(out_path)
