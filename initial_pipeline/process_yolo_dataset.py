"""
process_yolo_dataset.py
-----------------------
Process all annotated images in a YOLO-format dataset (train / val / test splits)
through the post-YOLO morphology pipeline WITHOUT running YOLO inference.

Steps per cell (bounding box annotation):
    1. Parse YOLO .txt annotation  →  pixel-space [x1, y1, x2, y2] boxes
    2. UNet segmentation (unet_inference)
    3. CRF / component connection (connect_all_masks)
    4. Gaussian soma filter (get_gaussian_filter_soma_masks)
    5. Skeletonisation (get_skeletons)
    6. Morphological features (get_morphology_dataframe)

Output:
    A single CSV at OUTPUT_CSV containing one row per cell with the 25
    morphological features plus metadata columns.

Usage:
    python process_yolo_dataset.py
    (no command-line args needed — configure the block below)
"""

from pathlib import Path
import sys
import time

# ---------------------------------------------------------------------------
# Ensure initial_pipeline/ is importable regardless of working directory
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from segmentation.custom_segmentation.training.unet import UNet
from segmentation.custom_segmentation.segmentation_inference import unet_inference
from segmentation.custom_segmentation.crf import connect_all_masks
from segmentation.soma_segmentation.gaussian_filter import get_gaussian_filter_soma_masks
from morphology.morphology_features import get_skeletons, get_morphology_dataframe

# ===========================================================================
#  CONFIGURATION  ← edit these paths if needed
# ===========================================================================
YOLO_DATASET_DIR = Path(
    r"C:\Users\chris\Desktop\University\Thesis\DetectionDatasets"
    r"\YOLO_dataset_fixed_bounding_boxes"
)
SPLITS = ["train", "val", "test"]

UNET_CHECKPOINT = (
    _SCRIPT_DIR
    / "segmentation"
    / "custom_segmentation"
    / "checkpoints"
    / "best_run_25_1.pth"
)

OUTPUT_CSV = (
    _SCRIPT_DIR / "morphology" / "morphology_outputs" / "yolo_labelled_dataset.csv"
)

# Image extensions accepted by the dataset
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
# ===========================================================================


def parse_yolo_annotation(label_path: Path, img_w: int, img_h: int) -> np.ndarray:
    """
    Parse a YOLO-format .txt annotation file and return pixel-space boxes.

    YOLO format per line: <class_id> <cx> <cy> <w> <h>  (all normalised 0-1)

    Returns
    -------
    np.ndarray of shape (N, 4) with columns [x1, y1, x2, y2] in pixel coords,
    dtype int32.  Empty array if file missing or has no valid lines.
    """
    if not label_path.exists():
        return np.empty((0, 4), dtype=np.int32)

    boxes = []
    with open(label_path, "r") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                _, cx, cy, w, h = map(float, parts)
            except ValueError:
                continue

            x1 = int((cx - w / 2) * img_w)
            y1 = int((cy - h / 2) * img_h)
            x2 = int((cx + w / 2) * img_w)
            y2 = int((cy + h / 2) * img_h)

            # Clamp to image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_w - 1, x2)
            y2 = min(img_h - 1, y2)

            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])

    if not boxes:
        return np.empty((0, 4), dtype=np.int32)

    return np.array(boxes, dtype=np.int32)


def load_model(checkpoint_path: Path, device: str) -> UNet:
    """Load the UNet model from a checkpoint."""
    model = UNet()
    state_dict = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def process_split(
    split: str,
    yolo_dir: Path,
    model: torch.nn.Module,
    device: str,
) -> list[dict]:
    """
    Process all images in one dataset split.

    Returns a list of dicts, one per cell, containing morphological features
    and metadata.
    """
    images_dir = yolo_dir / "images" / split
    labels_dir = yolo_dir / "labels" / split

    if not images_dir.exists():
        print(f"[WARN] images/{split} not found — skipping.")
        return []

    image_files = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

    if not image_files:
        print(f"[WARN] No images found in {images_dir} — skipping.")
        return []

    all_records: list[dict] = []

    # counter = 0
    for image_path in tqdm(image_files, desc=f"  [{split}]"):
        image_path_str = str(image_path)
        file_name = image_path.stem
        # counter = counter + 1
        # if (counter > 15): 
        #     break

        # ---- Load image ------------------------------------------------
        image_bgr = cv2.imread(image_path_str)
        if image_bgr is None:
            print(f"[WARN] Could not read {image_path_str} — skipping.")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w = image_rgb.shape[:2]

        # ---- Parse YOLO annotations ------------------------------------
        label_path = labels_dir / (image_path.stem + ".txt")
        boxes = parse_yolo_annotation(label_path, img_w, img_h)

        if len(boxes) == 0:
            continue

        # ---- UNet segmentation -----------------------------------------
        try:
            segmentation_masks = unet_inference(
                model,
                boxes,
                image_path=image_path_str,
                image_rgb=image_rgb,
                device=device,
                output_to_file=False,
                output_name="yolo_dataset_processing",
                expand_boxes=False,
                scan_folder=split,
            )
        except Exception as exc:
            print(f"[ERROR] UNet failed on {image_path_str}: {exc}")
            continue

        if not segmentation_masks:
            continue

        # ---- CRF / component connection --------------------------------
        try:
            connected_masks = connect_all_masks(
                segmentation_masks,
                image_bgr,
                image_path_str,
                output_to_file=False,
                output_name="yolo_dataset_processing",
                scan_folder=split,
            )
        except Exception as exc:
            print(f"[ERROR] CRF failed on {image_path_str}: {exc}")
            connected_masks = segmentation_masks

        # ---- Gaussian soma filter --------------------------------------
        try:
            soma_masks = get_gaussian_filter_soma_masks(
                boxes,
                image_path_str,
                image_rgb,
                output_name="yolo_dataset_processing",
                output_to_file=False,
                scan_folder=split,
            )
        except Exception as exc:
            print(f"[ERROR] Soma filter failed on {image_path_str}: {exc}")
            # Fall back to zero soma masks
            soma_masks = [
                np.zeros(image_rgb.shape[:2], dtype=np.uint8)
                for _ in range(len(connected_masks))
            ]

        # ---- Skeletonisation -------------------------------------------
        try:
            skeletons = get_skeletons(
                image_rgb,
                image_path_str,
                connected_masks,
                soma_masks,
                output_to_file=False,
                output_name="yolo_dataset_processing",
                scan_folder=split,
            )
        except Exception as exc:
            print(f"[ERROR] Skeleton failed on {image_path_str}: {exc}")
            continue

        # ---- Morphological features ------------------------------------
        try:
            results_df = get_morphology_dataframe(
                connected_masks,
                skeletons,
                soma_masks,
                boxes,
            )
        except Exception as exc:
            print(f"[ERROR] Morphology features failed on {image_path_str}: {exc}")
            continue

        # ---- Attach metadata -------------------------------------------
        results_df["image_name"] = file_name
        results_df["split"] = split
        results_df["global_cell_id"] = (
            file_name + "_cell_" + results_df["cell_id"].astype(str)
        )

        all_records.append(results_df)

    if not all_records:
        return []

    return all_records


def main():
    overall_start = time.perf_counter()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"YOLO dataset : {YOLO_DATASET_DIR}")
    print(f"Output CSV   : {OUTPUT_CSV}")
    print()

    # Load UNet model once
    print(f"Loading UNet from {UNET_CHECKPOINT} …")
    model = load_model(UNET_CHECKPOINT, device)
    print("Model loaded.\n")

    all_split_records: list[pd.DataFrame] = []

    for split in SPLITS:
        print(f"Processing split: {split}")
        split_records = process_split(split, YOLO_DATASET_DIR, model, device)
        if split_records:
            all_split_records.extend(split_records)
            cell_count = sum(len(df) for df in split_records)
            print(f"  → {cell_count} cells from {len(split_records)} images\n")
        else:
            print(f"  → 0 cells\n")

    if not all_split_records:
        print("[ERROR] No results produced. Check dataset path and annotations.")
        return

    final_df = pd.concat(all_split_records, ignore_index=True)

    # Ensure output directory exists
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(str(OUTPUT_CSV), index=False)

    elapsed = time.perf_counter() - overall_start
    print("=" * 60)
    print(f"Total cells processed : {len(final_df)}")
    print(f"Output saved to       : {OUTPUT_CSV}")
    print(f"Total runtime         : {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
