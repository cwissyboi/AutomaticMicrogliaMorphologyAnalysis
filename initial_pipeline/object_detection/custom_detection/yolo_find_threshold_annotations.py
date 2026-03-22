"""
yolo_find_threshold_annotations.py

Runs YOLO on an unlabelled dataset and finds images that contain at least one
detection near the confidence threshold (default: 0.45 < conf < 0.55).

Output is a valid Ultralytics/YOLO dataset:
    <output_dir>/
        images/
            <scan_folder>/
                <image_name>.jpg
        labels/
            <scan_folder>/
                <image_name>.txt   # one line per box: <cls> <cx> <cy> <w> <h>  (normalised)
        data.yaml

ALL boxes predicted for each qualifying image are written to the label file
(not just the threshold ones), so the reviewer sees full context when correcting.
"""

import sys
import platform
import numpy as np
from pathlib import Path

# Add initial_pipeline/ to sys.path so helpers.py and sub-packages are importable
_INITIAL_PIPELINE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_INITIAL_PIPELINE_DIR))

import cv2
import shutil
from tqdm import tqdm
import time
from ultralytics import YOLO

from helpers import parse_args, get_file_name

# Paths relative to custom_detection/ (this file's directory)
_SCRIPT_DIR = Path(__file__).resolve().parent
_MODEL_PATH = _SCRIPT_DIR / "yolo_good_runs" / "28_2_rat_pretraining.pt"
_OUTPUT_BASE = _SCRIPT_DIR / "threshold_annotation_output"

# Confidence band that flags an image for review
CONF_LOW  = 0.45
CONF_HIGH = 0.55

# Run YOLO with a low conf so we capture everything in the band
YOLO_CONF_THRESHOLD = 0.01

# \\?\ prefix lifts the 260-char MAX_PATH limit on Windows; not needed/valid on Linux
_LONG_PATH_PREFIX = "\\\\?\\" if platform.system() == "Windows" else ""


def imread_long_path(path_str: str):
    """cv2.imread fails silently on Windows paths > 260 chars — use imdecode instead."""
    with open(_LONG_PATH_PREFIX + path_str, "rb") as f:
        buf = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def save_image(path: Path, img_bgr):
    """Write a BGR numpy array to disk, bypassing Windows MAX_PATH (260 char) limit."""
    success, buf = cv2.imencode(".jpg", img_bgr)
    if not success:
        print(f"WARNING: imencode failed for {path}")
        return
    with open(_LONG_PATH_PREFIX + str(path.resolve()), "wb") as f:
        f.write(buf.tobytes())


def xyxy_to_yolo(box_xyxy, img_w: int, img_h: int) -> tuple:
    """Convert absolute xyxy box to normalised YOLO xywh (cx, cy, w, h)."""
    x1, y1, x2, y2 = box_xyxy
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return cx, cy, bw, bh


def write_label_file(label_path: Path, boxes_xyxy, img_w: int, img_h: int):
    """Write an Ultralytics-format label .txt file (class 0 = Microglia)."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        for box in boxes_xyxy:
            cx, cy, bw, bh = xyxy_to_yolo(box, img_w, img_h)
            f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def write_data_yaml(output_dir: Path, dataset_name: str):
    """Write a minimal data.yaml so the output is a self-contained YOLO dataset."""
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {output_dir.resolve()}\n")
        f.write("train: images\n")
        f.write("val: images\n")
        f.write("\n")
        f.write("nc: 1\n")
        f.write("names:\n")
        f.write("  0: Microglia\n")
    print(f"Wrote {yaml_path}")


def print_timing_summary(timings, total_images, total_flagged):
    print(f"\n========== TIMING SUMMARY ({total_images} images scanned, {total_flagged} flagged) ==========")
    total = sum(timings.values())
    for k, v in timings.items():
        pct = (v / total * 100) if total > 0 else 0
        print(f"  {k:<20}: {v:7.2f}s  ({pct:.1f}%)")
    print(f"  {'TOTAL':<20}: {total:7.2f}s")


def main():
    overall_start = time.perf_counter()
    args = parse_args()
    input_folder_path = args["input_folder_path"]
    output_name = args.get("output_name", "threshold_review")
    max_images = args.get("max_images")  # None means no limit

    yolo = YOLO(str(_MODEL_PATH))

    timings = {"yolo_inference": 0.0, "imread": 0.0, "save": 0.0}

    input_dir = Path(input_folder_path)
    output_dir = _OUTPUT_BASE / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    scan_folders = [p for p in input_dir.iterdir() if p.is_dir()]
    total_images = 0
    total_flagged = 0

    for scan_folder in scan_folders:
        scan_folder_name = scan_folder.name
        image_paths = list(scan_folder.iterdir())

        for image_path in tqdm(image_paths, desc=f"Scanning {scan_folder_name}"):
            if not image_path.is_file():
                continue

            # Stop early if the flagged-image limit has been reached
            if max_images is not None and total_flagged >= max_images:
                break

            image_path_str = str(image_path.resolve())

            # ------------------------------------------------------------------
            # Run YOLO with a very low threshold so we capture all boxes,
            # including the ones near the 0.5 boundary.
            # ------------------------------------------------------------------
            t0 = time.perf_counter()
            results = yolo(image_path_str, conf=YOLO_CONF_THRESHOLD, verbose=False)
            r = results[0]
            timings["yolo_inference"] += time.perf_counter() - t0

            total_images += 1

            if r.boxes is None or len(r.boxes) == 0:
                continue

            conf = r.boxes.conf.cpu().numpy()

            # Check whether any detection falls inside the threshold band
            in_band = (conf > CONF_LOW) & (conf < CONF_HIGH)
            if not in_band.any():
                continue

            # Keep ALL boxes that are at or above CONF_LOW so the reviewer has
            # full context (boxes well above threshold are shown but can be
            # trusted; boxes in the band are the ones needing attention).
            keep = conf > CONF_LOW
            all_boxes_xyxy = r.boxes.xyxy[keep].cpu().numpy()

            if len(all_boxes_xyxy) == 0:
                continue

            # ------------------------------------------------------------------
            # Load the full original image
            # ------------------------------------------------------------------
            t0 = time.perf_counter()
            image = imread_long_path(image_path_str)
            timings["imread"] += time.perf_counter() - t0

            if image is None:
                print(f"WARNING: could not read {image_path_str}")
                continue

            img_h, img_w = image.shape[:2]
            file_name = get_file_name(image_path_str)

            # ------------------------------------------------------------------
            # Save image and label into the Ultralytics dataset structure
            # ------------------------------------------------------------------
            t0 = time.perf_counter()

            img_out_dir   = output_dir / "images"  / scan_folder_name
            label_out_dir = output_dir / "labels"  / scan_folder_name
            img_out_dir.mkdir(parents=True, exist_ok=True)
            label_out_dir.mkdir(parents=True, exist_ok=True)

            img_out_path   = img_out_dir   / f"{file_name}.jpg"
            label_out_path = label_out_dir / f"{file_name}.txt"

            save_image(img_out_path, image)
            write_label_file(label_out_path, all_boxes_xyxy, img_w, img_h)

            timings["save"] += time.perf_counter() - t0
            total_flagged += 1

        # Also break out of the scan-folder loop if the limit is reached
        if max_images is not None and total_flagged >= max_images:
            print(f"\nReached max_images limit ({max_images}). Stopping early.")
            break

    # --------------------------------------------------------------------------
    # Write data.yaml at the dataset root
    # --------------------------------------------------------------------------
    write_data_yaml(output_dir, output_name)

    overall_time = time.perf_counter() - overall_start
    print(f"\nScanned {total_images} images.")
    print(f"Flagged {total_flagged} images with detections in confidence band ({CONF_LOW}, {CONF_HIGH}).")
    print(f"Output dataset: {output_dir.resolve()}")
    print_timing_summary(timings, total_images, total_flagged)
    print(f"\nTotal wall time: {overall_time:.2f}s")


if __name__ == "__main__":
    main()
