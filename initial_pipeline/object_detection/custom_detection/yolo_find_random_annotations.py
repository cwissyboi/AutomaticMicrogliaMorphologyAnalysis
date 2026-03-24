"""
yolo_find_random_annotations.py

Runs YOLO on a random subset of an unlabelled dataset and exports the result as
a valid Ultralytics/YOLO dataset for manual review.

Output structure:
    <output_dir>/
        train.txt
        images/
            train/
                <image_name>.jpg
        labels/
            train/
                <image_name>.txt   # one line per box: <cls> <cx> <cy> <w> <h>
        data.yaml

The random subset size is controlled by --max_images. Only images with at least
one kept detection are exported.
"""

import sys
import platform
import random
import numpy as np
from pathlib import Path

# Add initial_pipeline/ to sys.path so helpers.py and sub-packages are importable
_INITIAL_PIPELINE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_INITIAL_PIPELINE_DIR))

import cv2
from tqdm import tqdm
import time
from ultralytics import YOLO

from helpers import parse_args, get_file_name

# Paths relative to custom_detection/ (this file's directory)
_SCRIPT_DIR = Path(__file__).resolve().parent
_MODEL_PATH = _SCRIPT_DIR / "yolo_good_runs" / "28_2_rat_pretraining.pt"
_OUTPUT_BASE = _SCRIPT_DIR / "random_annotation_output"

# For very large datasets, only consider this many images for random sampling
MAX_CANDIDATE_IMAGES = 100000

# Keep boxes at or above this confidence for proposed labels
CONF_LOW = 0.5

# Run YOLO with a low conf so we can filter ourselves
YOLO_CONF_THRESHOLD = 0.01

# \\?\ prefix lifts the 260-char MAX_PATH limit on Windows; not needed/valid on Linux
_LONG_PATH_PREFIX = "\\\\?\\" if platform.system() == "Windows" else ""


def imread_long_path(path_str: str):
    """cv2.imread fails silently on Windows paths > 260 chars - use imdecode instead."""
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


def write_data_yaml(output_dir: Path):
    """Write a minimal data.yaml so the output is a self-contained YOLO dataset."""
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write("path: .\n")
        f.write("train: train.txt\n")
        f.write("val: train.txt\n")
        f.write("\n")
        f.write("nc: 1\n")
        f.write("names:\n")
        f.write("  0: microglia\n")
    print(f"Wrote {yaml_path}")


def write_train_txt(output_dir: Path, image_rel_paths):
    """Write train.txt with one image path per line, relative to dataset root."""
    train_txt_path = output_dir / "train.txt"
    with open(train_txt_path, "w") as f:
        for rel_path in image_rel_paths:
            f.write(f"{rel_path}\n")
    print(f"Wrote {train_txt_path}")


def print_timing_summary(timings, processed_images, saved_images):
    print(f"\n========== TIMING SUMMARY ({processed_images} images processed, {saved_images} saved) ==========")
    total = sum(timings.values())
    for k, v in timings.items():
        pct = (v / total * 100) if total > 0 else 0
        print(f"  {k:<20}: {v:7.2f}s  ({pct:.1f}%)")
    print(f"  {'TOTAL':<20}: {total:7.2f}s")


def collect_all_images(input_dir: Path, candidate_limit: int):
    """Collect files up to candidate_limit as (scan_folder_name, image_path)."""
    candidates = []
    scan_folders = [p for p in input_dir.iterdir() if p.is_dir()]
    for scan_folder in scan_folders:
        for image_path in scan_folder.iterdir():
            if image_path.is_file():
                candidates.append((scan_folder.name, image_path))
                if len(candidates) >= candidate_limit:
                    return candidates
    return candidates


def main():
    overall_start = time.perf_counter()
    args = parse_args()
    input_folder_path = args["input_folder_path"]
    output_name = args.get("output_name", "random_review")
    max_images = args.get("max_images")

    yolo = YOLO(str(_MODEL_PATH))

    timings = {"yolo_inference": 0.0, "imread": 0.0, "save": 0.0}

    input_dir = Path(input_folder_path)
    output_dir = _OUTPUT_BASE / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    all_candidates = collect_all_images(input_dir, MAX_CANDIDATE_IMAGES)
    total_candidates = len(all_candidates)

    if total_candidates == 0:
        print(f"No candidate images found under: {input_dir.resolve()}")
        return

    random.shuffle(all_candidates)
    target_exports = max_images if max_images is not None else total_candidates

    total_processed = 0
    total_saved = 0
    total_no_box = 0
    exported_image_rel_paths = []

    for _, image_path in tqdm(all_candidates, desc="Processing random sample"):
        image_path_str = str(image_path.resolve())

        t0 = time.perf_counter()
        results = yolo(image_path_str, conf=YOLO_CONF_THRESHOLD, verbose=False)
        r = results[0]
        timings["yolo_inference"] += time.perf_counter() - t0
        total_processed += 1

        if r.boxes is None or len(r.boxes) == 0:
            all_boxes_xyxy = np.empty((0, 4), dtype=float)
        else:
            conf = r.boxes.conf.cpu().numpy()
            keep = conf > CONF_LOW
            all_boxes_xyxy = r.boxes.xyxy[keep].cpu().numpy()

        if len(all_boxes_xyxy) == 0:
            total_no_box += 1
            continue

        t0 = time.perf_counter()
        image = imread_long_path(image_path_str)
        timings["imread"] += time.perf_counter() - t0

        if image is None:
            print(f"WARNING: could not read {image_path_str}")
            continue

        img_h, img_w = image.shape[:2]
        file_name = get_file_name(image_path_str)

        t0 = time.perf_counter()

        img_out_dir = output_dir / "images" / "train"
        label_out_dir = output_dir / "labels" / "train"
        img_out_dir.mkdir(parents=True, exist_ok=True)
        label_out_dir.mkdir(parents=True, exist_ok=True)

        img_out_path = img_out_dir / f"{file_name}.jpg"
        label_out_path = label_out_dir / f"{file_name}.txt"

        save_image(img_out_path, image)
        write_label_file(label_out_path, all_boxes_xyxy, img_w, img_h)
        exported_image_rel_paths.append(f"images/train/{img_out_path.name}")

        timings["save"] += time.perf_counter() - t0
        total_saved += 1

        if total_saved >= target_exports:
            break

    write_train_txt(output_dir, exported_image_rel_paths)
    write_data_yaml(output_dir)

    overall_time = time.perf_counter() - overall_start
    print(f"\nFound {total_candidates} candidate images.")
    print(f"Randomized candidate order and evaluated {total_processed} images.")
    print(f"Skipped {total_no_box} images with no kept detections.")
    print(f"Saved {total_saved} images and labels to: {output_dir.resolve()}")
    print(f"Label confidence filter: conf > {CONF_LOW}")
    if max_images is not None and total_saved < max_images:
        print(
            f"WARNING: Requested max_images={max_images}, but only {total_saved} images "
            f"with kept detections were found in the first {total_candidates} candidates."
        )
    print_timing_summary(timings, total_processed, total_saved)
    print(f"\nTotal wall time: {overall_time:.2f}s")


if __name__ == "__main__":
    main()
