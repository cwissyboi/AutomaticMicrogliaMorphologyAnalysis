import sys
import platform
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
_OUTPUT_BASE = _SCRIPT_DIR / "object_detection_output"

BOX_BUFFER = 0.20  # expand each box by 20% in all directions
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


def print_timing_summary(timings, total_images, total_detections):
    print(f"\n========== TIMING SUMMARY (first {total_images} images, {total_detections} detections) ==========")
    total = sum(timings.values())
    for k, v in timings.items():
        pct = (v / total * 100) if total > 0 else 0
        print(f"  {k:<20}: {v:7.2f}s  ({pct:.1f}%)")
    print(f"  {'TOTAL':<20}: {total:7.2f}s")


def main():
    overall_start = time.perf_counter()
    args = parse_args()
    input_folder_path = args["input_folder_path"]
    output_name = args["output_name"]

    yolo = YOLO(str(_MODEL_PATH))

    timings = {"yolo_inference": 0.0, "imread": 0.0, "crop_and_save": 0.0}

    input_dir = Path(input_folder_path)
    scan_folders = [p for p in input_dir.iterdir() if p.is_dir()]
    total_images = 0
    total_detections = 0

    scan_folder_count = 0
    for scan_folder in scan_folders:
        scan_folder_name = scan_folder.name
        image_paths = list(scan_folder.iterdir())
        scan_folder_count = scan_folder_count + 1

        if (scan_folder_count > 10):
            break

        for image_path in tqdm(image_paths, desc=f"Images in {scan_folder_name}"):
            if not image_path.is_file():
                continue

            image_path_str = str(image_path)

            # Run YOLO (single inference call)
            t0 = time.perf_counter()
            results = yolo(image_path_str)
            r = results[0]

            total_images += 1

            if r.boxes is None or len(r.boxes) == 0:
                timings["yolo_inference"] += time.perf_counter() - t0
                continue

            conf = r.boxes.conf
            keep = conf > 0.5
            yolo_boxes = r.boxes.xyxy[keep].cpu().numpy()
            timings["yolo_inference"] += time.perf_counter() - t0

            if len(yolo_boxes) == 0:
                continue

            t0 = time.perf_counter()
            image = cv2.imread(image_path_str)
            timings["imread"] += time.perf_counter() - t0

            img_h, img_w = image.shape[:2]
            file_name = get_file_name(image_path_str)
            output_folder = _OUTPUT_BASE / output_name / scan_folder_name
            output_folder.mkdir(parents=True, exist_ok=True)

            t0 = time.perf_counter()
            for box in yolo_boxes:
                x1, y1, x2, y2 = box

                # Apply 20% buffer, clamped to image bounds
                bw = x2 - x1
                bh = y2 - y1
                x1b = int(max(0,     x1 - bw * BOX_BUFFER))
                y1b = int(max(0,     y1 - bh * BOX_BUFFER))
                x2b = int(min(img_w, x2 + bw * BOX_BUFFER))
                y2b = int(min(img_h, y2 + bh * BOX_BUFFER))

                crop = image[y1b:y2b, x1b:x2b]

                # Name encodes the source tile and box position
                out_path = output_folder / f"{file_name}_yolo_identification_x{x1b}_y{y1b}.jpg"
                save_image(out_path, crop)
            timings["crop_and_save"] += time.perf_counter() - t0

    
            total_detections += len(yolo_boxes)



    overall_time = time.perf_counter() - overall_start
    print(f"\nProcessed {total_images} images, {total_detections} total detections.")
    print_timing_summary(timings, total_images, total_detections)
    print(f"\nTotal wall time: {overall_time:.2f}s")


if __name__ == "__main__":
    main()
