import argparse
from pathlib import Path
import os
import cv2
import numpy as np
import colorsys
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO inference arguments")

    # Input / output
    parser.add_argument(
        "--input_folder_path",
        type=str,
        default="input/",
        help="Root input directory"
    )

    parser.add_argument(
        "--image_path",
        type=str,
        default="input/example_image.jpeg",
        help="path to only 1 image that you would like to do an analysis on"
    )

    parser.add_argument(
        "--output_name",
        type=str,
        default="output",
        help="will be used as both a directory and csv file name"
    )

    parser.add_argument(
        "--scan_name",
        type=str,
        default=None,
        help="Name of the scan being processed; included in the output CSV filename"
    )

    # YOLO files
    parser.add_argument(
        "--cfg",
        type=str,
        default="yolo_files/yolov3.cfg",
        help="Path to YOLO config file"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolo_files/yolov3.weights",
        help="Path to YOLO weights file"
    )
    parser.add_argument(
        "--classes",
        type=str,
        default="yolo_files/yolov3.txt",
        help="Path to class labels file"
    )

    # YOLO parameters
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.5,
        help="Confidence threshold"
    )
    parser.add_argument(
        "--nms_threshold",
        type=float,
        default=0.1,
        help="Non-max suppression threshold"
    )
    parser.add_argument(
        "--ignore_class_labels",
        action="store_true",
        help="Ignore class labels during inference"
    )

    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of flagged images to save (e.g. 1000). "
             "Scanning stops as soon as this limit is reached."
    )

    return vars(parser.parse_args())


def get_file_name(file_path):
    """
    Extract the base filename (without directories or file extension)
    from a file path.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to a file.

    Returns
    -------
    str
        Filename without directory and extension.

    Examples
    --------
    >>> get_stem_from_path("C:/data/images/sample_image.jpg")
    'sample_image'

    >>> get_stem_from_path("../toy_dataset/third_screenshot.png")
    'third_screenshot'
    """
    return Path(file_path).stem


def index_to_color(i, total_masks):
    """
    Deterministic distinct color for mask i.
    Evenly spaced in HSV space.
    Returns RGB tuple (0–255).
    """
    if total_masks == 0:
        return (255, 0, 0)

    hue = i / total_masks
    saturation = 0.9
    value = 1.0

    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)

    return (
        int(r * 255),
        int(g * 255),
        int(b * 255),
    )


def output_masks_to_file_overlay(
    output_folder,
    image_path,
    masks,
    image_rgb,
    suffix='sam_outline'
):
    os.makedirs(output_folder, exist_ok=True)

    overlay = image_rgb.copy()

    total_masks = len(masks)

    for i, mask in enumerate(masks):

        mask_uint8 = (mask > 0).astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            mask_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        color = index_to_color(i, total_masks)

        cv2.drawContours(
            overlay,
            contours,
            contourIdx=-1,
            color=color,   # RGB
            thickness=1
        )

    file_name = get_file_name(image_path)
    out_path = f"{output_folder}/{file_name}_{suffix}.jpg"

    # Convert RGB → BGR for cv2 saving
    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
