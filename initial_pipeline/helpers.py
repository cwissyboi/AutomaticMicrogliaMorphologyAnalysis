import argparse
from pathlib import Path
import os
import cv2
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
        "--output_folder_path",
        type=str,
        default="output/",
        help="Root output directory"
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


def output_masks_to_file_overlay(output_folder, image_path, masks, image_rgb, suffix = 'sam_outline'):
    os.makedirs(output_folder, exist_ok=True)

    # Copy original image (RGB)
    overlay = image_rgb.copy()

    for i, mask in enumerate(masks):
        # soma_mask is uint8 {0,255} or bool, ensure uint8
        mask_uint8 = (mask > 0).astype(np.uint8) * 255

        # Find contours of soma
        contours, _ = cv2.findContours(
            mask_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Draw soma contours (blue, slightly thicker)
        cv2.drawContours(
            overlay,
            contours,
            contourIdx=-1,
            color=(0, 0, 255),  # blue soma
            thickness=2
        )

    # Save result
    file_name = get_file_name(image_path)
    out_path = f"{output_folder}{file_name}_{suffix}.jpg"
    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))