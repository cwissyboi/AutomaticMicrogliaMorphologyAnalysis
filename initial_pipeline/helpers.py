import argparse
from pathlib import Path


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