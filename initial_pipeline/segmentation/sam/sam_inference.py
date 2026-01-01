import torch
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import os
from helpers import get_file_name, output_masks_to_file_overlay

def sam_inference(sam_predictor, boxes, image_path, image_rgb, 
                  prompt_type="point", output_to_file = False, 
                  output_folder = "segmentation/segmentation_output/sam_segmentation/"):
    """
    Run Segment Anything Model (SAM) inference using either point-based
    or box-based prompts derived from YOLO bounding boxes.

    This function loads an image, sets it as the active image for the
    SAM predictor, and generates a segmentation mask for each bounding
    box using either:
      - a single foreground point at the box center, or
      - the full bounding box as a prompt.

    Parameters
    ----------
    sam_predictor : SamPredictor
        An initialized Segment Anything predictor instance with a loaded
        SAM model.

    boxes : numpy.ndarray
        Array of bounding boxes with shape (N, 4), where each box is
        defined as [x1, y1, x2, y2] in pixel coordinates.

    image_path : str
        Path to the input image on which segmentation will be performed.

    prompt_type : str, optional
        Type of SAM prompt to use. Options:
        - "point": Use the center point of each bounding box as a
          foreground prompt (default).
        - any other value: Use the entire bounding box as the prompt.

    Returns
    -------
    list of numpy.ndarray
        List of binary segmentation masks, one per input bounding box.
        Each mask has shape (H, W) and corresponds to the original
        image dimensions.
    """
    
    sam_predictor.set_image(image_rgb)

    masks = []

    for box in boxes:
        x1, y1, x2, y2 = box.astype(int)

        if prompt_type == "point":
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2


            mask, _, _ = sam_predictor.predict(
                point_coords=np.array([[cx, cy]]),
                point_labels=np.array([1]),  # foreground
                multimask_output=False
            )

        else:
            # by default use the entire box as the prompt
            mask, _, _ = sam_predictor.predict(
                box=np.array([x1, y1, x2, y2]),
                multimask_output=False
            )

        masks.append(mask[0])  # (H, W)

    
    
    if (output_to_file):
        output_masks_to_file_overlay(output_folder, image_path, masks, image_rgb, suffix = 'sam_outline')


    print('SAM done')
    return masks