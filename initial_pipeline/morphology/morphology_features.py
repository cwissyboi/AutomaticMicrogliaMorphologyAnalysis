import numpy as np
import cv2
import os
from skimage.morphology import skeletonize
from helpers import get_file_name
from scipy.ndimage import convolve, label
import pandas as pd

def get_skeletons(image_rgb, image_path, cell_masks, soma_masks, output_name,  output_to_file = False, output_folder = 'morphology/skeleton_outputs/'): 

    overlay = image_rgb.copy()
    skeletons = []

    for mask, soma_mask in zip(cell_masks, soma_masks):
    
        # 1. Skeletonise full cell
 
        cell_bin = mask > 0
        skeleton = skeletonize(cell_bin)

        # Remove skeleton inside soma
        soma_bin = soma_mask > 0
        process_skeleton = skeleton & (~soma_bin)
        skeletons.append(process_skeleton)



        if (output_to_file):
            # Draw process skeleton (GREEN)
            ys, xs = np.where(process_skeleton)
            overlay[ys, xs] = [0, 255, 0]

         
            # 2. Draw soma outline
        
            soma_uint8 = (soma_bin.astype(np.uint8) * 255)

            contours, _ = cv2.findContours(
                soma_uint8,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            # Draw soma contours (BLUE)
            cv2.drawContours(
                overlay,
                contours,
                contourIdx=-1,
                color=(0, 0, 255),
                thickness=2
            )

    if (output_to_file):
        # Save combined overlay
        file_name = get_file_name(image_path)
        out_path = f"{output_folder}{output_name}/{file_name}_skeleton.png"
        cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return skeletons



def compute_skeleton_length(skeleton):
    """
    Number of skeleton pixels.
    """
    return int(skeleton.sum())


def compute_branch_count(skeleton):
    """
    Count skeleton branch points (pixels with ≥ 3 neighbors).
    """
    kernel = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ])

    neighbors = convolve(
        skeleton.astype(np.uint8),
        kernel,
        mode="constant",
        cval=0
    )

    branch_points = np.logical_and(skeleton, neighbors >= 3)
    return int(branch_points.sum())


def compute_skeleton_components(skeleton):
    """
    Number of connected components in the skeleton (8-connectivity).
    """
    structure = np.ones((3, 3), dtype=np.uint8)
    _, num_components = label(skeleton, structure=structure)
    return int(num_components)



def compute_mask_area(mask):
    """
    Area of soma region in pixels.
    """
    if mask is None:
        return 0
    return int((mask > 0).sum())




def get_morphological_features(mask, skeleton=None, soma_mask=None):
    """
    mask: boolean or 0/1 numpy array (full cell mask)
    skeleton: optional precomputed skeleton (bool)
    soma_mask: boolean or 0/1 numpy array (soma region)

    returns:
        length_pixels: int
        num_branches: int
        soma_area: int (pixels)
        num_components: int (disconnected skeleton components)
    """

    # -------------------------
    # Skeleton
    # -------------------------
    if skeleton is None:
        skeleton = skeletonize(mask > 0)

    skeleton = skeleton.astype(bool)



    length_pixels = compute_skeleton_length(skeleton)
    num_branches = compute_branch_count(skeleton)
    soma_area = compute_mask_area(soma_mask)
    num_components = compute_skeleton_components(skeleton)
    cell_area = compute_mask_area(mask)

    return (
        length_pixels,
        num_branches,
        soma_area,
        num_components,
        cell_area
    )



def get_morphology_dataframe(
    cell_masks,
    skeletons,
    soma_masks
):
    """
    Compute morphological features for a set of segmented cells and
    return the results as a pandas DataFrame.

    Parameters
    ----------
    cell_masks : list of numpy.ndarray
        List of full cell segmentation masks (2D, binary).

    skeletons : list of numpy.ndarray
        List of skeletonized masks corresponding to `sam_masks`.

    soma_masks : list of numpy.ndarray
        List of soma masks (2D, binary), one per cell.

    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per cell and the following columns:
        - cell_id
        - length_pixels
        - num_branches
        - soma_area
        - num_components
        - cell_area
    """

    records = []

    for i, (mask, skeleton, soma_mask) in enumerate(
        zip(cell_masks, skeletons, soma_masks)
    ):
        (
            length_px,
            num_branches,
            soma_area,
            num_components,
            cell_area
        ) = get_morphological_features(mask, skeleton, soma_mask)

        record = {
            "cell_id": i,
            "length_pixels": length_px,
            "num_branches": num_branches,
            "soma_area": soma_area,
            "num_components": num_components,
            "cell_area": cell_area,
        }

        records.append(record)

    return pd.DataFrame(records)
