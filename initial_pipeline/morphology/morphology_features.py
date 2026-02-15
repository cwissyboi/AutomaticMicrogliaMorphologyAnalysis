import numpy as np
import cv2
import os
from skimage.morphology import skeletonize
from helpers import get_file_name
from scipy.ndimage import convolve, label
import pandas as pd
import math


def get_skeletons(image_rgb, image_path, cell_masks, soma_masks, output_name, scan_folder,   output_to_file = False, output_folder = 'morphology/skeleton_outputs/'): 

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
        out_path = f"{output_folder}{output_name}/{scan_folder}/{file_name}_skeleton.png"
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



def compute_mask_perimeter(mask):
    """
    Compute the perimeter of a binary mask in pixels.

    Parameters
    ----------
    mask : numpy.ndarray
        2D binary mask (bool or 0/1 or 0/255).

    Returns
    -------
    float
        Perimeter length in pixels.
    """

    # Ensure uint8 binary mask
    mask_uint8 = (mask > 0).astype(np.uint8) * 255

    # Find external contours
    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    # Sum perimeters of all contours
    perimeter = sum(
        cv2.arcLength(cnt, closed=True)
        for cnt in contours
    )

    return float(perimeter)


def compute_convex_hull(mask):
    """
    Compute the convex hull contour of a binary mask.

    Parameters
    ----------
    mask : numpy.ndarray
        2D binary mask (bool or 0/1 or 0/255).

    Returns
    -------
    numpy.ndarray or None
        Convex hull contour (Nx1x2), or None if mask is empty.
    """

    mask_uint8 = (mask > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    # Merge all contours into one
    all_points = np.vstack(contours)

    hull = cv2.convexHull(all_points)
    return hull



def compute_convex_hull_area(mask):
    """
    Compute convex hull area (pixels²) from a binary mask.
    """
    hull = compute_convex_hull(mask)
    if hull is None:
        return 0.0

    return float(cv2.contourArea(hull))


def compute_convex_hull_perimeter(mask):
    """
    Compute convex hull perimeter (pixels) from a binary mask.
    """
    hull = compute_convex_hull(mask)
    if hull is None:
        return 0.0

    return float(cv2.arcLength(hull, closed=True))





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
    num_components = compute_skeleton_components(skeleton)
    
    soma_area = compute_mask_area(soma_mask)
    soma_perimeter = compute_mask_perimeter(soma_mask)
    soma_circularity = (4 * math.pi * soma_area) / (soma_perimeter ** 2)


    cell_area = compute_mask_area(mask)
    cell_perimeter = compute_mask_perimeter(mask)
    cell_convex_hull_area = compute_convex_hull_area(mask)
    cell_convex_hull_perimeter = compute_convex_hull_perimeter(mask)
    cell_solidity = cell_area / cell_convex_hull_area
    cell_convexity = cell_convex_hull_perimeter / cell_perimeter
    cell_circularity = (4 * math.pi * cell_area) / (cell_perimeter ** 2)


    return (
        length_pixels, 
        num_branches, 
        num_components, 
        soma_area, 
        soma_perimeter, 
        soma_circularity, 
        cell_area, 
        cell_perimeter, 
        cell_convex_hull_area, 
        cell_convex_hull_perimeter, 
        cell_solidity, 
        cell_convexity, 
        cell_circularity
    )




def get_morphology_dataframe(
    cell_masks,
    skeletons,
    soma_masks, 
    boxes
):
    """
    Compute morphological features for a set of segmented cells and
    return the results as a pandas DataFrame.

    Parameters
    ----------
    cell_masks : list of numpy.ndarray
        List of full cell segmentation masks (2D, binary).

    skeletons : list of numpy.ndarray
        List of skeletonized masks corresponding to `cell_masks`.

    soma_masks : list of numpy.ndarray
        List of soma masks (2D, binary), one per cell.

    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per cell and the following columns:

        - cell_id
        - length_pixels
        - num_branches
        - num_components
        - soma_area
        - soma_perimeter
        - soma_circularity
        - cell_area
        - cell_perimeter
        - cell_convex_hull_area
        - cell_convex_hull_perimeter
        - cell_solidity
        - cell_convexity
        - cell_circularity
    """

    records = []

    for i, (mask, skeleton, soma_mask, box) in enumerate(
        zip(cell_masks, skeletons, soma_masks, boxes)
    ):
        (
            length_pixels,
            num_branches,
            num_components,
            soma_area,
            soma_perimeter,
            soma_circularity,
            cell_area,
            cell_perimeter,
            cell_convex_hull_area,
            cell_convex_hull_perimeter,
            cell_solidity,
            cell_convexity,
            cell_circularity
        ) = get_morphological_features(mask, skeleton, soma_mask)

        x_min, y_min, x_max, y_max = box


        record = {
            "cell_id": i,
            "length_pixels": length_pixels,
            "num_branches": num_branches,
            "num_components": num_components,
            "soma_area": soma_area,
            "soma_perimeter": soma_perimeter,
            "soma_circularity": soma_circularity,
            "cell_area": cell_area,
            "cell_perimeter": cell_perimeter,
            "cell_convex_hull_area": cell_convex_hull_area,
            "cell_convex_hull_perimeter": cell_convex_hull_perimeter,
            "cell_solidity": cell_solidity,
            "cell_convexity": cell_convexity,
            "cell_circularity": cell_circularity,
            "xmin": x_min,
            "ymin": y_min,
            "xmax": x_max,
            "ymax": y_max,
        }

        records.append(record)

    return pd.DataFrame(records)
