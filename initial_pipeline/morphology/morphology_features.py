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


def compute_junction_count(skeleton):
    """
    Count skeleton junction points (pixels with ≥ 3 neighbors).
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


def compute_end_nodes(skeleton, soma_mask):
    """
    Count branch tip pixels: skeleton pixels with exactly 1 neighbor that are
    NOT adjacent to the soma.

    A tip that sits next to the soma boundary is a start node (the point where
    a branch leaves the cell body), not a free end.  This function excludes
    those so only true free endings are counted.

    Parameters
    ----------
    skeleton : numpy.ndarray
        2D boolean skeleton (process skeleton, soma region already removed).
    soma_mask : numpy.ndarray
        2D binary soma mask (bool or 0/1).

    Returns
    -------
    int
        Number of free branch-tip pixels.
    """
    kernel = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ])

    skeleton = skeleton.astype(bool)
    soma_bin = soma_mask > 0

    neighbors = convolve(skeleton.astype(np.uint8), kernel, mode="constant", cval=0)

    # Tip pixels: on the skeleton and have exactly 1 neighbor
    tip_pixels = skeleton & (neighbors == 1)

    # Dilate soma by 1 pixel to catch skeleton pixels that directly border it
    soma_dilated = convolve(soma_bin.astype(np.uint8), kernel, mode="constant", cval=0) > 0

    # Exclude tips that touch the soma
    free_tips = tip_pixels & (~soma_dilated)

    return int(free_tips.sum())


def compute_start_nodes(skeleton, soma_mask):
    """
    Count soma-attachment pixels: skeleton pixels with exactly 1 neighbor that
    ARE adjacent to the soma.

    These are the points where each branch process departs from the cell body —
    the opposite of free branch tips.

    Parameters
    ----------
    skeleton : numpy.ndarray
        2D boolean skeleton (process skeleton, soma region already removed).
    soma_mask : numpy.ndarray
        2D binary soma mask (bool or 0/1).

    Returns
    -------
    int
        Number of soma-attachment tip pixels.
    """
    kernel = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ])

    skeleton = skeleton.astype(bool)
    soma_bin = soma_mask > 0

    neighbors = convolve(skeleton.astype(np.uint8), kernel, mode="constant", cval=0)

    # Tip pixels: on the skeleton and have exactly 1 neighbor
    tip_pixels = skeleton & (neighbors == 1)

    # Dilate soma by 1 pixel to catch skeleton pixels that directly border it
    soma_dilated = convolve(soma_bin.astype(np.uint8), kernel, mode="constant", cval=0) > 0

    # Keep only tips that touch the soma
    soma_tips = tip_pixels & soma_dilated

    return int(soma_tips.sum())


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





def compute_branch_mask(cell_mask, soma_mask):
    """
    Compute the branch mask: pixels that are inside the cell but outside the soma.

    Parameters
    ----------
    cell_mask : numpy.ndarray
        2D binary mask of the full cell (bool or 0/1 or 0/255).
    soma_mask : numpy.ndarray
        2D binary mask of the soma (bool or 0/1 or 0/255).

    Returns
    -------
    numpy.ndarray
        Boolean mask where True indicates a branch (process) pixel.
    """
    cell_bin = cell_mask > 0
    soma_bin = soma_mask > 0
    return cell_bin & (~soma_bin)


def compute_branch_area(cell_mask, soma_mask):
    """
    Area of the branch region (cell minus soma) in pixels.

    Parameters
    ----------
    cell_mask : numpy.ndarray
        2D binary mask of the full cell (bool or 0/1 or 0/255).
    soma_mask : numpy.ndarray
        2D binary mask of the soma (bool or 0/1 or 0/255).

    Returns
    -------
    int
        Number of pixels belonging to branches.
    """
    branch_mask = compute_branch_mask(cell_mask, soma_mask)
    return int(branch_mask.sum())


def compute_branch_perimeter(cell_mask, soma_mask):
    """
    Perimeter of the branch mask (cell minus soma) in pixels.

    Parameters
    ----------
    cell_mask : numpy.ndarray
        2D binary mask of the full cell (bool or 0/1 or 0/255).
    soma_mask : numpy.ndarray
        2D binary mask of the soma (bool or 0/1 or 0/255).

    Returns
    -------
    float
        Total perimeter length (in pixels) of all branch contours.
    """
    branch_mask = compute_branch_mask(cell_mask, soma_mask)
    return compute_mask_perimeter(branch_mask)


def get_morphological_features(mask, skeleton=None, soma_mask=None):
    """
    mask: boolean or 0/1 numpy array (full cell mask)
    skeleton: optional precomputed skeleton (bool)
    soma_mask: boolean or 0/1 numpy array (soma region)

    returns:
        skeleton_length: int
        num_branches: int
        soma_area: int (pixels)
        num_junctions: int (junction pixels in skeleton)
        num_components: int (disconnected skeleton components)
    """

    # -------------------------
    # Skeleton
    # -------------------------
    if skeleton is None:
        skeleton = skeletonize(mask > 0)

    skeleton = skeleton.astype(bool)



    skeleton_length = compute_skeleton_length(skeleton)
    num_junctions = compute_junction_count(skeleton)
    num_components = compute_skeleton_components(skeleton)
    num_end_nodes = compute_end_nodes(skeleton, soma_mask)
    num_start_nodes = compute_start_nodes(skeleton, soma_mask)
    
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

    branch_area = compute_branch_area(mask, soma_mask)
    branch_perimeter = compute_branch_perimeter(mask, soma_mask)

    cell_convex_circularity = (4 * math.pi * cell_convex_hull_area) / (cell_convex_hull_perimeter ** 2)
    end_to_start_ratio = num_end_nodes / num_start_nodes if num_start_nodes > 0 else 0.0
    total_nodes = num_end_nodes + num_start_nodes

    return (
        skeleton_length, 
        num_junctions, 
        num_components,
        num_end_nodes,
        num_start_nodes,
        total_nodes,
        end_to_start_ratio,
        soma_area, 
        soma_perimeter, 
        soma_circularity, 
        cell_area, 
        cell_perimeter, 
        cell_convex_hull_area, 
        cell_convex_hull_perimeter, 
        cell_solidity, 
        cell_convexity, 
        cell_circularity,
        cell_convex_circularity,
        branch_area,
        branch_perimeter,
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
        - skeleton_length
        - num_junctions
        - num_components
        - num_end_nodes
        - num_start_nodes
        - total_nodes
        - end_to_start_ratio
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
        - cell_convex_circularity
        - branch_area
        - branch_perimeter
    """

    records = []

    for i, (mask, skeleton, soma_mask, box) in enumerate(
        zip(cell_masks, skeletons, soma_masks, boxes)
    ):
        (
            skeleton_length,
            num_junctions,
            num_components,
            num_end_nodes,
            num_start_nodes,
            total_nodes,
            end_to_start_ratio,
            soma_area,
            soma_perimeter,
            soma_circularity,
            cell_area,
            cell_perimeter,
            cell_convex_hull_area,
            cell_convex_hull_perimeter,
            cell_solidity,
            cell_convexity,
            cell_circularity,
            cell_convex_circularity,
            branch_area,
            branch_perimeter,
        ) = get_morphological_features(mask, skeleton, soma_mask)

        x_min, y_min, x_max, y_max = box


        record = {
            "cell_id": i,
            "skeleton_length": skeleton_length,
            "num_junctions": num_junctions,
            "num_components": num_components,
            "num_end_nodes": num_end_nodes,
            "num_start_nodes": num_start_nodes,
            "total_nodes": total_nodes,
            "end_to_start_ratio": end_to_start_ratio,
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
            "cell_convex_circularity": cell_convex_circularity,
            "branch_area": branch_area,
            "branch_perimeter": branch_perimeter,
            "xmin": x_min,
            "ymin": y_min,
            "xmax": x_max,
            "ymax": y_max,
        }

        records.append(record)

    return pd.DataFrame(records)
