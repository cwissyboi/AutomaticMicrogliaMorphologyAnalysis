import numpy as np
import cv2
from skimage.graph import route_through_array
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import os
from helpers import get_file_name, output_masks_to_file_overlay



def compute_color_preference_cost_rgb(
    image,
    white_penalty=10.0,
    color_weight=1.0,
):
    """
    Low cost on brown/red pixels, high cost on white background.
    """
    img = image.astype(np.float32) / 255.0

    R = img[..., 0]
    G = img[..., 1]
    B = img[..., 2]

    # --- Red / brown heuristic ---
    # Brown/red ≈ high R, moderate G, low B
    redness = R - 0.5 * G - 0.5 * B
    redness = np.clip(redness, 0, 1)

    # Convert to cost (prefer red)
    color_cost = 1.0 - redness

    # --- Penalize white ---
    brightness = (R + G + B) / 3.0
    white_cost = (brightness ** 2) * white_penalty

    cost = color_weight * color_cost + white_cost
    cost += 1e-3

    # return cost / (cost.max() + 1e-6)
    return cost

def compute_smoothness_cost(image):
    img = image.astype(np.float32)

    dx = np.linalg.norm(img[:, 1:] - img[:, :-1], axis=-1)
    dx = np.pad(dx, ((0, 0), (1, 0)))

    dy = np.linalg.norm(img[1:, :] - img[:-1, :], axis=-1)
    dy = np.pad(dy, ((1, 0), (0, 0)))

    smooth = dx + dy
    return smooth / (smooth.max() + 1e-6)


def extract_components(labels, min_area_px):
    components = {}
    for label in np.unique(labels):
        if label == 0:
            continue
        ys, xs = np.where(labels == label)
        if len(xs) >= min_area_px:
            components[label] = np.column_stack([ys, xs])
    return components



def draw_thick_path(
    base_mask,
    path_pixels,
    thickness_map,
    min_radius=1,
    max_radius=10,
):
    """
    Draw a path whose thickness adapts to local component thickness.
    """

    H, W = base_mask.shape
    out = base_mask.copy()

    for y, x in path_pixels:
        r = int(np.clip(thickness_map[y, x], min_radius, max_radius))
        cv2.circle(out, (x, y), r, 1, -1)

    return out



def keep_largest_component(binary_mask):
    num_labels, labels = cv2.connectedComponents(binary_mask)

    if num_labels <= 2:
        return binary_mask

    max_area = 0
    max_label = 1

    for label in range(1, num_labels):
        area = np.sum(labels == label)
        if area > max_area:
            max_area = area
            max_label = label

    return (labels == max_label).astype(np.uint8)

def connect_components_geodesic_similarity(
    binary_mask,
    image,
    min_component_frac=0.0005,
    restrict_to_mask=False,
    thickness_scale=1.0,
    max_radius=12,
    bbox_padding=5,   # ← optional padding around mask
):
    """
    Connect components by greedily attaching the closest component
    to the current main (largest) component, updating distances after
    each merge.

    Now enforces geodesic path to remain inside mask bounding box.
    """

    H, W = binary_mask.shape
    image_area = H * W
    min_area_px = min_component_frac * image_area

    # --- Compute bounding box of mask ---
    ys, xs = np.where(binary_mask > 0)
    if len(xs) == 0:
        return binary_mask.astype(np.uint8)

    y_min = max(0, ys.min() - bbox_padding)
    y_max = min(H - 1, ys.max() + bbox_padding)
    x_min = max(0, xs.min() - bbox_padding)
    x_max = min(W - 1, xs.max() + bbox_padding)

    # --- Label components ---
    num_labels, labels = cv2.connectedComponents(binary_mask)
    if num_labels <= 2:
        return binary_mask.astype(np.uint8)

    # --- Compute cost map ---
    color_cost = compute_color_preference_cost_rgb(
        image,
        white_penalty=1000.0,
        color_weight=1.0,
    )
    smooth_cost = compute_smoothness_cost(image)
    cost = color_cost + 0.1 * smooth_cost

    cost = cost.copy()

    # --- Enforce bounding box constraint ---
    box_mask = np.zeros_like(binary_mask, dtype=bool)
    box_mask[y_min:y_max + 1, x_min:x_max + 1] = True

    cost[~box_mask] = np.inf   # ← HARD CONSTRAINT

    # --- Optional mask restriction ---
    if restrict_to_mask:
        cost[~binary_mask.astype(bool)] = np.inf

    # --- Thickness map ---
    thickness_map = distance_transform_edt(binary_mask) * thickness_scale

    # --- Extract valid components ---
    components = {}
    for label in range(1, num_labels):
        ys, xs = np.where(labels == label)
        if len(xs) >= min_area_px:
            components[label] = np.column_stack([ys, xs])

    if len(components) <= 1:
        return keep_largest_component(binary_mask).astype(np.uint8)

    # --- Find largest component ---
    areas = {label: len(pixels) for label, pixels in components.items()}
    main_label = max(areas, key=areas.get)

    main_mask = (labels == main_label).astype(np.uint8)
    connected_mask = main_mask.copy()

    remaining = set(components.keys())
    remaining.remove(main_label)

    # --- Greedy nearest-to-main loop ---
    while remaining:

        dist_to_main = distance_transform_edt(1 - main_mask)

        best_label = None
        best_dist = np.inf
        best_point_other = None

        for label in remaining:
            pixels = components[label]
            dists = dist_to_main[pixels[:, 0], pixels[:, 1]]
            idx = np.argmin(dists)

            if dists[idx] < best_dist:
                best_dist = dists[idx]
                best_label = label
                best_point_other = tuple(pixels[idx])

        main_pixels = np.column_stack(np.where(main_mask))
        main_idx = np.argmin(
            np.linalg.norm(
                main_pixels - np.array(best_point_other),
                axis=1
            )
        )
        best_point_main = tuple(main_pixels[main_idx])

        # --- Geodesic path (now constrained to box) ---
        path, _ = route_through_array(
            cost,
            best_point_main,
            best_point_other,
            fully_connected=True
        )

        connected_mask = draw_thick_path(
            connected_mask,
            path,
            thickness_map,
            min_radius=1,
            max_radius=max_radius,
        )

        main_mask = np.maximum(main_mask, connected_mask)
        main_mask[labels == best_label] = 1

        remaining.remove(best_label)

    connected_mask = keep_largest_component(main_mask)

    return connected_mask.astype(np.uint8)


import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt
from skimage.graph import route_through_array


import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt
from skimage.graph import route_through_array


def connect_components_adaptive(
    binary_mask,
    image,
    min_component_frac=0.0005,
    red_weight=3.0,
    smooth_weight=0.2,
    max_connection_distance=60,
):
    """
    Connect disconnected components inside a mask using:
    - Red-favoring geodesic routing
    - Adaptive thickness based on component thickness (robust median DT)
    - Smooth dilation-based bridge drawing
    """

    H, W = binary_mask.shape
    image_area = H * W
    min_area_px = min_component_frac * image_area

    binary_mask = binary_mask.astype(np.uint8)

    # Label components
    num_labels, labels = cv2.connectedComponents(binary_mask)
    if num_labels <= 2:
        return binary_mask

    # Extract valid components
    components = {}
    for label in range(1, num_labels):
        ys, xs = np.where(labels == label)
        if len(xs) >= min_area_px:
            components[label] = np.column_stack([ys, xs])

    if len(components) <= 1:
        return binary_mask

    # --- Build cost map favoring red ---
    image_float = image.astype(np.float32)

    # Adjust if your image is RGB instead of BGR
    red = image_float[:, :, 2]
    green = image_float[:, :, 1]
    blue = image_float[:, :, 0]

    red_score = red - 0.5 * (green + blue)
    red_score = (red_score - red_score.min()) / (red_score.ptp() + 1e-6)

    red_cost = (1.0 - red_score) * red_weight

    # Smoothness cost
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_mag = grad_mag / (grad_mag.max() + 1e-6)

    smooth_cost = smooth_weight * grad_mag

    cost = red_cost + smooth_cost

    current_mask = binary_mask.copy()

    while True:

        num_labels, labels = cv2.connectedComponents(current_mask)
        if num_labels <= 2:
            break

        components = {}
        for label in range(1, num_labels):
            ys, xs = np.where(labels == label)
            components[label] = np.column_stack([ys, xs])

        label_ids = list(components.keys())
        best_pair = None
        best_distance = np.inf

        # --- Find closest pair of components ---
        for i in range(len(label_ids)):
            for j in range(i + 1, len(label_ids)):
                comp1 = components[label_ids[i]]
                comp2 = components[label_ids[j]]

                dists = np.linalg.norm(
                    comp1[:, None, :] - comp2[None, :, :],
                    axis=2
                )

                min_dist = dists.min()

                if min_dist < best_distance:
                    best_distance = min_dist
                    idx = np.unravel_index(np.argmin(dists), dists.shape)
                    best_pair = (
                        label_ids[i],
                        label_ids[j],
                        tuple(comp1[idx[0]]),
                        tuple(comp2[idx[1]])
                    )

        if best_distance > max_connection_distance:
            break

        label1, label2, p1, p2 = best_pair

        # --- Geodesic path ---
        path, _ = route_through_array(
            cost,
            p1,
            p2,
            fully_connected=True
        )

        # --- Robust thickness estimation ---
        comp_mask1 = (labels == label1).astype(np.uint8)
        comp_mask2 = (labels == label2).astype(np.uint8)

        dt1 = distance_transform_edt(comp_mask1)
        dt2 = distance_transform_edt(comp_mask2)

        t1 = 2 * np.median(dt1[dt1 > 0])
        t2 = 2 * np.median(dt2[dt2 > 0])

        # Average thickness
        radius = int(max(1, 0.25 * (t1 + t2)))

        # --- Create thin path mask ---
        path_mask = np.zeros_like(current_mask, dtype=np.uint8)
        for (r, c) in path:
            path_mask[r, c] = 1

        # --- Dilate path to desired thickness ---
        kernel_size = int(2 * radius + 1)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size)
        )

        thick_path = cv2.dilate(path_mask, kernel)

        # Merge
        current_mask = np.maximum(current_mask, thick_path)

    return current_mask.astype(np.uint8)




def connect_all_masks(masks, image, image_path,  output_to_file = True,   
        output_name = 'temp', 
        output_folder = 'segmentation/segmentation_output/custom_segmentation/postprocessing'):
    
    connected_masks = []

    for mask in masks: 
        # connected_mask = connect_components_geodesic_similarity(mask, image)
        connected_mask = connect_components_adaptive(mask, image)
        connected_masks.append(connected_mask)


    if output_to_file:
        if output_name is None:
            output_name = get_file_name(image_path)

        out_dir = os.path.join(output_folder, output_name)
        os.makedirs(out_dir, exist_ok=True)
        # If image was loaded with cv2 (BGR), convert to RGB
        image_fixed = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        output_masks_to_file_overlay(
            out_dir,
            image_path,
            connected_masks,
            image_fixed,
            suffix="connect_masks"
        )

    return connected_masks

def remove_blue_pixels(mask, image, hue_range=(90, 140), min_saturation=0.1):
    """
    Remove mask pixels that are strongly blue.

    Parameters
    ----------
    mask : np.ndarray (H, W), uint8
        Binary mask (0/1)
    image : np.ndarray (H, W, 3), uint8 RGB
    hue_range : tuple
        Hue range (OpenCV scale 0–179) considered blue
    min_saturation : float
        Minimum saturation to consider a pixel colored

    Returns
    -------
    cleaned_mask : np.ndarray (H, W), uint8
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    h = hsv[..., 0]
    s = hsv[..., 1] / 255.0

    blue = (
        (h >= hue_range[0]) &
        (h <= hue_range[1]) &
        (s >= min_saturation)
    )

    cleaned_mask = mask.copy()
    cleaned_mask[blue] = 0

    return cleaned_mask


def compute_mask_otsu_inside_annotation(
    image_path,
    mask_path,
    connect_components=True,
    remove_blue = True, 
):
    image = np.array(Image.open(image_path).convert("RGB"))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    orig_mask = np.array(Image.open(mask_path)) > 0

    roi = gray.copy()
    roi[~orig_mask] = 255

    _, thresh = cv2.threshold(
        roi, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    mask = (thresh & orig_mask).astype(np.uint8)

    if remove_blue: 
        mask = remove_blue_pixels(mask, image)

    if connect_components:
        mask = connect_components_geodesic_similarity(mask, image = image)
    

    return mask


def add_calculated_mask_column(df):
    df = df.copy()

    df["calculated_mask"] = df.apply(
        lambda row: compute_mask_otsu_inside_annotation(
            row.image_path,
            row.mask_path
        ),
        axis=1
    )

    return df
