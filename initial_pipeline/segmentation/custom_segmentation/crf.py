import numpy as np
import cv2
from skimage.graph import route_through_array
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path


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
    restrict_to_mask=True,
    thickness_scale=1.0,
    max_radius=12,
):
    """
    Connect components by greedily attaching the closest component
    to the current main (largest) component, updating distances after
    each merge.
    """

    H, W = binary_mask.shape
    image_area = H * W
    min_area_px = min_component_frac * image_area

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


    if restrict_to_mask:
        cost = cost.copy()
        cost[~binary_mask.astype(bool)] += cost.max() * 10

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

    # --- Find largest component (main/root) ---
    areas = {label: len(pixels) for label, pixels in components.items()}
    main_label = max(areas, key=areas.get)

    main_mask = (labels == main_label).astype(np.uint8)
    connected_mask = main_mask.copy()

    remaining = set(components.keys())
    remaining.remove(main_label)

    # --- Greedy nearest-to-main loop ---
    while remaining:
        # Distance to current main component
        dist_to_main = distance_transform_edt(1 - main_mask)

        best_label = None
        best_dist = np.inf
        best_point_other = None

        # Find closest remaining component
        for label in remaining:
            pixels = components[label]
            dists = dist_to_main[pixels[:, 0], pixels[:, 1]]
            idx = np.argmin(dists)

            if dists[idx] < best_dist:
                best_dist = dists[idx]
                best_label = label
                best_point_other = tuple(pixels[idx])

        # Closest point in main component
        main_pixels = np.column_stack(np.where(main_mask))
        main_idx = np.argmin(
            np.linalg.norm(
                main_pixels - np.array(best_point_other),
                axis=1
            )
        )
        best_point_main = tuple(main_pixels[main_idx])

        # --- Geodesic path ---
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

        # --- Merge into main component ---
        main_mask = np.maximum(main_mask, connected_mask)
        main_mask[labels == best_label] = 1

        remaining.remove(best_label)

    # --- Final cleanup ---
    connected_mask = keep_largest_component(main_mask)

    return connected_mask.astype(np.uint8)


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
