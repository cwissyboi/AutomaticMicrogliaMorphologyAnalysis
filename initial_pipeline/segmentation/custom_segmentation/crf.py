import numpy as np
import cv2
from skimage.graph import route_through_array
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

def compute_similarity_cost_rgb(image):
    img = image.astype(np.float32)

    dx = np.linalg.norm(
        img[:, 1:] - img[:, :-1],
        axis=-1
    )
    dx = np.pad(dx, ((0, 0), (1, 0)))

    dy = np.linalg.norm(
        img[1:, :] - img[:-1, :],
        axis=-1
    )
    dy = np.pad(dy, ((1, 0), (0, 0)))

    cost = dx + dy
    cost = cost / (cost.max() + 1e-6)
    cost += 1e-3

    return cost


def compute_similarity_cost_gray(gray, white_penalty=10.0):
    gray = gray.astype(np.float32)

    dx = np.abs(gray[:, 1:] - gray[:, :-1])
    dx = np.pad(dx, ((0, 0), (1, 0)))

    dy = np.abs(gray[1:, :] - gray[:-1, :])
    dy = np.pad(dy, ((1, 0), (0, 0)))

    similarity = dx + dy
    similarity = similarity / (similarity.max() + 1e-6)

    # Penalize bright pixels (white background)
    brightness = gray / 255.0
    brightness_penalty = 1.0 + white_penalty * (brightness ** 2)

    cost = similarity * brightness_penalty
    cost += 1e-3

    return cost


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
    connected_mask = binary_mask.copy()

    H, W = binary_mask.shape
    image_area = H * W
    min_area_px = min_component_frac * image_area

    num_labels, labels = cv2.connectedComponents(binary_mask)
    if num_labels <= 2:
        return connected_mask

    # --- Cost computation ---
    if image.ndim == 3:
        cost = compute_similarity_cost_rgb(image)
    elif image.ndim == 2:
        cost = compute_similarity_cost_gray(image)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    if restrict_to_mask:
        penalty = cost.max() * 10
        cost[~binary_mask.astype(bool)] += penalty

    # --- Thickness estimation ---
    thickness_map = distance_transform_edt(binary_mask) * thickness_scale

    # --- Collect large components ---
    components = []
    for label in range(1, num_labels):
        ys, xs = np.where(labels == label)
        area = len(xs)

        if area < min_area_px:
            continue

        cy = int(np.mean(ys))
        cx = int(np.mean(xs))
        components.append((cy, cx))

    # Nothing to connect
    if len(components) < 2:
        # Still enforce single component
        return keep_largest_component(connected_mask)

    # Sort components spatially
    components = sorted(components, key=lambda p: (p[1], p[0]))

    # --- Connect components ---
    for i in range(len(components) - 1):
        start = components[i]
        end = components[i + 1]

        path, _ = route_through_array(
            cost,
            start,
            end,
            fully_connected=True
        )

        connected_mask = draw_thick_path(
            connected_mask,
            path,
            thickness_map,
            min_radius=1,
            max_radius=max_radius,
        )

    # --- FINAL CLEANUP: keep only one connected component ---
    connected_mask = keep_largest_component(connected_mask)

    return connected_mask.astype(np.uint8)


def compute_mask_otsu_inside_annotation(
    image_path,
    mask_path,
    connect_components=True,
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

    if connect_components:
        mask = connect_components_geodesic_similarity(mask, gray)

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
