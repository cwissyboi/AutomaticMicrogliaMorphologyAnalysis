import numpy as np

def generate_floating_branch(
    start_xy,
    length=40,
    step=2,
    angle_jitter=0.4
):
    x, y = start_xy
    angle = np.random.uniform(0, 2*np.pi)
    points = []

    for _ in range(length):
        angle += np.random.normal(0, angle_jitter)
        x += step * np.cos(angle)
        y += step * np.sin(angle)
        points.append((int(x), int(y)))

    return points

def compute_mean_foreground_color(image, mask01):
    """
    Compute mean RGB colour over the positive mask.

    Parameters
    ----------
    image : np.ndarray (H, W, 3), uint8
    mask01 : np.ndarray (H, W), binary {0,1}

    Returns
    -------
    mean_color : np.ndarray (3,), uint8
    """
    fg_pixels = image[mask01 == 1]

    if fg_pixels.size == 0:
        # fallback: neutral brown-ish
        return np.array([160, 120, 90], dtype=np.uint8)

    mean_color = fg_pixels.mean(axis=0)
    return mean_color.astype(np.uint8)


import numpy as np
import cv2

def compute_branch_color(
    image,
    mask01,
    branch_dist_frac=0.35,
    percentile=70,
    fallback_color=(165, 120, 95)
):
    """
    Estimate the RGB colour of cell branches (excluding soma).

    Parameters
    ----------
    image : np.ndarray (H, W, 3), uint8
        RGB image
    mask01 : np.ndarray (H, W), binary {0,1}
        Foreground mask
    branch_dist_frac : float
        Fraction of max distance transform used to define branches.
        Smaller = thinner structures.
    percentile : int
        Percentile of colour distribution to return (robust to noise).
    fallback_color : tuple
        Used if branch pixels cannot be reliably detected.

    Returns
    -------
    branch_color : np.ndarray (3,), uint8
    """

    # --- distance transform inside the mask ---
    dt = cv2.distanceTransform(
        (mask01 > 0).astype(np.uint8),
        cv2.DIST_L2,
        5
    )

    max_dt = dt[mask01 > 0].max() if mask01.any() else 0
    if max_dt <= 0:
        return np.array(fallback_color, dtype=np.uint8)

    # --- thin regions = branches ---
    branch_mask = (dt > 0) & (dt <= branch_dist_frac * max_dt)

    branch_pixels = image[branch_mask]

    if branch_pixels.shape[0] < 20:
        # not enough samples → fallback
        return np.array(fallback_color, dtype=np.uint8)

    # robust colour estimate
    branch_color = np.percentile(branch_pixels, percentile, axis=0)

    return branch_color.astype(np.uint8)




import cv2
import numpy as np

def add_floating_synthetic_fragments(
    image,
    mask01,
    num_objects=(1, 2),
    branch_length=(5, 30),
    thickness=2,
    color_jitter=2,
    min_dist_to_mask=40,
    max_tries=100,
    brightness_range=(0.75, 1.15), 
):
    """
    Add floating synthetic cell-like fragments to an image,
    with smooth light→dark colour variation along each branch.
    """

    H, W, _ = image.shape
    image_out = image.copy()

    # Base branch colour (RGB)
    base_color = compute_branch_color(image, mask01).astype(np.float32)

    # Distance map from GT mask
    dist = cv2.distanceTransform(
        (1 - mask01).astype(np.uint8),
        cv2.DIST_L2,
        5
    )

    n_objects = np.random.randint(*num_objects)

    for _ in range(n_objects):
        # --- sample a safe starting point ---
        for _ in range(max_tries):
            y = np.random.randint(0, H)
            x = np.random.randint(0, W)
            if dist[y, x] >= min_dist_to_mask:
                break
        else:
            continue

        length = np.random.randint(*branch_length)
        branch = generate_floating_branch((x, y), length)

        # --- sample brightness gradient for this branch ---
        start_b = np.random.uniform(*brightness_range)
        end_b   = np.random.uniform(*brightness_range)

        L = max(1, len(branch) - 1)

        for i in range(L):
            x0, y0 = branch[i]
            x1, y1 = branch[i + 1]

            if not (0 <= x0 < W and 0 <= y0 < H):
                continue
            if dist[y0, x0] < min_dist_to_mask:
                break

            # smooth brightness interpolation
            t = i / max(1, L - 1)
            brightness = (1 - t) * start_b + t * end_b

            # tiny jitter to avoid flatness
            jitter = np.random.randint(
                -color_jitter, color_jitter + 1, size=3
            )

            color = base_color * brightness + jitter
            color = np.clip(color, 0, 255).astype(np.uint8)

            cv2.line(
                image_out,
                (x0, y0),
                (x1, y1),
                color=tuple(int(c) for c in color),
                thickness=thickness
            )

    return image_out
