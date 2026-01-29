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



import cv2
import numpy as np
def add_floating_synthetic_fragments(
    image,
    mask01,
    num_objects=(1, 4),
    branch_length=(10, 50),
    thickness=2,
    color_jitter=3,          # keep small now
    min_dist_to_mask=40,
    max_tries=100
):
    """
    Add floating synthetic cell-like fragments to an image,
    using the mean foreground colour, and keeping distance
    from the ground-truth mask.
    """

    H, W, _ = image.shape
    image_out = image.copy()

    # --- compute mean foreground colour ONCE ---
    # base_color = compute_mean_foreground_color(image, mask01).astype(np.int32)
    base_color = (255, 0, 0)

    # distance map from ground truth mask
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

        # --- draw branch ---
        for i in range(len(branch) - 1):
            x0, y0 = branch[i]
            x1, y1 = branch[i + 1]

            if not (0 <= x0 < W and 0 <= y0 < H):
                continue
            if dist[y0, x0] < min_dist_to_mask:
                break

            # small jitter around global foreground colour
            jitter = np.random.randint(
                -color_jitter, color_jitter + 1, size=3
            )

            color = np.clip(base_color + jitter, 0, 255).astype(np.uint8)

            cv2.line(
                image_out,
                (x0, y0),
                (x1, y1),
                color=tuple(int(c) for c in color),
                thickness=thickness
            )

    return image_out
