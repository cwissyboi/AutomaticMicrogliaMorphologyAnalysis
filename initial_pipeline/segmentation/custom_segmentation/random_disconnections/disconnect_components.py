
import numpy as np
import cv2




def sample_random_background_patch(
    image,
    mask01,
    patch_h,
    patch_w,
    max_tries=100,
    sample_full_box=False,
):
    """
    Sample a random patch for background filling.

    Parameters
    ----------
    image : np.ndarray (H, W, 3)
    mask01 : np.ndarray (H, W), binary (1 = foreground)
    patch_h, patch_w : int
        Size of patch to sample
    max_tries : int
        Number of attempts before giving up
    sample_full_box : bool
        If False (default): require patch to be pure background
        If True: allow any patch and overwrite entire box

    Returns
    -------
    patch : np.ndarray (patch_h, patch_w, 3)
    """

    H, W = mask01.shape

    # --- strict mode: background-only ---
    if not sample_full_box:
        for _ in range(max_tries):
            y0 = np.random.randint(0, H - patch_h + 1)
            x0 = np.random.randint(0, W - patch_w + 1)

            patch_mask = mask01[y0:y0 + patch_h, x0:x0 + patch_w]

            if patch_mask.sum() == 0:
                return image[y0:y0 + patch_h, x0:x0 + patch_w].copy()

    # --- relaxed mode OR fallback ---
    y0 = np.random.randint(0, H - patch_h + 1)
    x0 = np.random.randint(0, W - patch_w + 1)

    return image[y0:y0 + patch_h, x0:x0 + patch_w].copy()



def feather_blend_box(
    image,
    image_modified,
    x0, y0, x1, y1,
    feather_radius=10
):
    """
    Feather-blend a rectangular region so its edges are not noticeable.

    Parameters
    ----------
    image : np.ndarray (H, W, 3)
        Original image
    image_modified : np.ndarray (H, W, 3)
        Image after box modification (background fill, white box, etc.)
    x0, y0, x1, y1 : int
        Box coordinates
    feather_radius : int
        Width of edge blending (pixels)
    """

    H, W = image.shape[:2]

    # 1. Create box mask
    mask = np.zeros((H, W), dtype=np.float32)
    mask[y0:y1, x0:x1] = 1.0

    # 2. Blur mask → soft edges
    k = max(3, feather_radius * 2 + 1)
    mask = cv2.GaussianBlur(mask, (k, k), 0)

    # Normalize to [0,1]
    mask = mask / (mask.max() + 1e-6)

    mask = mask[..., None]  # (H,W,1)

    # 3. Alpha blend
    blended = (
        mask * image_modified.astype(np.float32) +
        (1 - mask) * image.astype(np.float32)
    )

    return blended.astype(image.dtype)


def disconnect_branches_with_gap(
    image_path,
    mask_path,
    points_xy,
    box_size=15,
    blur_output=True, 
    replace_full_box = True
):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask  = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(image_path)
    if mask is None:
        raise FileNotFoundError(mask_path)

    mask01 = (mask > 0).astype(np.uint8)

    image_out = image.copy()
    mask_out  = mask01.copy()

    half = box_size // 2
    H, W = mask01.shape

    for (x, y) in points_xy:
        x, y = int(x), int(y)

        x0 = max(0, x - half)
        x1 = min(W, x + half + 1)
        y0 = max(0, y - half)
        y1 = min(H, y + half + 1)

        ph = y1 - y0
        pw = x1 - x0

        # sample background patch
        bg_patch = sample_random_background_patch(
            image,
            mask01,
            ph,
            pw
        )

        if replace_full_box:
            image_out[y0:y1, x0:x1] = bg_patch
            mask_out[y0:y1, x0:x1] = 0

        # otherwise only replace the forground region
        else:
            fg_region = mask01[y0:y1, x0:x1] == 1
            image_out[y0:y1, x0:x1][fg_region] = bg_patch[fg_region]
            mask_out[y0:y1, x0:x1] = 0



        # --- optional blur ---
        if blur_output:
            image_out = feather_blend_box(
                image=image,
                image_modified=image_out,
                x0=x0, y0=y0, x1=x1, y1=y1,
                feather_radius=20
            )

    return image_out, mask_out
