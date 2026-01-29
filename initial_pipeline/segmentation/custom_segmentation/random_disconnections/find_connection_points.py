from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

def _ensure_binary_mask(mask: np.ndarray, thresh=127) -> np.ndarray:
    """Return uint8 binary mask in {0,1}."""
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    # If mask is already 0/1-ish
    if mask.max() <= 1:
        return (mask > 0).astype(np.uint8)
    return (mask >= thresh).astype(np.uint8)

def _clean_mask(mask01: np.ndarray, min_area=200) -> np.ndarray:
    """Remove small components and fill tiny holes a bit."""
    mask = (mask01 * 255).astype(np.uint8)

    # close small gaps
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    # keep largest component (often safest for single-cell crops)
    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if num <= 1:
        return (mask > 0).astype(np.uint8)

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + int(np.argmax(areas))
    clean = (labels == largest_idx).astype(np.uint8)

    # drop if too small
    if clean.sum() < min_area:
        return clean

    return clean

def _skeletonize(mask01: np.ndarray) -> np.ndarray:
    """
    Try skimage skeletonize; fallback to cv2.ximgproc.thinning if available.
    Returns uint8 {0,1}.
    """
    try:
        from skimage.morphology import skeletonize
        skel = skeletonize(mask01.astype(bool))
        return skel.astype(np.uint8)
    except Exception:
        pass

    # fallback: OpenCV thinning (needs opencv-contrib-python)
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        skel = cv2.ximgproc.thinning((mask01 * 255).astype(np.uint8))
        return (skel > 0).astype(np.uint8)

    raise ImportError(
        "Skeletonization requires either scikit-image (skimage) "
        "or opencv-contrib-python (cv2.ximgproc.thinning)."
    )

def _cluster_points(points_xy, cluster_radius=6):
    """
    Greedy clustering: merge points within cluster_radius (pixels).
    Returns list of integer (x,y) centroids.
    """
    if len(points_xy) == 0:
        return []

    pts = np.array(points_xy, dtype=np.float32)
    used = np.zeros(len(pts), dtype=bool)
    clusters = []

    for i in range(len(pts)):
        if used[i]:
            continue
        # start a cluster
        seed = pts[i]
        d = np.linalg.norm(pts - seed, axis=1)
        idx = np.where((d <= cluster_radius) & (~used))[0]
        used[idx] = True
        centroid = pts[idx].mean(axis=0)
        clusters.append((int(round(centroid[0])), int(round(centroid[1]))))

    return clusters

def find_branch_soma_connection_points(
    image_path,
    mask_path,
    soma_dist_frac=0.35,
    soma_erode_iters=1,
    candidate_cluster_radius=7,
    debug=False
):
    """
    Parameters
    ----------
    image_path : str|Path
    mask_path : str|Path
    soma_dist_frac : float
        Soma threshold = soma_dist_frac * max(distance_transform).
        Larger => smaller soma; smaller => bigger soma.
    soma_erode_iters : int
        Erode soma a bit so we catch touching skeleton points at boundary robustly.
    candidate_cluster_radius : int
        Cluster candidate pixels into one point per connection.
    debug : bool
        If True, returns intermediate masks too.

    Returns
    -------
    points_xy : list[(x,y)]
    (optional) debug_dict
    """
    image_path = Path(image_path)
    mask_path = Path(mask_path)

    # --- load image (BGR->RGB for plotting) ---
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # --- load mask ---
    mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask_raw is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")

    mask01 = _ensure_binary_mask(mask_raw)
    mask01 = _clean_mask(mask01)

    # --- distance transform to estimate soma (thick region) ---
    # dt expects 0/255 uint8
    dt = cv2.distanceTransform((mask01 * 255).astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=5)
    max_dt = float(dt.max()) if dt.size else 0.0
    if max_dt <= 0:
        return ([], {"reason": "empty mask"}) if debug else []

    soma = (dt >= soma_dist_frac * max_dt).astype(np.uint8)

    # keep largest soma component (in case branches contain thicker bits)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(soma, connectivity=8)
    if num > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        soma = (labels == (1 + int(np.argmax(areas)))).astype(np.uint8)

    # optional erosion so boundary-touch detection is stable
    if soma_erode_iters > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        soma = cv2.erode(soma, k, iterations=soma_erode_iters)

    # --- skeletonize whole cell mask ---
    skel = _skeletonize(mask01)

    # --- soma boundary band: dilate(soma) - soma ---
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    soma_dil = cv2.dilate(soma, k, iterations=1)
    soma_boundary_band = (soma_dil & (1 - soma)).astype(np.uint8)

    # candidate connection pixels: skeleton that lies in boundary band
    candidates = (skel & soma_boundary_band).astype(np.uint8)

    # get candidate pixel coords (x,y)
    ys, xs = np.where(candidates > 0)
    pts = list(zip(xs.tolist(), ys.tolist()))

    # cluster to one point per connection
    points_xy = _cluster_points(pts, cluster_radius=candidate_cluster_radius)

    if debug:
        dbg = {
            "mask01": mask01,
            "dt": dt,
            "soma": soma,
            "skel": skel,
            "soma_boundary_band": soma_boundary_band,
            "candidates": candidates,
        }
        return points_xy, dbg

    return points_xy



def find_random_branch_points(
    image_path,
    mask_path,
    soma_dist_frac=0.35,
    soma_erode_iters=1,
    points_per_branch=2,
    min_branch_length=50,
    debug=False
):
    """
    Find random points along each branch (excluding soma).

    Parameters
    ----------
    image_path : str | Path
        Path to image (only used for consistency / debugging)
    mask_path : str | Path
        Path to binary mask
    soma_dist_frac : float
        Fraction of max distance transform to define soma
    soma_erode_iters : int
        Erode soma to cleanly separate from branches
    points_per_branch : int
        Number of random points sampled per branch
    min_branch_length : int
        Minimum skeleton length (pixels) for a branch to be considered
    debug : bool
        If True, return intermediate masks

    Returns
    -------
    points_xy : list[(x, y)]
        Randomly sampled branch points
    (optional) debug_dict
    """

    image_path = Path(image_path)
    mask_path = Path(mask_path)

    # --- load mask ---
    mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask_raw is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")

    mask01 = _ensure_binary_mask(mask_raw)
    mask01 = _clean_mask(mask01)

    # --- distance transform to estimate soma ---
    dt = cv2.distanceTransform(
        (mask01 * 255).astype(np.uint8),
        cv2.DIST_L2,
        5
    )

    max_dt = float(dt.max())
    if max_dt <= 0:
        return ([], {"reason": "empty mask"}) if debug else []

    soma = (dt >= soma_dist_frac * max_dt).astype(np.uint8)

    # keep largest soma component
    num, labels, stats, _ = cv2.connectedComponentsWithStats(soma, connectivity=8)
    if num > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        soma = (labels == (1 + int(np.argmax(areas)))).astype(np.uint8)

    if soma_erode_iters > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        soma = cv2.erode(soma, k, iterations=soma_erode_iters)

    # --- branches = foreground minus soma ---
    branch_mask = mask01 & (~soma)

    # --- skeletonize branches ---
    branch_skel = _skeletonize(branch_mask)

    # --- connected components on skeleton ---
    num, labels = cv2.connectedComponents(branch_skel, connectivity=8)

    rng = np.random.default_rng()
    points_xy = []

    for lab in range(1, num):
        ys, xs = np.where(labels == lab)
        if len(xs) < min_branch_length:
            continue

        coords = list(zip(xs, ys))

        if len(coords) <= points_per_branch:
            chosen = coords
        else:
            idx = rng.choice(len(coords), size=points_per_branch, replace=False)
            chosen = [coords[i] for i in idx]

        points_xy.extend(chosen)

    if debug:
        dbg = {
            "mask01": mask01,
            "soma": soma,
            "branch_mask": branch_mask,
            "branch_skel": branch_skel,
            "branch_labels": labels,
        }
        return points_xy, dbg

    return points_xy
