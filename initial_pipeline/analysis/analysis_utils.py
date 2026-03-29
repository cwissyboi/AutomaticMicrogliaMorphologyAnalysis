from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import entropy as scipy_entropy
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


def continuity(X_high: np.ndarray, X_low: np.ndarray, n_neighbors: int = 10) -> float:
    n = X_high.shape[0]

    nbrs_high = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X_high)
    _, ind_high = nbrs_high.kneighbors(X_high)
    ind_high = ind_high[:, 1:]

    nbrs_low = NearestNeighbors(n_neighbors=n).fit(X_low)
    _, ind_low_full = nbrs_low.kneighbors(X_low)
    rank_low = np.empty((n, n), dtype=int)
    for i in range(n):
        rank_low[i, ind_low_full[i]] = np.arange(n)

    penalty = 0
    for i in range(n):
        for j in ind_high[i]:
            r = rank_low[i, j]
            if r > n_neighbors:
                penalty += r - n_neighbors

    normalizer = n_neighbors * n * (2 * n - 3 * n_neighbors - 1) / 2
    return 1 - penalty / normalizer


def show_yolo_box(
    image_path: str | Path,
    box: tuple[int, int, int, int],
    color: tuple[int, int, int] = (0, 0, 255),
    linewidth: int = 2,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    x_min, y_min, x_max, y_max = map(int, box)
    cv2.rectangle(image_rgb, (x_min, y_min), (x_max, y_max), color, linewidth)

    plt.figure(figsize=(6, 6))
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.title("YOLO bounding box")
    plt.show()


def show_cluster_representatives(
    df_source: pd.DataFrame,
    cluster_col: str,
    X_feature: np.ndarray,
    title_prefix: str,
    scan_data_root: str | Path,
    n_representatives: int = 4,
    idx_array: np.ndarray | None = None,
    cluster_labels_array: np.ndarray | None = None,
) -> None:
    scan_data_root = Path(scan_data_root)

    if idx_array is not None:
        sampled_df_index = df_source.index.to_numpy()[idx_array]
        working_df = df_source.loc[sampled_df_index].copy()
    else:
        working_df = df_source.copy()

    if cluster_labels_array is not None:
        if len(cluster_labels_array) != len(working_df):
            raise ValueError("cluster_labels_array length must match working dataframe length.")
        working_df[cluster_col] = cluster_labels_array

    for cluster_id in sorted(working_df[cluster_col].unique()):
        cluster_mask = working_df[cluster_col] == cluster_id
        cluster_df = working_df[cluster_mask]
        X_cluster = X_feature[cluster_mask.values]

        centroid = X_cluster.mean(axis=0, keepdims=True)
        distances = cdist(X_cluster, centroid, metric="euclidean").flatten()
        closest_idxs = np.argsort(distances)[: min(n_representatives, len(distances))]

        reps = []
        for idx in closest_idxs:
            row = cluster_df.iloc[idx]
            reps.append(
                {
                    "global_cell_id": row.get("global_cell_id", row.name),
                    "distance_to_centroid": float(distances[idx]),
                    "scan_name": row["scan_name"],
                    "image_name": row["image_name"],
                    "xmin": row["xmin"],
                    "ymin": row["ymin"],
                    "xmax": row["xmax"],
                    "ymax": row["ymax"],
                }
            )

        n = len(reps)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
        if n == 1:
            axes = [axes]

        fig.suptitle(f"{title_prefix} {cluster_id} - {n} most central cells", fontsize=14)

        for ax, rep in zip(axes, reps):
            image_path = scan_data_root / rep["scan_name"] / f"{rep['image_name']}.jpg"
            image = cv2.imread(str(image_path))
            if image is None:
                ax.set_title(f"NOT FOUND\n{rep['image_name']}")
                ax.axis("off")
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            x_min = int(rep["xmin"])
            y_min = int(rep["ymin"])
            x_max = int(rep["xmax"])
            y_max = int(rep["ymax"])
            cv2.rectangle(image_rgb, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

            ax.imshow(image_rgb)
            ax.axis("off")
            ax.set_title(
                f"{rep['global_cell_id']}\nd={rep['distance_to_centroid']:.3f}",
                fontsize=7,
            )

        plt.tight_layout()
        plt.show()


def soft_cluster_metrics(
    probs: np.ndarray,
    embedding: np.ndarray,
    name: str,
    threshold: float = 0.8,
    sample_size: int | None = None,
    random_state: int | None = None,
) -> dict[str, float | int | str]:
    n_samples, k = probs.shape

    if embedding.shape[0] != n_samples:
        raise ValueError("probs and embedding must have the same number of samples.")

    if sample_size is None or sample_size >= n_samples:
        probs_eval = probs
        embedding_eval = embedding
    else:
        if not isinstance(sample_size, int) or sample_size <= 0:
            raise ValueError("sample_size must be a positive integer or None.")
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n_samples, size=sample_size, replace=False)
        probs_eval = probs[idx]
        embedding_eval = embedding[idx]

    hard_labels = probs_eval.argmax(axis=1)
    sil = (
        silhouette_score(embedding_eval, hard_labels)
        if len(set(hard_labels)) > 1
        else float("nan")
    )

    per_sample_entropy = scipy_entropy(probs_eval.T, base=2)
    mean_ent = float(per_sample_entropy.mean())
    max_possible_ent = float(np.log2(k))

    max_membership = probs_eval.max(axis=1)
    pct_confident = float((max_membership > threshold).mean() * 100)

    return {
        "Method": name,
        "k": k,
        "Silhouette (argmax)": round(sil, 4),
        "Mean entropy (bits)": round(mean_ent, 4),
        f"Max-entropy for k={k} (bits)": round(max_possible_ent, 4),
        f"% cells > {threshold} confidence": round(pct_confident, 1),
    }
