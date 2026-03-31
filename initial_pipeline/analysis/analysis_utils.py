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
import glob
import os
import time
from openTSNE import TSNE as openTSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


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

def get_morphology_results_from_folder(folder_path = r"C:\Users\chris\Desktop\University\Thesis\PipelineResults\Many_Patients", remove_outlier_sizes = False, remove_edges = False): 
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    dfs = []
    for path in csv_files:
        temp_df = pd.read_csv(path)
        temp_df = temp_df.rename(columns={"scan_folder": "scan_name"})
        dfs.append(temp_df)

        df = pd.concat(dfs, ignore_index=True)

    features = [
            "skeleton_length",
            "num_junctions",
            "num_components",
            "num_end_nodes",
            "num_start_nodes",
            "total_nodes",
            "end_to_start_ratio",
            "soma_area",
            "soma_perimeter",
            "soma_circularity",
            "cell_area",
            "cell_perimeter",
            "cell_convex_hull_area",
            "cell_convex_hull_perimeter",
            "cell_solidity",
            "cell_convexity",
            "cell_circularity",
            "cell_convex_circularity",
            "branch_area",
            "branch_perimeter",
            "sholl_min_radius",
            "sholl_peak_radius",
            "sholl_max_radius",
            "sholl_peak",
            "sholl_sum",
    ]

    print(f"Number of cells in analysis {len(df)}")

    if (remove_outlier_sizes):
        lower = df["cell_area"].quantile(0.002)
        upper = df["cell_area"].quantile(0.998)

        df = df[
            (df["cell_area"] >= lower) &
            (df["cell_area"] <= upper)
        ].copy()

        print(f"Removed very small or very large cells")

        print(f"Number of cells in analysis {len(df)}")
    
    if (remove_edges):
        # Tiles are 512x512 pixels.
        # A cell whose bounding box touches any edge of the tile was clipped and
        # therefore not fully visible. We remove any cell where:
        #   xmin == 0  (touches left edge)
        #   ymin == 0  (touches top edge)
        #   xmax >= TILE_SIZE - 1  (touches right edge)
        #   ymax >= TILE_SIZE - 1  (touches bottom edge)

        TILE_SIZE = 512

        before = len(df)

        df = df[
            (df["xmin"] > 0 + 1) &
            (df["ymin"] > 0 + 1) &
            (df["xmax"] < TILE_SIZE - 1) &
            (df["ymax"] < TILE_SIZE - 1)
        ].copy()

        print(f"Removed {before - len(df)} border-clipped cells ({before} -> {len(df)})")

    X = df[features].dropna()
    X_scaled = StandardScaler().fit_transform(X)

    return df, X_scaled


def get_tsne_subsampled_and_projected(X_scaled, tsne_subsample = 30000, n_components = 4):
    tsne_subsample = 30000  # fit t-SNE on this many cells, then transform the rest
  
    n_cells = len(X_scaled)
    if tsne_subsample is not None and tsne_subsample < n_cells:
        rng = np.random.default_rng(42)
        tsne_fast_idx = np.sort(rng.choice(n_cells, size=tsne_subsample, replace=False))
    else:
        tsne_fast_idx = np.arange(n_cells)

    X_tsne_fast_input = X_scaled[tsne_fast_idx]
    tsne_fast_rest_idx = np.setdiff1d(np.arange(n_cells), tsne_fast_idx, assume_unique=True)

    t0 = time.perf_counter()

    tsne_fast = openTSNE(
        n_components=n_components,
        perplexity=30,
        # negative_gradient_method='fft',  # FFT-accelerated, much faster than exact/bh
        negative_gradient_method='bh',  # bh allows you to do dim reduction to 4 rather than 2
        n_jobs=6,                        # use all CPU cores
        random_state=42,
        # n_iter = 10
    )
    tsne_fast_embedding = tsne_fast.fit(X_tsne_fast_input)
    fast_time = time.perf_counter() - t0

    print(f'openTSNE runtime (fit {len(tsne_fast_idx)} data points) in {fast_time:.2f}s')
    X_tsne_fast = np.empty((n_cells, n_components), dtype=np.float32)
    X_tsne_fast[tsne_fast_idx] = np.asarray(tsne_fast_embedding)

    if len(tsne_fast_rest_idx) > 0:
        X_tsne_fast[tsne_fast_rest_idx] = np.asarray(tsne_fast_embedding.transform(X_scaled[tsne_fast_rest_idx], n_iter = 5, learning_rate = 0.1, perplexity = 30 ))
        # X_tsne_fast[tsne_fast_rest_idx] = np.asarray(tsne_fast_embedding.transform(X_scaled[tsne_fast_rest_idx]))

    transform_time = time.perf_counter() - fast_time
    print(f'openTSNE transform time (transformed {len(tsne_fast_rest_idx)} datapoints): {transform_time:.2f}s')
    return X_tsne_fast



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def plot_cluster_distribution_per_scan(
    final_props_df,
    cluster_type="hard",   # "hard" or "soft"
    title_prefix="Per-scan cluster proportions",
):
    if cluster_type not in {"hard", "soft"}:
        raise ValueError("cluster_type must be 'hard' or 'soft'")

    required_cols = {"scan_name", "diagnosis_group"}
    missing_required = required_cols - set(final_props_df.columns)
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    cluster_prefix = f"{cluster_type}_cluster_"
    cluster_cols = [c for c in final_props_df.columns if c.startswith(cluster_prefix)]
    if not cluster_cols:
        raise ValueError(f"No columns found matching prefix '{cluster_prefix}'")

    # sort clusters numerically: *_0, *_1, ...
    cluster_cols = sorted(cluster_cols, key=lambda x: int(x.split("_")[-1]))
    n_clusters_local = len(cluster_cols)

    cmap = cm.viridis
    norm = mcolors.Normalize(vmin=0, vmax=max(1, n_clusters_local - 1))
    cluster_colours = {c: cmap(norm(i)) for i, c in enumerate(cluster_cols)}

    for diagnosis in ["AD", "100+"]:
        sub = final_props_df.loc[final_props_df["diagnosis_group"] == diagnosis].copy()
        if sub.empty:
            print(f"No rows found for diagnosis_group='{diagnosis}'")
            continue

        # optional: stable order
        sub = sub.sort_values("scan_name")

        scans = sub["scan_name"].tolist()
        n_scans = len(scans)
        x = np.arange(n_scans)
        bar_width = 0.8 / n_clusters_local

        fig, ax = plt.subplots(figsize=(max(12, n_scans * 0.5), 6))

        for i, col in enumerate(cluster_cols):
            offsets = x + (i - n_clusters_local / 2 + 0.5) * bar_width
            values = sub[col].values
            cluster_id = col.replace(cluster_prefix, "")

            ax.bar(
                offsets,
                values,
                width=bar_width,
                color=cluster_colours[col],
                label=f"{cluster_type.capitalize()} {cluster_id}",
                edgecolor="white",
                linewidth=0.4,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(scans, rotation=60, ha="right", fontsize=8)
        ax.set_ylabel("Proportion of cells")
        ax.set_ylim(0, 1)
        ax.set_title(f"{title_prefix} ({cluster_type.capitalize()}) - {diagnosis}")
        ax.legend(title="Cluster", bbox_to_anchor=(1.01, 1), loc="upper left")

        plt.tight_layout()
        plt.show()