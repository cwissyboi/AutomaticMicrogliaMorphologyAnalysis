import copy
import json
import textwrap
import uuid
from pathlib import Path


SRC = Path(r"C:\Users\chris\Desktop\University\Thesis\AutomaticMicrogliaMorphologyAnalysis\initial_pipeline\analysis\yolo_active_learning.ipynb")
DST = Path(r"C:\Users\chris\Desktop\University\Thesis\AutomaticMicrogliaMorphologyAnalysis\initial_pipeline\analysis\cluster_label_cell_selection.ipynb")


def mk_md(text):
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def mk_code(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": code.splitlines(keepends=True),
    }


with SRC.open("r", encoding="utf-8") as f:
    nb = json.load(f)


new_cells = []
for cell in nb["cells"]:
    cell = copy.deepcopy(cell)
    if cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    if cell.get("cell_type") == "markdown" and any(
        "## Active Learning" in line for line in cell.get("source", [])
    ):
        break
    new_cells.append(cell)


config_code = textwrap.dedent(
    """
    from pathlib import Path

    # =====================================================================
    # CLUSTER LABELLING - configuration
    # =====================================================================

    CLUSTER_SELECTION_OUTPUT_CSV = (
        r"C:\Users\chris\Desktop\University\Thesis"
        r"\AutomaticMicrogliaMorphologyAnalysis"
        r"\initial_pipeline\morphology\morphology_outputs"
        r"\cluster_label_cell_selection_top500.csv"
    )

    YOLO_CLUSTER_DIR = Path(
        r"C:\Users\chris\Desktop\University\Thesis"
        r"\AutomaticMicrogliaMorphologyAnalysis"
        r"\initial_pipeline\morphology\morphology_outputs"
        r"\yolo_cluster_label_dataset"
    )

    N_CLUSTER_SELECTION = 500
    TILE_SIZE = 512
    OTHER_CLASS_ID = 0
    TARGET_CLASS_ID = 1
    """
).lstrip("\n")


selection_code = textwrap.dedent(
    """
    import os

    # Greedy farthest-point sampling in the 2-D t-SNE embedding.
    # Seed with the cell farthest from the embedding centroid, then repeatedly
    # add the cell whose nearest selected neighbour is farthest away.

    X_diverse = X_tsne_fast
    print(f"Selection pool : {len(X_diverse):,} cells")
    print(f"Selecting top-{N_CLUSTER_SELECTION} morphologically diverse cells...")

    centroid = X_diverse.mean(axis=0, keepdims=True)
    centroid_dists = np.linalg.norm(X_diverse - centroid, axis=1)
    first_idx = int(np.argmax(centroid_dists))

    selected_indices = [first_idx]
    selection_scores = [float(centroid_dists[first_idx])]

    D = np.linalg.norm(X_diverse - X_diverse[first_idx], axis=1).astype(np.float32)
    D[first_idx] = -np.inf

    print(
        f"Seed cell chosen at index {first_idx} "
        f"(distance to centroid = {selection_scores[0]:.4f})"
    )

    for step in range(1, N_CLUSTER_SELECTION):
        best = int(np.argmax(D))
        best_dist = float(D[best])

        if not np.isfinite(best_dist):
            print(f"Stopping early at step {step}: no finite candidates left.")
            break

        selected_indices.append(best)
        selection_scores.append(best_dist)

        d_new = np.linalg.norm(X_diverse - X_diverse[best], axis=1)
        D = np.minimum(D, d_new)
        D[best] = -np.inf

        if (step + 1) % 100 == 0 or step == 1:
            print(
                f"  step {step + 1:4d} / {N_CLUSTER_SELECTION}  "
                f"min_dist_to_selected={best_dist:.4f}"
            )

    print(f"Diverse selection done. {len(selected_indices)} cells chosen.")

    selected_np = np.array(selected_indices, dtype=int)
    selected_full_positions = np.array(tsne_fast_idx)[selected_np]
    df_index_array = np.array(X.index)
    selected_df_indices = df_index_array[selected_full_positions]

    df_selected = df.loc[selected_df_indices, [
        "global_cell_id", "scan_name", "image_name",
        "xmin", "ymin", "xmax", "ymax",
    ] + features].copy()

    df_selected.insert(0, "rank", range(1, len(df_selected) + 1))
    df_selected["diversity_score"] = selection_scores
    df_selected["tsne_x"] = X_diverse[selected_np, 0]
    df_selected["tsne_y"] = X_diverse[selected_np, 1]
    df_selected["x_row_position"] = selected_full_positions
    df_selected["is_annotation_target"] = True

    os.makedirs(os.path.dirname(CLUSTER_SELECTION_OUTPUT_CSV), exist_ok=True)
    df_selected.to_csv(CLUSTER_SELECTION_OUTPUT_CSV, index=False)

    print(f"Saved diverse-cell selection -> {CLUSTER_SELECTION_OUTPUT_CSV}")
    print(f"Images containing selected cells: {df_selected['image_name'].nunique():,}")
    display(df_selected.head())
    """
).lstrip("\n")


overlay_plot_code = textwrap.dedent(
    """
    sel_coords = X_diverse[selected_np]

    fig, ax = plt.subplots(figsize=(9, 7))

    sc = ax.scatter(
        X_tsne_fast[:, 0], X_tsne_fast[:, 1],
        c=tsne_fast_cluster_labels, cmap="tab10",
        s=2, alpha=0.35, zorder=1,
    )
    plt.colorbar(sc, ax=ax, label="Cluster")

    ax.scatter(
        sel_coords[:, 0], sel_coords[:, 1],
        marker="*", color="black", s=20, zorder=2,
        label=f"Selected ({len(sel_coords):,})",
    )

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("All cells + diverse cluster-labelling selections in t-SNE space")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()
    """
).lstrip("\n")


rank_plot_code = textwrap.dedent(
    """
    ranks = np.arange(len(selected_np))

    fig, ax = plt.subplots(figsize=(9, 7))

    ax.scatter(
        X_tsne_fast[:, 0], X_tsne_fast[:, 1],
        c="lightgrey", s=2, alpha=0.3, zorder=1, rasterized=True
    )

    sc2 = ax.scatter(
        sel_coords[:, 0], sel_coords[:, 1],
        c=ranks, cmap="plasma_r", s=20, zorder=2, alpha=0.95
    )
    plt.colorbar(sc2, ax=ax, label="Selection rank (0 = most diverse seed)")

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(
        f"Diverse cell selection: top-{len(selected_np)} cells "
        f"(plasma=rank order)"
    )
    plt.tight_layout()
    plt.show()

    print(f"Rank 1: global_cell_id={df_selected.iloc[0]['global_cell_id']}")
    print(f"Rank {len(df_selected)}: global_cell_id={df_selected.iloc[-1]['global_cell_id']}")
    """
).lstrip("\n")


export_code = textwrap.dedent(
    """
    import shutil

    selected_global_ids = set(df_selected["global_cell_id"])
    selected_image_names = set(df_selected["image_name"].unique())

    print(f"Images with >=1 selected cell: {len(selected_image_names):,}")

    df_export = df[df["image_name"].isin(selected_image_names)].copy()
    df_export["class_id"] = np.where(
        df_export["global_cell_id"].isin(selected_global_ids),
        TARGET_CLASS_ID,
        OTHER_CLASS_ID,
    )

    print(
        f"Total annotations to export: {len(df_export):,}  "
        f"across {df_export['image_name'].nunique():,} images"
    )
    print(
        f"  target cells   : {(df_export['class_id'] == TARGET_CLASS_ID).sum():,}\n"
        f"  other microglia: {(df_export['class_id'] == OTHER_CLASS_ID).sum():,}"
    )

    df_export["x_center"] = ((df_export["xmin"] + df_export["xmax"]) / 2.0) / TILE_SIZE
    df_export["y_center"] = ((df_export["ymin"] + df_export["ymax"]) / 2.0) / TILE_SIZE
    df_export["box_w"] = (df_export["xmax"] - df_export["xmin"]) / TILE_SIZE
    df_export["box_h"] = (df_export["ymax"] - df_export["ymin"]) / TILE_SIZE

    img_out_dir = YOLO_CLUSTER_DIR / "images" / "train"
    lbl_out_dir = YOLO_CLUSTER_DIR / "labels" / "train"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    written_imgs = 0
    skipped_imgs = 0

    for image_name, grp in df_export.groupby("image_name"):
        scan_name = grp["scan_name"].iloc[0]
        src_img = Path(scan_data_root) / scan_name / (image_name + ".jpg")

        if not src_img.exists():
            print(f"  [SKIP] image not found: {src_img}")
            skipped_imgs += 1
            continue

        dst_img = img_out_dir / (image_name + ".jpg")
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)
        written_imgs += 1

        lbl_path = lbl_out_dir / (image_name + ".txt")
        with open(lbl_path, "w", encoding="utf-8") as f:
            for _, row in grp.iterrows():
                f.write(
                    f"{int(row['class_id'])} "
                    f"{row['x_center']:.6f} "
                    f"{row['y_center']:.6f} "
                    f"{row['box_w']:.6f} "
                    f"{row['box_h']:.6f}\n"
                )

    print(f"\nDone - {written_imgs} images copied, {skipped_imgs} skipped.")

    yaml_path = YOLO_CLUSTER_DIR / "data.yaml"
    yaml_content = (
        f"path: {YOLO_CLUSTER_DIR.resolve()}\n"
        f"train: images/train\n"
        f"val: images/train\n"
        f"\n"
        f"names:\n"
        f"  0: other_microglia\n"
        f"  1: microglia_to_annotate_cluster\n"
    )
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"Wrote data.yaml  -> {yaml_path}")
    print(f"Dataset ready at : {YOLO_CLUSTER_DIR.resolve()}")
    print(f"  images/train/  : {len(list(img_out_dir.glob('*.jpg'))):,} files")
    print(f"  labels/train/  : {len(list(lbl_out_dir.glob('*.txt'))):,} files")
    """
).lstrip("\n")


verify_code = textwrap.dedent(
    """
    import cv2
    from matplotlib.patches import Patch

    check_image_name = df_selected.iloc[0]["image_name"]
    print(f"Inspecting: {check_image_name}")

    df_orig = df_export[df_export["image_name"] == check_image_name].copy()
    lbl_path = YOLO_CLUSTER_DIR / "labels" / "train" / (check_image_name + ".txt")
    yolo_boxes = []
    with open(lbl_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            cls = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            x1 = (cx - w / 2) * TILE_SIZE
            y1 = (cy - h / 2) * TILE_SIZE
            x2 = (cx + w / 2) * TILE_SIZE
            y2 = (cy + h / 2) * TILE_SIZE
            yolo_boxes.append((cls, x1, y1, x2, y2))

    print(f"  df boxes: {len(df_orig)}")
    print(f"  YOLO txt boxes: {len(yolo_boxes)}")
    print(f"  target boxes in image: {(df_orig['class_id'] == TARGET_CLASS_ID).sum()}")

    scan_name = df_orig["scan_name"].iloc[0]
    img_path = Path(scan_data_root) / scan_name / (check_image_name + ".jpg")
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    ax = axes[0]
    ax.imshow(img_rgb)
    for _, row in df_orig.iterrows():
        color = "yellow" if row["class_id"] == TARGET_CLASS_ID else "lime"
        ax.add_patch(
            plt.Rectangle(
                (row["xmin"], row["ymin"]),
                row["xmax"] - row["xmin"],
                row["ymax"] - row["ymin"],
                fill=False,
                edgecolor=color,
                linewidth=1.5,
            )
        )
    ax.set_title(f"Original df - {len(df_orig)} boxes")
    ax.axis("off")
    ax.legend(
        handles=[
            Patch(facecolor="none", edgecolor="yellow", label="selected target"),
            Patch(facecolor="none", edgecolor="lime", label="other microglia"),
        ],
        loc="upper right",
    )

    ax = axes[1]
    ax.imshow(img_rgb)
    for cls, x1, y1, x2, y2 in yolo_boxes:
        color = "red" if cls == TARGET_CLASS_ID else "cyan"
        ax.add_patch(
            plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor=color,
                linewidth=1.5,
            )
        )
    ax.set_title(f"YOLO export - {len(yolo_boxes)} boxes")
    ax.axis("off")
    ax.legend(
        handles=[
            Patch(facecolor="none", edgecolor="red", label="microglia_to_annotate_cluster"),
            Patch(facecolor="none", edgecolor="cyan", label="other_microglia"),
        ],
        loc="upper right",
    )

    fig.suptitle(f"Annotation comparison: {check_image_name}", fontsize=13)
    plt.tight_layout()
    plt.show()
    """
).lstrip("\n")


new_cells.extend(
    [
        mk_md("## Cluster Labelling - Diverse Cell Selection\n"),
        mk_code(config_code),
        mk_code(selection_code),
        mk_code(overlay_plot_code),
        mk_code(rank_plot_code),
        mk_md(
            "## Export Ultralytics YOLO Dataset for Cluster Labelling\n\n"
            "Images that contain at least 1 selected cell are included.\n"
            "Every cell in those images is exported.\n"
            "The selected cell(s) are written as `microglia_to_annotate_cluster`\n"
            "and all remaining cells are written as `other_microglia`.\n"
        ),
        mk_code(export_code),
        mk_code(verify_code),
    ]
)


nb["cells"] = new_cells
for cell in nb["cells"]:
    if cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []


with DST.open("w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
    f.write("\n")


print(f"Wrote {DST}")
