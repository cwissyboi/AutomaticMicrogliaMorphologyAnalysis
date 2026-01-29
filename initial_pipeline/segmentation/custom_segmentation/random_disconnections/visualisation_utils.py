import cv2
import matplotlib.pyplot as plt
from .find_connection_points import find_random_branch_points
from .disconnect_components import disconnect_branches_with_gap




def draw_crosses(img_rgb, points_xy, size=8, thickness=2):
    """Return a copy with cross markers drawn."""
    out = img_rgb.copy()
    for (x, y) in points_xy:
        # draw an "X"
        cv2.line(out, (x - size, y - size), (x + size, y + size), (0, 255, 0), thickness)
        cv2.line(out, (x - size, y + size), (x + size, y - size), (0, 255, 0), thickness)
    return out



def visualize_first_n(train_df, n=5):
    for i, row in train_df.head(n).iterrows():
        image_path = row["image_path"]
        mask_path  = row["mask_path"]

        # points = find_branch_soma_connection_points(
        #     image_path,
        #     mask_path,
        #     soma_dist_frac=0.35,
        #     candidate_cluster_radius=7,
        #     debug=False
        # )

        points = find_random_branch_points(
        image_path,
        mask_path,
        points_per_branch=5
        )

        img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        vis = draw_crosses(img_rgb, points, size=1, thickness=1)

        plt.figure(figsize=(6, 6))
        plt.imshow(vis)
        plt.axis("off")
        plt.title(
            f"{row['scan']} | {row['class']} | "
            f"{len(points)} connections"
        )
        plt.show()


def visualize_first_n_with_branch_cuts(
    train_df,
    n=5,
    box_size = 15,
    show_points=True, 
    blur_output = True, 
    replace_full_box = True,
):
    """
    Visualize branch disconnections for the first n rows of train_df.

    Parameters
    ----------
    train_df : pd.DataFrame
        Must contain image_path and mask_path
    n : int
        Number of rows to visualize
    gap_length : int
        Length of the cut gap (pixels)
    gap_width : int
        Width of the cut gap (pixels)
    show_points : bool
        Whether to overlay branch–soma connection points
    """

    for i, row in train_df.head(n).iterrows():
        image_path = row["image_path"]
        mask_path  = row["mask_path"]

        # --- find branch connection points ---
        # points = find_branch_soma_connection_points(
        #     image_path,
        #     mask_path,
        #     soma_dist_frac=0.35,
        #     candidate_cluster_radius=7,
        #     debug=False
        # )

        points = find_random_branch_points(
            image_path=image_path, 
            mask_path = mask_path, 
            points_per_branch=5, 
            min_branch_length=10
        )

        # --- cut branches locally ---
        image_cut, _ = disconnect_branches_with_gap(
            image_path,
            mask_path,
            points, 
            box_size = box_size,
            blur_output = blur_output
        )

        print(type(image_cut))

        # --- convert to RGB for plotting ---
        image_cut_rgb = cv2.cvtColor(image_cut, cv2.COLOR_BGR2RGB)

        # --- optionally overlay connection points ---
        if show_points:
            image_cut_rgb = draw_crosses(
                image_cut_rgb,
                points,
                size=2,
                thickness=1
            )

        # --- show ---
        plt.figure(figsize=(6, 6))
        plt.imshow(image_cut_rgb)
        plt.axis("off")
        plt.title(
            f"{row['scan']} | {row['class']} | "
            f"{len(points)} cuts"
        )
        plt.show()
