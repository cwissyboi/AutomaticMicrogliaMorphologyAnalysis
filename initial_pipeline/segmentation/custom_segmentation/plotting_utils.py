import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path

def show_row_visuals(
    df,
    idx,
    columns,
    cmap_mask="gray"
):
    """
    Show original image + multiple masks/images side by side.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing image/mask columns
    idx : int
        Row index in the DataFrame
    columns : list[str]
        Columns to visualize (can be paths or numpy arrays)
        e.g. ["mask_path", "true_mask"]
    cmap_mask : str
        Colormap for mask-like visuals
    """
    row = df.iloc[idx]

    # Always load original image
    image = np.array(Image.open(row.image_path).convert("RGB"))

    n_plots = 1 + len(columns)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))

    if n_plots == 1:
        axes = [axes]

    # --- Original image ---
    axes[0].imshow(image)
    axes[0].set_title("Original image")
    axes[0].axis("off")

    # --- Additional columns ---
    for ax, col in zip(axes[1:], columns):
        value = row[col]

        # Case 1: numpy array (e.g. computed mask)
        if isinstance(value, np.ndarray):
            ax.imshow(value, cmap=cmap_mask)

        # Case 2: path to image/mask
        elif isinstance(value, (str, Path)):
            arr = np.array(Image.open(value))
            if arr.ndim == 2:
                ax.imshow(arr, cmap=cmap_mask)
            else:
                ax.imshow(arr)

        else:
            raise TypeError(f"Unsupported type for column '{col}': {type(value)}")

        ax.set_title(col)
        ax.axis("off")

    plt.suptitle(
        f"Scan: {row.scan} | Class: {row['class']}",
        fontsize=12
    )
    plt.tight_layout()
    plt.show()


def show_mask_outline(
    df,
    idx,
    mask_col,
    outline_color="red",
    linewidth=2
):
    """
    Show original image with a mask overlaid as an outline.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing image and mask columns
    idx : int
        Row index in the DataFrame
    mask_col : str
        Column name of the mask (numpy array or path)
    outline_color : str
        Color of the mask outline
    linewidth : float
        Width of the outline
    """
    row = df.iloc[idx]

    # --- Load original image ---
    image = np.array(Image.open(row.image_path).convert("RGB"))

    # --- Load mask ---
    mask_value = row[mask_col]

    if isinstance(mask_value, np.ndarray):
        mask = mask_value
    elif isinstance(mask_value, (str, Path)):
        mask = np.array(Image.open(mask_value))
    else:
        raise TypeError(
            f"Unsupported type for column '{mask_col}': {type(mask_value)}"
        )

    # Ensure binary mask
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = mask > 0

    # --- Plot ---
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(image)

    # Draw mask outline
    ax.contour(
        mask.astype(np.uint8),
        levels=[0.5],
        colors=outline_color,
        linewidths=linewidth
    )

    ax.set_title(f"Mask outline: {mask_col}")
    ax.axis("off")

    plt.suptitle(
        f"Scan: {row.scan} | Class: {row['class']}",
        fontsize=12
    )
    plt.tight_layout()
    plt.show()
