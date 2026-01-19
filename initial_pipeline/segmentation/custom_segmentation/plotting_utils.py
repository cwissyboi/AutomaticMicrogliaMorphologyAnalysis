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
