import torch
import cv2
import numpy as np
from PIL import Image
import os
from helpers import get_file_name, output_masks_to_file_overlay


@torch.no_grad()
def unet_inference(
    model,
    boxes,
    image_path,
    image_rgb=None,
    device="cuda",
    threshold=0.5,
    resize_to=(256, 256),
    output_to_file=False,
    output_name=None,
    output_folder="segmentation/segmentation_output/custom_segmentation/",
):
    """
    Run UNet inference per YOLO bounding box.

    Parameters
    ----------
    model : torch.nn.Module
        Trained UNet model.

    boxes : np.ndarray
        Array of boxes with shape (N, 4), format [x1, y1, x2, y2].

    image_path : str
        Path to input image.

    image_rgb : np.ndarray, optional
        RGB image (H, W, 3). Loaded if None.

    Returns
    -------
    list[np.ndarray]
        List of binary masks, each (H, W), aligned to original image.
    """

    model.eval()
    model.to(device)

    if image_rgb is None:
        image_rgb = np.array(Image.open(image_path).convert("RGB"))

    H, W = image_rgb.shape[:2]
    masks = []

    for box in boxes:
        x1, y1, x2, y2 = box.astype(int)

        # Safety clamp
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        if x2 <= x1 or y2 <= y1:
            masks.append(np.zeros((H, W), dtype=np.uint8))
            continue

        crop = image_rgb[y1:y2, x1:x2]
        crop_h, crop_w = crop.shape[:2]

        crop_resized = cv2.resize(crop, resize_to)
        crop_tensor = torch.from_numpy(crop_resized).float() / 255.0
        crop_tensor = crop_tensor.permute(2, 0, 1).unsqueeze(0).to(device)


        logits = model(crop_tensor)
        probs = torch.sigmoid(logits)
        mask_crop = (probs > threshold).float()[0, 0].cpu().numpy()

        mask_crop = cv2.resize(
            mask_crop,
            (crop_w, crop_h),
            interpolation=cv2.INTER_NEAREST
        )

        full_mask = np.zeros((H, W), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = mask_crop.astype(np.uint8)

        masks.append(full_mask)

    if output_to_file:
        if output_name is None:
            output_name = get_file_name(image_path)

        out_dir = os.path.join(output_folder, output_name)
        os.makedirs(out_dir, exist_ok=True)

        output_masks_to_file_overlay(
            out_dir,
            image_path,
            masks,
            image_rgb,
            suffix="unet_box_outline"
        )

    print("UNet box-wise inference done")
    return masks
