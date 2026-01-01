import cv2
import numpy as np
from helpers import output_masks_to_file_overlay


def get_gaussian_filter_soma_masks(boxes, image_path, image_rgb, output_to_file = False, output_folder = "segmentation/segmentation_output/soma_segmentation/"):
    soma_masks = []
    red = image_rgb[:, :, 0]

    for box in boxes:
        x1, y1, x2, y2 = box.astype(int)

        # 1. Extract ROI from RED channel
        roi = red[y1:y2, x1:x2].astype(np.float32)

        # 2. Normalize
        roi = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX)

        # 3. INVERT so soma (dark) becomes bright
        roi_inv = 255 - roi

        # 4. Gaussian blur (soma-scale)
        blurred = cv2.GaussianBlur(
            roi_inv,
            (0, 0),
            sigmaX=6,   # tune: soma radius / ~2
            sigmaY=6
        )

        # 5. Relative threshold (NOT global)
        thresh = 0.7 * blurred.max()
        soma_roi = (blurred > thresh).astype(np.uint8) * 255

        # 6. Keep component containing the brightest pixel
        _, _, _, maxLoc = cv2.minMaxLoc(blurred)

        num_labels, labels = cv2.connectedComponents(soma_roi)
        soma_label = labels[maxLoc[1], maxLoc[0]]
        soma_roi = (labels == soma_label).astype(np.uint8) * 255

        # 7. Place back into full image
        soma_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)

        soma_mask[y1:y2, x1:x2] = soma_roi

        soma_masks.append(soma_mask)
    

    if (output_to_file): 
        output_masks_to_file_overlay(output_folder, image_path, soma_masks, image_rgb, suffix = 'soma_mask')
    
    return soma_masks