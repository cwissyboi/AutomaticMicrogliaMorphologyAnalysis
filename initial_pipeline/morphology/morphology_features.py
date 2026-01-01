import numpy as np
import cv2
import os
from skimage.morphology import skeletonize
from helpers import get_file_name


def get_skeletons(image_rgb, image_path, cell_masks, soma_masks, output_to_file = False, output_folder = 'morphology/skeleton_outputs/'): 

    overlay = image_rgb.copy()
    skeletons = []

    for mask, soma_mask in zip(cell_masks, soma_masks):
        # ----------------------------
        # 1. Skeletonise full cell
        # ----------------------------
        cell_bin = mask > 0
        skeleton = skeletonize(cell_bin)

        # Remove skeleton inside soma
        soma_bin = soma_mask > 0
        process_skeleton = skeleton & (~soma_bin)
        skeletons.append(process_skeleton)



        if (output_to_file):
            # Draw process skeleton (GREEN)
            ys, xs = np.where(process_skeleton)
            overlay[ys, xs] = [0, 255, 0]

            # ----------------------------
            # 2. Draw soma outline
            # ----------------------------
            soma_uint8 = (soma_bin.astype(np.uint8) * 255)

            contours, _ = cv2.findContours(
                soma_uint8,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            # Draw soma contours (BLUE)
            cv2.drawContours(
                overlay,
                contours,
                contourIdx=-1,
                color=(0, 0, 255),
                thickness=2
            )

    if (output_to_file):
        # Save combined overlay
        file_name = get_file_name(image_path)
        out_path = f"{output_folder}{file_name}_skeleton.png"
        cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return skeletons