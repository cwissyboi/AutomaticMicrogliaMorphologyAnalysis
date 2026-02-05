import torch
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from helpers import get_file_name
import os

def yolo_inference(
    yolo,
    image_path,
    output_name,
    output_to_file=False,
    confidence_threshold=0.5,
    yolo_version="v8",  # "v5" or "v8"
):
    """
    Run YOLO object detection on a single image and return filtered bounding boxes.
    """

    results = yolo(image_path)

    # ----------------------------
    # YOLOv5 handling
    # ----------------------------
    if yolo_version == "v5":
        # results.xyxy[0] -> (N, 6): [x1, y1, x2, y2, conf, cls]
        detections = results.xyxy[0]

        # confidence filtering
        filtered = detections[detections[:, 4] > confidence_threshold]

        # update results for rendering
        results.xyxy[0] = filtered

        # boxes
        boxes = filtered[:, :4].cpu().numpy()

        if output_to_file:
            annotated_img = results.render()[0]  # RGB
            annotated_img_bgr = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)

    # ----------------------------
    # YOLOv8 handling
    # ----------------------------
    elif yolo_version == "v8":
        # results is a list of Results objects
        r = results[0]

        if r.boxes is None or len(r.boxes) == 0:
            return np.empty((0, 4))

        # tensors
        boxes_xyxy = r.boxes.xyxy
        conf = r.boxes.conf
        cls = r.boxes.cls  # kept for completeness

        # confidence filtering
        keep = conf > confidence_threshold

        boxes = boxes_xyxy[keep].cpu().numpy()

        if output_to_file:
            annotated_img = r.plot()  # BGR already

    else:
        raise ValueError("yolo_version must be 'v5' or 'v8'")

    # ----------------------------
    # Save annotated image
    # ----------------------------
    if output_to_file:
        file_name = get_file_name(image_path)
        output_folder = f"object_detection/object_detection_output/{output_name}/"
        os.makedirs(output_folder, exist_ok=True)

        cv2.imwrite(
            f"{output_folder}{file_name}_yolo_identification.jpg",
            annotated_img
        )

    return boxes
