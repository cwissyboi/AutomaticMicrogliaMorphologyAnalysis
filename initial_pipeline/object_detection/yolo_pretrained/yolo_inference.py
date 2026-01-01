import torch
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from helpers import get_file_name

def yolo_inference(yolo, image_path, output_to_file = False, confidence_threshold = 0.5):
    """
    Run YOLO object detection on a single image and return filtered bounding boxes.

    This function performs inference using a pre-loaded YOLO model, filters
    detections based on a confidence threshold, optionally visualizes and saves
    the detection results, and returns the remaining bounding boxes.

    Parameters
    ----------
    yolo : object
        A YOLO model instance (e.g. loaded via `torch.hub.load` or Ultralytics API)
        that supports callable inference on an image path.
    image_path : str or pathlib.Path
        Path to the input image on which object detection will be performed.
    output_to_file : bool, optional (default=False)
        If True, saves an annotated image with bounding boxes drawn to disk.
    confidence_threshold : float, optional (default=0.5)
        Minimum confidence score required for a detection to be retained.

    Returns
    -------
    boxes : np.ndarray
        Array of filtered bounding boxes with shape (N, 4), where each box is
        represented as [x1, y1, x2, y2] in pixel coordinates.

    Notes
    -----
    - This function assumes output, where detections are returned
      in the format: [x1, y1, x2, y2, confidence, class_id].
    - Bounding boxes are returned on the CPU as NumPy arrays.
    """

    results = yolo(image_path)

    # YOLO format: [x1, y1, x2, y2, conf, class]
    detections = results.xyxy[0]

    # Keep only confidence > 0.5
    filtered = detections[detections[:, 4] > confidence_threshold]

    # Replace detections inside results object
    results.xyxy[0] = filtered

    # opens a window with a temp file to show the results
    # results.show()

    # Get bounding boxes (xyxy)
    boxes = results.xyxy[0][:, :4].cpu().numpy()  # shape (N, 4)

    if output_to_file:
        annotated_img = results.render()[0]  # RGB
        annotated_img_bgr = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
        file_name = get_file_name(image_path)

        cv2.imwrite(
            f"object_detection/object_detection_output/{file_name}_yolo_identification.jpg",
            annotated_img_bgr
        )

    print('YOLO object identification done')
    return boxes