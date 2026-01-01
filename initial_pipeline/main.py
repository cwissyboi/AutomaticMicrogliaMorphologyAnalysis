from helpers import parse_args
from object_detection.yolo_pretrained.yolo_inference import yolo_inference
import torch
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor


def main():
    args = parse_args()
    image_path = args["image_path"]


    # Load yolo model from saved output
    yolo = torch.hub.load(
        'ultralytics/yolov5',
        'custom',
        path='../yolov3_weights.pt'
    )

    
    boxes = yolo_inference(yolo, image_path, output_to_file=True)


if __name__ == "__main__":
    main()
