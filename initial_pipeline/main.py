from helpers import parse_args
from object_detection.yolo_pretrained.yolo_inference import yolo_inference
from segmentation.sam.sam_inference import sam_inference
import torch
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor


def main():
    args = parse_args()
    image_path = args["image_path"]


    # Load yolo model from saved output
    # TO DO: Make the parameters here input args and the same for SAM paths
    yolo = torch.hub.load(
        'ultralytics/yolov5',
        'custom',
        path='../yolov3_weights.pt'
    )

    # Initiate SAM model
    sam = sam_model_registry["vit_b"](
        checkpoint="../sam_files/sam_vit_b_01ec64.pth"
    )
    sam.to("cuda" if torch.cuda.is_available() else "cpu")

    sam_predictor = SamPredictor(sam)


    # Pipeline steps
    yolo_boxes = yolo_inference(yolo, image_path, output_to_file=True)

    sam_masks = sam_inference(sam_predictor, yolo_boxes, image_path, output_to_file=True)

if __name__ == "__main__":
    main()
