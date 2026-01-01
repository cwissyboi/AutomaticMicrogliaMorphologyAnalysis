from helpers import parse_args
from object_detection.yolo_pretrained.yolo_inference import yolo_inference
from segmentation.sam.sam_inference import sam_inference
from segmentation.soma_segmentation.gaussian_filter import get_gaussian_filter_soma_masks
from morphology.morphology_features import get_skeletons, skeleton_metrics
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


    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    # Pipeline steps
    yolo_boxes = yolo_inference(yolo, image_path, output_to_file=True)

    sam_masks = sam_inference(sam_predictor, yolo_boxes, image_path, image_rgb = image_rgb, output_to_file=True)

    soma_masks = get_gaussian_filter_soma_masks(yolo_boxes, image_path, image_rgb, output_to_file= True)
    skeletons = get_skeletons(image_rgb, image_path, sam_masks, soma_masks, output_to_file = True)



    results = []

    for i, mask in enumerate(sam_masks):
        skeleton = skeletons[i]
        filled_soma = soma_masks[i]
        length_px, num_branches, soma_area, num_components, cell_area = skeleton_metrics(mask, skeleton, filled_soma)

        results.append({
            "cell_id": i,
            "length_pixels": length_px,
            "num_branches": num_branches, 
            "soma_area": soma_area, 
            "num_components": num_components, 
            "cell_area": cell_area, 
        })

        print(
            f"Cell {i:02d} | "
            f"Length: {length_px:5d} px | "
            f"Branches: {num_branches}  | "
            f"soma_area: {soma_area:5d} px | "
            f"num_components: {num_components:5d}  | "
            f"cell_area: {cell_area:5d}  | "
        )



if __name__ == "__main__":
    main()