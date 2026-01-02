from helpers import parse_args, get_file_name
from object_detection.yolo_pretrained.yolo_inference import yolo_inference
from segmentation.sam.sam_inference import sam_inference
from segmentation.soma_segmentation.gaussian_filter import get_gaussian_filter_soma_masks
from morphology.morphology_features import get_skeletons, get_morphology_dataframe
import torch
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def main():
    args = parse_args()
    input_folder_path = args["input_folder_path"]
    output_name = args["output_name"]


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


    input_dir = Path(input_folder_path)
    all_results = []

    image_paths = list(input_dir.iterdir())
    for image_path in tqdm(image_paths, desc="Processing images"):
        if image_path.is_file():
            # convert it from a path to a string to avoid YOLO's 'ends with extension' error
            image_path = str(image_path)
            print(image_path)
            file_name = get_file_name(image_path)
            image = cv2.imread(image_path)
            h, w = image.shape[:2]
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Pipeline steps
            yolo_boxes = yolo_inference(yolo, image_path, output_name = output_name,  output_to_file=True)
            sam_masks = sam_inference(sam_predictor, yolo_boxes, image_path,  output_name = output_name, image_rgb = image_rgb, output_to_file=True)
            soma_masks = get_gaussian_filter_soma_masks(yolo_boxes, image_path, image_rgb, output_name = output_name,  output_to_file= True)
            skeletons = get_skeletons(image_rgb, image_path, sam_masks, soma_masks, output_to_file = True, output_name = output_name)
            results_df = get_morphology_dataframe(sam_masks, skeletons, soma_masks, yolo_boxes)

            # results_df["image_path"] = image_path
            results_df["image_name"] = file_name

            # Global unique cell ID
            results_df["global_cell_id"] = (
                results_df["image_name"]
                + "_cell_"
                + results_df["cell_id"].astype(str)
            )

            all_results.append(results_df)

    final_df = pd.concat(all_results, ignore_index=True)

    output_csv = f"morphology/morphology_outputs/{output_name}.csv"
    final_df.to_csv(output_csv, index=False)




if __name__ == "__main__":
    main()