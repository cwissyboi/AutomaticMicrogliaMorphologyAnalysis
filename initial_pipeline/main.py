from helpers import parse_args, get_file_name
from object_detection.yolo_pretrained.yolo_inference import yolo_inference
from segmentation.soma_segmentation.gaussian_filter import get_gaussian_filter_soma_masks
from morphology.morphology_features import get_skeletons, get_morphology_dataframe
from segmentation.custom_segmentation.training.unet import UNet
from segmentation.custom_segmentation.segmentation_inference import unet_inference
from segmentation.custom_segmentation.crf import connect_all_masks

import torch
import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
import time


def main():

    overall_start = time.perf_counter()

    args = parse_args()
    input_folder_path = args["input_folder_path"]
    output_name = args["output_name"]
    scan_name = args["scan_name"]

    print("custom yolo")
    yolo = YOLO(r"object_detection/custom_detection/yolo_good_runs/28_2_rat_pretraining.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet()
    ckpt_path = "segmentation/custom_segmentation/checkpoints/best_run_25_1.pth"
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # ---- Timing accumulators ----
    timings = {
        "yolo": 0.0,
        "unet": 0.0,
        "crf": 0.0,
        "gaussian": 0.0,
        "skeleton": 0.0,
        "dataframe": 0.0
    }

    input_dir = Path(input_folder_path)
    all_results = []

    image_paths = [p for p in input_dir.iterdir() if p.is_file()]
    counter = 0

    for image_path in tqdm(image_paths, desc=f"Images in {scan_name or input_dir.name}"):

            if not image_path.is_file():
                continue

            # if (counter > 100): 
            #     break

            counter = counter + 1

            image_path = str(image_path)
            file_name = get_file_name(image_path)

            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # ---------------- YOLO ----------------
            t0 = time.perf_counter()
            yolo_boxes = yolo_inference(
                yolo,
                image_path,
                output_name=output_name,
                output_to_file=True,
                scan_folder=scan_name, 
                confidence_threshold=0.5
            )
            if device == "cuda":
                torch.cuda.synchronize()
            timings["yolo"] += time.perf_counter() - t0

            if len(yolo_boxes) == 0:
                continue

            # ---------------- UNet ----------------
            t0 = time.perf_counter()
            segmentation_masks = unet_inference(
                model,
                yolo_boxes,
                image_path=image_path,
                image_rgb=image_rgb,
                device=device,
                output_to_file=True,
                output_name=output_name,
                expand_boxes=False,
                scan_folder=scan_name
            )

            if device == "cuda":
                torch.cuda.synchronize()
            timings["unet"] += time.perf_counter() - t0

            

            t0 = time.perf_counter()
            connected_masks = connect_all_masks(
                segmentation_masks,
                image,
                image_path,
                output_to_file=True,
                output_name=output_name,
                scan_folder=scan_name
            )
            
            timings["crf"] += time.perf_counter() - t0

            # ---------------- Gaussian ----------------
            t0 = time.perf_counter()
            soma_masks = get_gaussian_filter_soma_masks(
                yolo_boxes,
                image_path,
                image_rgb,
                output_name=output_name,
                output_to_file=True,
                scan_folder=scan_name
            )
            timings["gaussian"] += time.perf_counter() - t0

           
            t0 = time.perf_counter()
            skeletons = get_skeletons(
                image_rgb,
                image_path,
                connected_masks,
                soma_masks,
                output_to_file=True,
                output_name=output_name,
                scan_folder=scan_name
            )
            timings["skeleton"] += time.perf_counter() - t0

            
            t0 = time.perf_counter()
            results_df = get_morphology_dataframe(
                segmentation_masks,
                skeletons,
                soma_masks,
                yolo_boxes
            )
            timings["dataframe"] += time.perf_counter() - t0

            results_df["image_name"] = file_name
            results_df["scan_name"] = scan_name
            results_df["global_cell_id"] = (
                results_df["image_name"]
                + "_cell_"
                + results_df["cell_id"].astype(str)
            )

            all_results.append(results_df)

    final_df = pd.concat(all_results, ignore_index=True)

    csv_stem = f"{output_name}_{scan_name}" if scan_name else output_name
    output_csv = f"morphology/morphology_outputs/{csv_stem}.csv"
    final_df.to_csv(output_csv, index=False)

    # ---------------- Final Timing Summary ----------------
    overall_time = time.perf_counter() - overall_start

    print("\n========== TIMING SUMMARY ==========")
    for k, v in timings.items():
        print(f"{k:<12}: {v:.2f} seconds")

    print(f"\nTotal runtime : {overall_time:.2f} seconds")
    print("=====================================")


if __name__ == "__main__":
    main()
