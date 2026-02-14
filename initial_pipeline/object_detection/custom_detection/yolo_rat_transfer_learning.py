from ultralytics import YOLO
import torch
import os
from datetime import datetime

def print_yolo_metrics(metrics):
    print("\n=== YOLO Evaluation Metrics (Test Set) ===")

    print(f"mAP@50:      {metrics.box.map50:.5f}")
    print(f"mAP@50–95:   {metrics.box.map:.5f}")
    print(f"Precision:   {metrics.box.mp:.5f}")
    print(f"Recall:      {metrics.box.mr:.5f}")

    print("\nPer-class results:")
    for i, name in metrics.names.items():
        print(
            f"  Class '{name}': "
            f"P={metrics.box.p[i]:.5f}, "
            f"R={metrics.box.r[i]:.5f}, "
            f"AP50={metrics.box.ap50[i]:.5f}, "
            f"AP={metrics.box.ap[i]:.5f}"
        )


def main():

    RAT_DATA_YAML = "rat_data.yaml"
    HUMAN_DATA_YAML = "data.yaml"

    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f'using device {device}')

    script_dir = os.path.dirname(os.path.abspath(__file__))
    runs_dir = os.path.join(script_dir, "yolo_runs")
    os.makedirs(runs_dir, exist_ok=True)


    model = YOLO("yolov8s.pt")

    print('Started training on rat dataset')
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    rat_run_name = f"rat_pretraining_{timestamp}"

    rat_results = model.train(
        data=RAT_DATA_YAML,
        epochs=100,
        patience=10,
        imgsz=512,
        batch=16,
        device=device,
        name=rat_run_name,
        project=runs_dir
    )

    rat_best_path = os.path.join(rat_results.save_dir, "weights", "best.pt")

    print("\nFinished rat pretraining")
    print(f"Best rat weights: {rat_best_path}")


    # fine tune on human data

    model = YOLO(rat_best_path)  # ← IMPORTANT CHANGE

    human_run_name = f"human_finetune_{timestamp}"

    human_results = model.train(
        data=HUMAN_DATA_YAML,
        epochs=100,              
        patience=10,
        imgsz=512,
        batch=16,
        device=device,
        lr0=0.001,              # lower learning rate for finetuning
        name=human_run_name,
        project=runs_dir
    )

    human_best_path = os.path.join(human_results.save_dir, "weights", "best.pt")

    print("Finished human finetuning")
    print(f"Best human weights: {human_best_path}")

    # EVALUATE ON HUMAN TEST

    model = YOLO(human_best_path)

    metrics = model.val(
        data=HUMAN_DATA_YAML,
        split="test",
        imgsz=512
    )

    print_yolo_metrics(metrics)


if __name__ == "__main__":
    main()