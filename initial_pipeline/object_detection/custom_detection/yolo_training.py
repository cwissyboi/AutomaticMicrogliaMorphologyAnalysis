from ultralytics import YOLO
import torch
import os
from datetime import datetime

def print_yolo_metrics(metrics):
    print("\n=== YOLO Evaluation Metrics (Test Set) ===")

    print(f"mAP@50:      {metrics.box.map50:.3f}")
    print(f"mAP@50–95:   {metrics.box.map:.3f}")
    print(f"Precision:   {metrics.box.mp:.3f}")
    print(f"Recall:      {metrics.box.mr:.3f}")

    print("\nPer-class results:")
    for i, name in metrics.names.items():
        print(
            f"  Class '{name}': "
            f"P={metrics.box.p[i]:.3f}, "
            f"R={metrics.box.r[i]:.3f}, "
            f"AP50={metrics.box.ap50[i]:.3f}, "
            f"AP={metrics.box.ap[i]:.3f}"
        )


def main():

    DATA_YAML = "data.yaml"

    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f'using device {device}')

    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(script_dir)

    # Create main runs folder
    runs_dir = os.path.join(script_dir, "yolo_runs")
    os.makedirs(runs_dir, exist_ok=True)  # ensures folder exists

    model = YOLO("yolov8s.pt")

    # Timestamped run name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"yolo_output_{timestamp}"

    results = model.train(
        data=DATA_YAML,
        epochs=1,
        imgsz=512,
        batch=16,
        device=device,
        name=run_name,
        project=runs_dir  
    )


    # Load best model from this specific run
    best_path = os.path.join(results.save_dir, "weights", "best.pt")
    model = YOLO(best_path)

    # Evaluate on test split
    metrics = model.val(
        data=DATA_YAML,
        split="test",
        imgsz=512
    )

    print_yolo_metrics(metrics)


if __name__ == "__main__":
    main()