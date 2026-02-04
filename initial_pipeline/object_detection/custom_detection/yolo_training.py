
from ultralytics import YOLO


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

    # Paths
    DATA_YAML = "data.yaml"

    # Load pretrained model
    model = YOLO("yolov8s.pt")  # or yolov8n.pt

    results = model.train(
    data=DATA_YAML,
    epochs=1,
    imgsz=512,
    batch=16,
    device='cpu',      # set to "cpu" if no GPU
    name="yolo_cells"
    )



        # Load best model
    model = YOLO(r"yolo_cells\weights\best.pt")

    # Evaluate on test split
    metrics = model.val(
        data=DATA_YAML,
        split="test",
        imgsz=512
    )

    print_yolo_metrics(metrics)


if __name__ == "__main__":
    main()