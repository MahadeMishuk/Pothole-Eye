import argparse
from pathlib import Path


#Search space override for pothole-specific tuning
#Keys not listed here use Ultralytics defaults.
POTHOLE_SEARCH_SPACE = {
    "lr0":          (1e-5, 1e-1),
    "lrf":          (0.01, 1.0),
    "momentum":     (0.6, 0.98),
    "weight_decay": (0.0, 1e-3),
    "warmup_epochs": (0.0, 5.0),
    "box":          (1.0, 20.0),
    "cls":          (0.2, 4.0),
    "hsv_h":        (0.0, 0.1),
    "hsv_s":        (0.0, 0.9),
    "hsv_v":        (0.0, 0.9),
    "degrees":      (0.0, 15.0),
    "translate":    (0.0, 0.3),
    "scale":        (0.0, 0.9),
    "shear":        (0.0, 5.0),
    "perspective":  (0.0, 0.001),
    "flipud":       (0.0, 0.0),   #disabled: road is always below
    "fliplr":       (0.0, 1.0),
    "mosaic":       (0.0, 1.0),
    "mixup":        (0.0, 0.3),
    "copy_paste":   (0.0, 0.3),
    "label_smoothing": (0.0, 0.1),
}


def _best_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def tune(
    dataset_dir: str | Path,
    model: str = "yolov8n.pt",
    iterations: int = 10,
    iter_epochs: int = 30,
    imgsz: int = 640,
    batch: int = -1,
    device: str = "auto",
    project: str = "runs/pothole",
    name: str = "tune",
):
    """
    Run genetic-algorithm hyperparameter search.

    Each iteration trains a fresh model for iter_epochs epochs and evaluates
    mAP50 on the validation set. The best hyperparameters are saved to YAML.
    """
    from ultralytics import YOLO

    dataset_dir = Path(dataset_dir)
    data_yaml = dataset_dir / "data.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    if device == "auto":
        device = _best_device()

    print("\n" + "=" * 60)
    print("  Hyperparameter Tuning")
    print("=" * 60)
    print(f"  Model      : {model}")
    print(f"  Iterations : {iterations}")
    print(f"  Epochs/iter: {iter_epochs}")
    print(f"  Device     : {device.upper()}")
    print(f"  Dataset    : {data_yaml}")
    print(f"\n  Est. time  : {iterations * iter_epochs * 0.05:.0f}–{iterations * iter_epochs * 0.2:.0f} min (hardware-dependent)")
    print("=" * 60 + "\n")

    yolo = YOLO(model)

    yolo.tune(
        data=str(data_yaml),
        epochs=iter_epochs,
        iterations=iterations,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        space=POTHOLE_SEARCH_SPACE,
        plots=True,
        save=True,
        exist_ok=True,
        val=True,
        amp=device == "cuda",
    )

    best_yaml = Path(project) / name / "best_hyperparameters.yaml"
    if best_yaml.exists():
        print(f"\n[tune] Best hyperparameters saved → {best_yaml}")
        print("[tune] To use them in full training:")
        print(f"       python training/train.py --dataset-dir {dataset_dir} \\")
        print(f"           --pretrained {Path(project)/name/'weights'/'best.pt'}")
    else:
        print("[tune] Done. Check runs/pothole/tune/ for results.")


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for pothole YOLOv8",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-dir", default="pothole-detection-1")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of tuning iterations (each trains a full model)")
    parser.add_argument("--iter-epochs", type=int, default=30,
                        help="Epochs per tuning iteration")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--project", default="runs/pothole")
    parser.add_argument("--name", default="tune")
    args = parser.parse_args()

    tune(
        dataset_dir=args.dataset_dir,
        model=args.model,
        iterations=args.iterations,
        iter_epochs=args.iter_epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
