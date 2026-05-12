import argparse
import json
import shutil
import sys
import time
from pathlib import Path


#Device selection─────────

def _best_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _recommend_batch(device: str) -> int:
    """Heuristic batch size when --batch is not specified."""
    import torch
    if device == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram >= 40: return 64   
        if vram >= 24: return 32   
        if vram >= 16: return 16  
        if vram >= 8:  return 8  
        return 4
    if device == "mps":
        return 8
    return 4


def _gpu_memory_summary() -> str:
    """Return a one-line GPU memory status string (CUDA only)."""
    import torch
    if not torch.cuda.is_available():
        return ""
    allocated = torch.cuda.memory_allocated(0) / 1e9
    total     = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"{allocated:.1f} GB allocated / {total:.0f} GB total"


AUGMENTATION_PROFILE = {
    #Geometry─────────
    "degrees":     12.0,   #±12° yaw jitter (dashcam vibration)
    "translate":   0.15,   #±15% translation
    "scale":       0.7,    #scale jitter [0.3, 1.7]× — trains on close & far potholes
    "shear":       3.0,    #±3° shear (mild lens barrel distortion)
    "perspective": 0.0008, #subtle perspective warp
    "flipud":      0.0,    #never — road is always below horizon
    "fliplr":      0.5,    #safe horizontal flip

    #Composition──────
    "mosaic":      1.0,    #full-rate mosaic: exposes model to 4 scenes per step
    "mixup":       0.15,   #slightly higher than 0.1 — smooths the decision boundary
    "copy_paste":  0.15,   #paste potholes from other images onto clean road patches
    "erasing":     0.4,    #random rectangular erasure simulates motion-blur occlusion

    #Photometric — most important for night / rain / shadow robustness ─
    "hsv_h":  0.015,  #minimal hue shift (roads stay gray/asphalt-colored)
    "hsv_s":  0.9,    #heavy saturation jitter — covers dry/wet/rained-on asphalt
    "hsv_v":  0.6,    #strong brightness jitter — night tunnels → bright noon sun

    #Learning rate schedule (cosine with warmup)
    "lr0":             0.01,
    "lrf":             0.005,   #final LR 50% of initial (less aggressive decay)
    "momentum":        0.937,
    "weight_decay":    0.0005,
    "warmup_epochs":   3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr":  0.1,

    #Loss weights — upweight box for small potholes
    "box":   7.5,
    "cls":   0.5,
    "dfl":   1.5,

    #Label smoothing — prevents overconfident probability outputs─
    "label_smoothing": 0.05,
}


#Core training function───

def train(
    dataset_dir: str | Path,
    model_size: str = "yolov8m.pt",   #upgraded from yolov8n (3.2M) to yolov8m (25M)
    epochs: int = 150,                 #more epochs for better convergence on small objects
    batch: int = -1,                   #-1 = auto
    imgsz: int = 1280,                 #larger images improve small-pothole recall at distance
    device: str = "auto",
    project: str = "runs/pothole",
    name: str = "train",
    patience: int = 50,                #longer patience — small-object mAP improves slowly
    workers: int = 8,                  #8 workers optimal for A40; capped to 4 on CPU/MPS
    resume: bool = False,
    pretrained_weights: str | Path | None = None,  #override base weights
    amp: bool = True,                  #mixed precision (ignored on CPU/MPS)
    close_mosaic: int = 15,            #disable mosaic in last 15 epochs for clean fine-tuning
    save_period: int = 10,             #checkpoint every 10 epochs for safety on long runs
    cache: bool | str = False,         #True=disk cache, "ram"=RAM cache (fast if dataset fits)
) -> Path:
    """
    Train YOLOv8 on the pothole dataset.

    Returns:
        Path to the best trained weights file.
    """
    from ultralytics import YOLO

    dataset_dir = Path(dataset_dir)
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found in {dataset_dir}")

  
    if device == "auto":
        device = _best_device()


    if batch == -1:
        batch = _recommend_batch(device)

    use_amp = amp and device == "cuda"

    if device in ("mps", "cpu"):
        workers = min(workers, 4)

    weights = str(pretrained_weights) if pretrained_weights else model_size
    model = YOLO(weights)

    gpu_mem = _gpu_memory_summary()
    print("\n" + "=" * 60)
    print(f"  Model          : {weights}")
    print(f"  Device         : {device.upper()}")
    print(f"  Batch size     : {batch}")
    print(f"  Epochs         : {epochs}  (early stop patience={patience})")
    print(f"  Image size     : {imgsz}")
    print(f"  Mixed precision: {use_amp}")
    print(f"  Workers        : {workers}")
    print(f"  Cache          : {cache}")
    if gpu_mem:
        print(f"  GPU memory     : {gpu_mem}")
    print(f"  Dataset        : {data_yaml}")
    print("=" * 60 + "\n")

    start = time.time()

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        patience=patience,
        workers=workers,
        resume=resume,
        amp=use_amp,
        close_mosaic=close_mosaic,
        save_period=save_period,
        cache=cache, 
        **AUGMENTATION_PROFILE,
        verbose=True,
        plots=True,         
        save=True,
        exist_ok=True,
    )

    elapsed = time.time() - start
    print(f"\n[train] Finished in {elapsed/60:.1f} min")

    #Locate best weights
    best_weights = Path(project) / name / "weights" / "best.pt"
    if not best_weights.exists():
        #Fallback: search recursively
        candidates = list(Path(project).rglob("best.pt"))
        if candidates:
            best_weights = sorted(candidates)[-1]
        else:
            raise FileNotFoundError("best.pt not found after training")

    print(f"[train] Best weights: {best_weights}")
    return best_weights


#Post-training: copy model into project

def deploy_model(best_weights: Path, project_root: Path | None = None) -> Path:
    """
    Copy trained weights to models/pothole_yolov8.pt so the existing
    pipeline (config.POTHOLE_MODEL_PATH) picks it up automatically.
    """
    if project_root is None:
        #Assume training/train.py lives two levels below project root
        project_root = Path(__file__).parent.parent

    dest = project_root / "models" / "pothole_yolov8.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, dest)
    print(f"\n[deploy] Model deployed → {dest}")
    print("         Set POTHOLE_MODEL_PATH or restart the Flask app to use it.")
    return dest


#CLI──

def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 pothole detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-dir", default="Pothole-Detection-1",
                        help="Path to Roboflow YOLOv8 dataset directory")
    parser.add_argument("--model", default="yolov8m.pt",
                        help="Base YOLO weights: yolov8n.pt | yolov8s.pt | yolov8m.pt | yolov8l.pt")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=-1,
                        help="-1 = auto-detect based on device VRAM")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="auto",
                        help="auto | cuda | mps | cpu | 0 | 0,1 (for multi-GPU)")
    parser.add_argument("--project", default="runs/pothole")
    parser.add_argument("--name", default="train")
    parser.add_argument("--patience", type=int, default=50,
                        help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--workers", type=int, default=8,
                        help="DataLoader workers (8 for CUDA/A40; auto-capped to 4 on CPU/MPS)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--no-amp", dest="amp", action="store_false",
                        help="Disable mixed-precision training")
    parser.add_argument("--cache", default=False,
                        const="ram", nargs="?",
                        help="Cache images: --cache (disk) or --cache ram (RAM). "
                             "RAM cache is fastest; requires ~dataset_size × 3 GB free RAM.")
    parser.add_argument("--no-deploy", action="store_true",
                        help="Skip copying weights to models/pothole_yolov8.pt")
    parser.add_argument("--pretrained", default=None,
                        help="Path to custom pre-trained weights (overrides --model)")
    args = parser.parse_args()

    best = train(
        dataset_dir=args.dataset_dir,
        model_size=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        resume=args.resume,
        amp=args.amp,
        pretrained_weights=args.pretrained,
        cache=args.cache,
    )

    if not args.no_deploy:
        deploy_model(best)

    print("\n[done] Next step: python training/evaluate.py --weights", best)


if __name__ == "__main__":
    main()
