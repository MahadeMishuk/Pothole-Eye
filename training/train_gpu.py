import argparse
import shutil
import sys
import time
from pathlib import Path

import torch
import yaml

#Absolute paths (RunPod)──
PROJECT_ROOT  = Path("/workspace/Pothole-I")
DATASET_DIR   = PROJECT_ROOT / "Pothole-Detection-1"
ORIGINAL_YAML = DATASET_DIR / "data.yaml"
FIXED_YAML    = DATASET_DIR / "data_abs.yaml"   #absolute-path version we generate
RUNS_DIR      = PROJECT_ROOT / "runs" / "train"
MODELS_DIR    = PROJECT_ROOT / "models"
DEPLOY_PATH   = MODELS_DIR / "best.pt"



A40_AUGMENTATION = {
    #Geometry
    "degrees":      10.0,    #±10° — dashcam tilt/banking
    "translate":    0.15,
    "scale":        0.6,     #scale jitter [0.4×, 1.6×] — potholes at varying distances
    "shear":        2.0,
    "perspective":  0.0005,
    "flipud":       0.0,     #DISABLED — vertical flip breaks road-scene depth cues
    "fliplr":       0.5,
    #Mosaic / mixing
    "mosaic":       1.0,
    "mixup":        0.15,
    "copy_paste":   0.15,    #paste pothole instances onto background road crops
    "erasing":      0.4,     #random rectangular erase — simulates debris / occlusion
    #Photometric — critical for wet/dry/night/noon road variation
    "hsv_h":        0.015,
    "hsv_s":        0.9,     #wet vs dry asphalt: ±40% saturation shift
    "hsv_v":        0.6,     #dawn to noon: wide brightness range
    #LR schedule (cosine with warmup)
    "lr0":          0.01,
    "lrf":          0.01,
    "momentum":     0.937,
    "weight_decay": 0.0005,
    "warmup_epochs":    3.0,
    "warmup_momentum":  0.8,
    "warmup_bias_lr":   0.1,
    #Loss weights — upweight box precision for small pothole localisation
    "box":          7.5,
    "cls":          0.5,
    "dfl":          1.5,
    #Regularisation
    "label_smoothing": 0.05,
    "dropout":      0.0,
}


#VRAM helpers─────────────

def _vram_stats(device_idx: int = 0) -> str:
    allocated = torch.cuda.memory_allocated(device_idx) / 1e9
    reserved  = torch.cuda.memory_reserved(device_idx) / 1e9
    total     = torch.cuda.get_device_properties(device_idx).total_memory / 1e9
    free      = (torch.cuda.mem_get_info(device_idx)[0]) / 1e9
    return (
        f"allocated={allocated:.1f}GB  reserved={reserved:.1f}GB  "
        f"free={free:.1f}GB  total={total:.0f}GB"
    )


def _print_gpu_banner(device: str):
    """Print GPU stats before training starts."""
    if not torch.cuda.is_available():
        print(f"  Device: {device} (no CUDA)")
        return
    idx = 0
    props = torch.cuda.get_device_properties(idx)
    free, total = torch.cuda.mem_get_info(idx)
    print(f"  GPU              : {props.name}")
    print(f"  CUDA capability  : {props.major}.{props.minor}")
    print(f"  SM count         : {props.multi_processor_count}")
    print(f"  VRAM total       : {total/1e9:.1f} GB")
    print(f"  VRAM free        : {free/1e9:.1f} GB")
    print(f"  PyTorch          : {torch.__version__}  (CUDA {torch.version.cuda})")


#data.yaml fix────────────

def build_absolute_yaml(original: Path, fixed: Path, dataset_dir: Path) -> Path:
    """
    Roboflow exports data.yaml with relative paths like `../train/images`.
    When YOLO resolves these relative to the yaml file location, they point
    outside the dataset directory and cause FileNotFoundError.

    This function writes a new yaml with absolute paths.
    """
    with open(original) as f:
        cfg = yaml.safe_load(f)

    cfg["train"] = str(dataset_dir / "train"  / "images")
    cfg["val"]   = str(dataset_dir / "valid"  / "images")
    cfg["test"]  = str(dataset_dir / "test"   / "images")

    #Fix class name: Roboflow sometimes exports names: ['0'] instead of ['pothole']
    if cfg.get("names") == ["0"] or cfg.get("names") == [0]:
        cfg["names"] = ["pothole"]
        print("  [yaml] Renamed class '0' → 'pothole'")

    with open(fixed, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"  [yaml] Wrote absolute-path data.yaml → {fixed}")
    print(f"         train : {cfg['train']}")
    print(f"         val   : {cfg['val']}")
    print(f"         test  : {cfg['test']}")
    print(f"         nc    : {cfg['nc']}  names: {cfg['names']}")
    return fixed


#Core training function───

def train(
    model_size: str = "yolov8m.pt",
    epochs: int = 150,
    batch: int = 16,
    imgsz: int = 1280,
    device: str = "cuda:0",
    patience: int = 30,
    workers: int = 8,
    resume: bool = False,
    cache: str | bool = False,
    save_period: int = -1,
    close_mosaic: int = 10,
    name: str = "pothole_v1",
) -> Path:
    """
    Train YOLOv8 on the pothole dataset with A40-optimized settings.

    Default config (yolov8m + imgsz=1280):
      - VRAM usage: ~22 GB on A40 → safe headroom for batch=16
      - Training time: ~3–4 hours for 150 epochs
      - Expected mAP50: 0.75–0.85 on Roboflow pothole dataset

    Returns path to best.pt.
    """
    from ultralytics import YOLO

    #Pre-flight checks───
    if not torch.cuda.is_available() and device.startswith("cuda"):
        print("[ERROR] CUDA not available but device='cuda:0' requested.")
        print("        Run: python scripts/verify_gpu.py")
        sys.exit(1)

    if not DATASET_DIR.exists():
        print(f"[ERROR] Dataset not found: {DATASET_DIR}")
        sys.exit(1)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    #Fix data.yaml paths─
    data_yaml = build_absolute_yaml(ORIGINAL_YAML, FIXED_YAML, DATASET_DIR)

    #Banner 
    print("\n" + "=" * 60)
    print("  Pothole Detection — A40 GPU Training")
    print("=" * 60)
    _print_gpu_banner(device)
    print(f"  Model            : {model_size}")
    print(f"  Device           : {device}")
    print(f"  Batch size       : {batch}")
    print(f"  Epochs           : {epochs}  (patience={patience})")
    print(f"  Image size       : {imgsz}")
    print(f"  AMP              : True (CUDA)")
    print(f"  Workers          : {workers}")
    print(f"  Cache            : {cache}")
    print(f"  Output           : {RUNS_DIR / name}")
    print("=" * 60 + "\n")

    #Clear any residual VRAM before loading model
    torch.cuda.empty_cache()
    print(f"[VRAM pre-load] {_vram_stats()}\n")

    #Load model──────────
    weights_path = PROJECT_ROOT / model_size
    weights = str(weights_path) if weights_path.exists() else model_size
    model = YOLO(weights)

    print(f"[VRAM post-load] {_vram_stats()}\n")

    #Train ─
    start = time.time()

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(RUNS_DIR),
        name=name,
        patience=patience,
        workers=workers,
        resume=resume,
        amp=True,                  #mixed precision — A40 supports bfloat16
        cache=cache,
        close_mosaic=close_mosaic,
        save_period=save_period,
        #Performance
        rect=False,                #rectangular training disabled — mosaic needs square
        cos_lr=True,               #cosine LR schedule
        #Augmentation & hyp
        **A40_AUGMENTATION,
        #Logging
        verbose=True,
        plots=True,
        save=True,
        exist_ok=True,
    )

    elapsed = time.time() - start
    print(f"\n[train] Finished in {elapsed/60:.1f} min")
    print(f"[VRAM post-train] {_vram_stats()}")

    torch.cuda.empty_cache()
    print(f"[VRAM after cache clear] {_vram_stats()}")

    #Find best weights───
    exp_dir = RUNS_DIR / name
    best_pt = exp_dir / "weights" / "best.pt"
    if not best_pt.exists():
        candidates = sorted(exp_dir.rglob("best.pt"))
        if not candidates:
            raise FileNotFoundError(f"best.pt not found under {exp_dir}")
        best_pt = candidates[-1]

    print(f"\n[train] Best weights: {best_pt}")
    return best_pt


#Deploy

def deploy(best_pt: Path) -> Path:
    """Copy best.pt → models/best.pt + models/pothole_yolov8.pt + models/last.pt.

    Returns path to models/pothole_yolov8.pt (used by push step).
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_pt, DEPLOY_PATH)

    #pothole_yolov8.pt is the name model_manager.py and POTHOLE_MODEL_PATH expect
    legacy_path = MODELS_DIR / "pothole_yolov8.pt"
    shutil.copy2(best_pt, legacy_path)

    #last.pt enables --resume on future training runs without re-downloading base weights
    last_src = best_pt.parent / "last.pt"
    last_dst = MODELS_DIR / "last.pt"
    if last_src.exists():
        shutil.copy2(last_src, last_dst)
        print(f"[deploy] last.pt      → {last_dst}  (resume checkpoint)")
    else:
        print(f"[deploy] WARNING: last.pt not found at {last_src} — resume won't be available")

    print(f"[deploy] {best_pt.name} → {DEPLOY_PATH}")
    print(f"[deploy] {best_pt.name} → {legacy_path}  (pipeline auto-pickup)")
    print("\n  Restart the Flask app to load the new weights.")
    return legacy_path


#CLI──

def main():
    parser = argparse.ArgumentParser(
        description="A40-optimized YOLOv8 pothole training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="yolov8m.pt",
                        help="yolov8n.pt (fast baseline) | yolov8m.pt (production) | "
                             "yolov8l.pt (max quality)")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=16,
                        help="16 is safe for yolov8m + imgsz=1280 on A40 48GB (~22GB VRAM). "
                             "Use 64 for yolov8n + imgsz=640 with cache=ram.")
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="1280 improves small-pothole recall by 15-25%% vs 640. "
                             "Use 640 only for fast iteration / yolov8n.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--workers", type=int, default=8,
                        help="DataLoader workers. 8 is optimal for A40; "
                             "reduce to 4 if you see CPU bottleneck in nvidia-smi dmon.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cache", default=False, const="ram", nargs="?",
                        help="--cache ram: decode all images into RAM before training "
                             "(fastest epochs; needs ~8GB RAM for this dataset).")
    parser.add_argument("--name", default="pothole_v1",
                        help="Experiment name under runs/train/")
    parser.add_argument("--no-deploy", action="store_true",
                        help="Skip copying best.pt to models/")
    parser.add_argument("--no-push", action="store_true",
                        help="Skip pushing weights to HuggingFace Hub (requires HF_TOKEN + HF_MODEL_REPO)")
    args = parser.parse_args()

    best_pt = train(
        model_size=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        patience=args.patience,
        workers=args.workers,
        resume=args.resume,
        cache=args.cache,
        name=args.name,
    )

    model_path = None
    if not args.no_deploy:
        model_path = deploy(best_pt)

    if not args.no_push and model_path is not None:
        _proj_root = str(Path(__file__).parent.parent)
        if _proj_root not in sys.path:
            sys.path.insert(0, _proj_root)
        try:
            from training.model_store import push_model
            exp_dir  = RUNS_DIR / args.name
            last_pt  = MODELS_DIR / "last.pt"
            print("\n[push] Uploading weights to HuggingFace Hub...")
            push_model(
                model_path=model_path,
                last_pt=last_pt if last_pt.exists() else None,
                run_dir=exp_dir,
            )
        except Exception as exc:
            print(f"\n[push] WARNING: Hub push failed: {exc}")
            print("[push] Weights are saved locally. Push manually with:")
            print("       python training/model_store.py push")

    print(f"\n[done] Evaluate with:")
    print(f"       python scripts/evaluate.py --weights {best_pt} --imgsz {args.imgsz}")
    print(f"\n[done] Validate dataset first (if you haven't):")
    print(f"       python training/validate_dataset.py")
    print(f"\n[done] Production-readiness targets:")
    print(f"       mAP50 >= 0.75, Precision >= 0.75, Recall >= 0.70")


if __name__ == "__main__":
    main()
