import argparse
import shutil
import time
from pathlib import Path


def _best_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"



AUGMENTATION = {
    #Geometry
    "degrees":     12.0,
    "translate":   0.15,
    "scale":       0.7,
    "shear":       3.0,
    "perspective": 0.0008,
    "flipud":      0.0,    
    "fliplr":      0.5,

    #Composition
    "mosaic":     1.0,
    "mixup":      0.15,
    "copy_paste": 0.15,
    "erasing":    0.40,

    #Photometric — night/rain/shadow robustness
    "hsv_h": 0.015,
    "hsv_s": 0.90,
    "hsv_v": 0.60,

    #Label smoothing
    "label_smoothing": 0.05,
}


#Stage 1: Domain adaptation

def stage1_domain_adaptation(
    dataset_dir: str,
    epochs:      int  = 100,
    batch:       int  = 16,
    imgsz:       int  = 640,
    device:      str  = "auto",
    project:     str  = "runs/train",
    name:        str  = "rtdetr_stage1",
) -> Path:
    #RT-DETR in ultralytics prepends its default 'runs/detect/' to a relative
    #project path.  Using an absolute path prevents the double-nesting.
    project = str(Path(project).resolve())
    """
    Train RT-DETR-L from COCO weights on pothole dataset.
    Maximizes recall at this stage — we'll fix FPR in Stage 2.
    """
    from ultralytics import RTDETR

    if device == "auto":
        device = _best_device()

    data_yaml = Path(dataset_dir) / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found in {dataset_dir}")

    print(f"\n{'='*60}")
    print(f"  STAGE 1: Domain Adaptation")
    print(f"  Model:   rtdetr-l.pt (COCO pretrained)")
    print(f"  Dataset: {data_yaml}")
    print(f"  Device:  {device.upper()}")
    print(f"  Epochs:  {epochs}")
    print(f"{'='*60}\n")

    model = RTDETR("rtdetr-l.pt")

    model.train(
        data    = str(data_yaml),
        epochs  = epochs,
        imgsz   = imgsz,
        batch   = batch,
        device  = device,
        project = project,
        name    = name,
        patience = 30,
        amp     = (device == "cuda"),
        lr0     = 1e-3,
        lrf     = 0.01,
        warmup_epochs = 5,
        save_period   = 10,
        plots   = True,
        verbose = True,
        **AUGMENTATION,
    )

    best = Path(project) / name / "weights" / "best.pt"
    if not best.exists():
        candidates = list(Path(project).rglob("best.pt"))
        best = sorted(candidates)[-1] if candidates else None
    if best is None:
        raise FileNotFoundError("best.pt not found after Stage 1 training")

    print(f"\n[Stage 1 complete] Best weights: {best}")
    return best


#Stage 2: Hard negative mining

def stage2_hard_negative_mining(
    stage1_weights: str | Path,
    dataset_dir:    str,
    neg_image_dirs: list,
    epochs:         int  = 50,
    batch:          int  = 16,
    device:         str  = "auto",
    project:        str  = "runs/train",
    name:           str  = "rtdetr_stage2",
    conf_thresh:    float = 0.10,
) -> Path:
    """
    Collect false positives from Stage 1 model on negative images,
    add them to the dataset, and fine-tune at lower LR.

    neg_image_dirs: list of directories containing images with NO potholes
      (road patches with shadows, drain covers, road markings, tar patches)
    """
    project = str(Path(project).resolve())
    from ultralytics import RTDETR
    import os

    if device == "auto":
        device = _best_device()

    model_s1 = RTDETR(str(stage1_weights))

    print(f"\n[Stage 2] Mining hard negatives from {len(neg_image_dirs)} directories …")

    hard_neg_paths = []
    for neg_dir in neg_image_dirs:
        neg_dir = Path(neg_dir)
        if not neg_dir.exists():
            print(f"  WARNING: {neg_dir} not found, skipping")
            continue
        img_files = list(neg_dir.glob("*.jpg")) + list(neg_dir.glob("*.png"))
        for img_path in img_files:
            preds = model_s1.predict(str(img_path), conf=conf_thresh, verbose=False)
            if len(preds[0].boxes) > 0:
                hard_neg_paths.append(img_path)

    print(f"[Stage 2] Found {len(hard_neg_paths)} hard negative images")

    #Add hard negatives to dataset (empty label files)
    hn_img_dir   = Path(dataset_dir) / "train" / "images"
    hn_label_dir = Path(dataset_dir) / "train" / "labels"
    hn_label_dir.mkdir(parents=True, exist_ok=True)

    for img_path in hard_neg_paths:
        dest_img   = hn_img_dir / f"hn_{img_path.name}"
        dest_label = hn_label_dir / f"hn_{img_path.stem}.txt"
        shutil.copy2(img_path, dest_img)
        dest_label.write_text("")   #empty = no potholes in this image

    print(f"[Stage 2] Added {len(hard_neg_paths)} hard negatives to training set")

    #Fine-tune with lower LR, safer augmentations
    model_ft = RTDETR(str(stage1_weights))
    data_yaml = Path(dataset_dir) / "data.yaml"

    #mosaic=0 and copy_paste=0: mixing empty-label hard-negative images inside
    #mosaic batches produces NaN loss under AMP — disable both.
    #amp=False: AMP amplifies NaN from empty-label gradients → train in FP32.
    aug_stage2 = {
        **AUGMENTATION,
        "mosaic":     0.0,   #must be 0 when hard negatives (empty labels) are present
        "copy_paste": 0.0,
        "mixup":      0.05,
    }

    model_ft.train(
        data    = str(data_yaml),
        epochs  = epochs,
        imgsz   = 640,
        batch   = batch,
        device  = device,
        project = project,
        name    = name,
        patience = 20,
        amp     = False,   #AMP off — empty-label hard negatives cause NaN under FP16
        lr0     = 5e-5,    #20× lower than Stage 1 to prevent divergence
        lrf     = 0.1,
        warmup_epochs = 3,
        save_period   = 5,
        plots   = True,
        verbose = True,
        **aug_stage2,
    )

    best = Path(project) / name / "weights" / "best.pt"
    if not best.exists():
        candidates = list(Path(project).rglob("best.pt"))
        best = sorted(candidates)[-1] if candidates else None
    if best is None:
        raise FileNotFoundError("best.pt not found after Stage 2 training")

    print(f"\n[Stage 2 complete] Best weights: {best}")
    return best


#Stage 3: Threshold calibration

def stage3_calibrate(
    weights:     str | Path,
    dataset_dir: str,
    device:      str = "auto",
    imgsz:       int = 1280,
) -> float:
    """
    Sweep confidence thresholds on validation set.
    Find the threshold that maximises F1 score.
    Returns optimal confidence threshold.

    Works with any Ultralytics model (YOLOv8, RT-DETR, YOLOv9).
    Uses the generic YOLO loader so it handles both architectures correctly.
    """
    from ultralytics import YOLO
    import numpy as np

    if device == "auto":
        device = _best_device()

    #YOLO() is the universal Ultralytics loader — it handles YOLOv8 AND RT-DETR
    #weights correctly by reading the architecture from the checkpoint.
    #RTDETR() only works for RT-DETR checkpoints and silently zeros predictions
    #when fed a YOLOv8 checkpoint, producing F1=0.000 at every threshold.
    model    = YOLO(str(weights))
    data_yaml = Path(dataset_dir) / "data.yaml"
    if not data_yaml.exists():
        #Try RunPod absolute path
        rp_path = Path("/workspace/Pothole-I") / dataset_dir / "data.yaml"
        if rp_path.exists():
            data_yaml = rp_path
        else:
            raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    print(f"\n[Stage 3] Calibrating confidence threshold …")
    print(f"  Weights: {weights}")
    print(f"  Dataset: {data_yaml}")

    best_f1, best_conf = 0.0, 0.25

    thresholds = [round(x, 2) for x in [0.10, 0.15, 0.20, 0.25, 0.30,
                                          0.35, 0.40, 0.45, 0.50, 0.55, 0.60]]

    for conf_thresh in thresholds:
        metrics = model.val(
            data    = str(data_yaml),
            conf    = conf_thresh,
            iou     = 0.50,
            imgsz   = imgsz,
            device  = device,
            verbose = False,
            plots   = False,
        )
        p = float(metrics.box.mp)
        r = float(metrics.box.mr)
        f1 = 2 * p * r / (p + r + 1e-8)
        print(f"  conf={conf_thresh:.2f}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}")
        if f1 > best_f1:
            best_f1, best_conf = f1, conf_thresh

    print(f"\n[Stage 3] Optimal conf: {best_conf:.2f}  (F1={best_f1:.3f})")
    print(f"  → Set POTHOLE_CONFIDENCE={best_conf} in .env.gpu")
    return best_conf


#Deploy

def deploy_model(weights: Path, project_root: Path | None = None) -> Path:
    if project_root is None:
        project_root = Path(__file__).parent.parent
    dest = project_root / "models" / "pothole_rtdetr.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights, dest)
    print(f"\n[deploy] Model copied → {dest}")
    print("  Set MODEL_PATH=/app/models/pothole_rtdetr.pt in .env.gpu")
    print("  Set DETECTION_BACKEND=rtdetr in .env.gpu")
    return dest


#CLI──

def main():
    parser = argparse.ArgumentParser(
        description="3-stage RT-DETR pothole training pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-dir",  default="Pothole-Detection-1")
    parser.add_argument("--stage",        default="all",
                        choices=["1", "2", "3", "all", "calibrate"])
    parser.add_argument("--epochs-s1",   type=int, default=100)
    parser.add_argument("--epochs-s2",   type=int, default=50)
    parser.add_argument("--batch",        type=int, default=16)
    parser.add_argument("--device",       default="auto")
    parser.add_argument("--project",      default="runs/train")
    parser.add_argument("--neg-dirs",     nargs="*", default=[],
                        help="Directories of hard-negative images (no potholes)")
    parser.add_argument("--weights",      default=None,
                        help="Existing weights to skip Stage 1/2")
    parser.add_argument("--no-deploy",    action="store_true")
    parser.add_argument("--imgsz",        type=int, default=1280,
                        help="Must match the imgsz used during training")
    args = parser.parse_args()

    stage1_w = Path(args.weights) if args.weights else None
    stage2_w = None

    run_s1 = args.stage in ("1", "all")
    run_s2 = args.stage in ("2", "all")
    run_s3 = args.stage in ("3", "all", "calibrate")

    if run_s1:
        stage1_w = stage1_domain_adaptation(
            dataset_dir = args.dataset_dir,
            epochs      = args.epochs_s1,
            batch       = args.batch,
            device      = args.device,
            project     = args.project,
        )

    if run_s2:
        if stage1_w is None:
            raise ValueError("--weights required for Stage 2")
        stage2_w = stage2_hard_negative_mining(
            stage1_weights = stage1_w,
            dataset_dir    = args.dataset_dir,
            neg_image_dirs = args.neg_dirs,
            epochs         = args.epochs_s2,
            batch          = args.batch,
            device         = args.device,
            project        = args.project,
        )

    final_weights = stage2_w or stage1_w or (Path(args.weights) if args.weights else None)

    if run_s3 and final_weights:
        opt_conf = stage3_calibrate(
            weights     = final_weights,
            dataset_dir = args.dataset_dir,
            device      = args.device,
            imgsz       = args.imgsz,
        )
        print(f"\nAdd to .env.gpu:  POTHOLE_CONFIDENCE={opt_conf:.2f}")

    if not args.no_deploy and final_weights:
        deploy_model(final_weights)


if __name__ == "__main__":
    main()
