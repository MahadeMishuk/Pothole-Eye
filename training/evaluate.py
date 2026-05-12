import argparse
import json
import sys
from pathlib import Path


def _best_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def evaluate(
    weights: str | Path,
    dataset_dir: str | Path,
    split: str = "val",          #val | test
    imgsz: int = 1280,           #must match training imgsz for valid mAP numbers
    batch: int = 16,             #safe for yolov8m + imgsz=1280 on A40
    device: str = "auto",
    conf_threshold: float = 0.001,  #low conf → full PR curve coverage
    iou_threshold: float = 0.6,
    save_dir: str | Path | None = None,
) -> dict:
    """
    Run YOLO validation and return a metrics dict.

    Returns:
        {
          "mAP50": float,
          "mAP50_95": float,
          "precision": float,
          "recall": float,
          "per_class": { class_name: {mAP50, precision, recall} },
          "results_dir": str,
        }
    """
    from ultralytics import YOLO
    import yaml

    weights = Path(weights)
    dataset_dir = Path(dataset_dir)
    data_yaml = dataset_dir / "data.yaml"

    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    if device == "auto":
        device = _best_device()

    if save_dir is None:
        save_dir = weights.parent.parent / "eval"

    model = YOLO(weights)

    print(f"\n[evaluate] Weights  : {weights}")
    print(f"[evaluate] Dataset  : {data_yaml}")
    print(f"[evaluate] Split    : {split}")
    print(f"[evaluate] Device   : {device.upper()}")

    results = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        conf=conf_threshold,
        iou=iou_threshold,
        project=str(Path(save_dir).parent),
        name=Path(save_dir).name,
        plots=True,
        save_json=True,
        verbose=True,
        exist_ok=True,
    )

    #Extract scalar metrics
    box = results.box

    #per-class breakdown
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    class_names = cfg.get("names", [])

    per_class = {}
    if hasattr(box, "ap_class_index") and box.ap_class_index is not None:
        for i, cls_idx in enumerate(box.ap_class_index):
            name = class_names[cls_idx] if cls_idx < len(class_names) else f"cls_{cls_idx}"
            per_class[name] = {
                "mAP50":     float(box.ap50[i])   if hasattr(box, "ap50")   else None,
                "precision": float(box.p[i])       if hasattr(box, "p")      else None,
                "recall":    float(box.r[i])       if hasattr(box, "r")      else None,
            }

    summary = {
        "weights": str(weights),
        "dataset": str(data_yaml),
        "split": split,
        "mAP50":     round(float(box.map50), 4),
        "mAP50_95":  round(float(box.map),   4),
        "precision": round(float(box.mp),    4),
        "recall":    round(float(box.mr),    4),
        "per_class": per_class,
        "results_dir": str(save_dir),
    }

    #Save JSON
    out_json = Path(save_dir) / "eval_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    _print_summary(summary)
    print(f"\n[evaluate] Full results saved → {save_dir}")
    print(f"[evaluate] JSON summary      → {out_json}")

    return summary


def _print_summary(s: dict):
    w = 50
    print("\n" + "=" * w)
    print("  EVALUATION RESULTS")
    print("=" * w)
    print(f"  mAP50      : {s['mAP50']:.4f}  ({s['mAP50']*100:.1f}%)")
    print(f"  mAP50-95   : {s['mAP50_95']:.4f}  ({s['mAP50_95']*100:.1f}%)")
    print(f"  Precision  : {s['precision']:.4f}")
    print(f"  Recall     : {s['recall']:.4f}")

    if s["per_class"]:
        print(f"\n  Per-class breakdown:")
        print(f"  {'Class':<20} {'mAP50':>7} {'Prec':>7} {'Rec':>7}")
        print("  " + "-" * 44)
        for cls, m in s["per_class"].items():
            mAP  = f"{m['mAP50']:.4f}"  if m.get("mAP50")    is not None else "  N/A "
            prec = f"{m['precision']:.4f}" if m.get("precision") is not None else "  N/A "
            rec  = f"{m['recall']:.4f}"  if m.get("recall")   is not None else "  N/A "
            print(f"  {cls:<20} {mAP:>7} {prec:>7} {rec:>7}")

    print("=" * w)

    #Production-readiness diagnostics
    #Thresholds for a dashcam pothole detection system:
    #  mAP50 ≥ 0.75  — reliable detection across lighting/weather conditions
    #  Precision ≥ 0.75  — <25% of alerts are false positives (driver trust)
    #  Recall ≥ 0.70     — <30% of real potholes are missed (safety)
    map50 = s["mAP50"]
    prec  = s["precision"]
    rec   = s["recall"]

    print("\n  Diagnostics (targets: mAP50≥0.75, P≥0.75, R≥0.70):")
    if map50 < 0.50:
        print("  [x] mAP50 < 0.50 — model is severely underfit. Fix:")
        print("      1. Run validate_dataset.py — check class IDs and label quality")
        print("      2. Increase epochs: --epochs 200")
        print("      3. Larger model: --model yolov8m.pt (if still using nano/small)")
        print("      4. Larger resolution: --imgsz 1280 (critical for small potholes)")
    elif map50 < 0.65:
        print("  [~] mAP50 in 0.50-0.65 — below production target.")
        print("      - Check PR curve: low plateau suggests label noise")
        print("      - Try RT-DETR: python training/train_rtdetr.py --stage all")
        print("      - Increase imgsz to 1280 if training used 640")
    elif map50 < 0.75:
        print("  [~] mAP50 in 0.65-0.75 — acceptable, approaching production.")
        print("      - Try hard negative mining: train_rtdetr.py Stage 2")
        print("      - Consider yolov8l.pt for +2-4%% mAP50")
    else:
        print("  [✓] mAP50 ≥ 0.75 — PRODUCTION READY.")

    if prec < 0.65:
        print("  [x] Precision < 0.65 — too many false positives.")
        print("      - Raise POTHOLE_CONFIDENCE in .env.gpu (try 0.35-0.45)")
        print("      - Add hard negatives (shadows, drain covers, tar patches)")
    elif prec < 0.75:
        print("  [~] Precision 0.65-0.75 — marginal. Tune POTHOLE_CONFIDENCE upward.")

    if rec < 0.60:
        print("  [x] Recall < 0.60 — missing too many potholes.")
        print("      - Lower POTHOLE_CONFIDENCE in .env.gpu (try 0.20-0.25)")
        print("      - Check imgsz: training at 640 misses small potholes")
        print("      - More training data / copy_paste augmentation")
    elif rec < 0.70:
        print("  [~] Recall 0.65-0.70 — marginal. Lower POTHOLE_CONFIDENCE slightly.")


#CLI──

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained YOLOv8 pothole model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", required=True,
                        help="Path to best.pt or last.pt")
    parser.add_argument("--dataset-dir", default="Pothole-Detection-1")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="Must match training imgsz for accurate mAP numbers")
    parser.add_argument("--batch", type=int, default=16,
                        help="Eval batch size (16 safe for yolov8m+imgsz=1280 on A40)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--conf", type=float, default=0.001,
                        help="Confidence threshold for PR curve (keep low)")
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--save-dir", default=None,
                        help="Output directory for plots and JSON")
    args = parser.parse_args()

    #Resolve dataset path: try as-given first, then as RunPod absolute path
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        rp_path = Path("/workspace/Pothole-I") / args.dataset_dir
        if rp_path.exists():
            dataset_dir = rp_path
        else:
            print(f"[error] Dataset directory not found: {dataset_dir.resolve()}")
            print(f"        Also tried: {rp_path}")
            sys.exit(1)

    evaluate(
        weights=args.weights,
        dataset_dir=dataset_dir,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
