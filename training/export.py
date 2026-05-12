import argparse
import shutil
import sys
from pathlib import Path


def _best_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def export_onnx(
    weights: Path,
    imgsz: int = 640,
    opset: int = 17,
    dynamic: bool = True,
    simplify: bool = True,
    half: bool = False,          #FP16 ONNX — only useful on GPU-based runtimes
    device: str = "auto",
) -> Path:
    """
    Export to ONNX.

    dynamic=True is recommended for production: it allows the ONNX model to
    accept any batch size and resolution without re-export.
    simplify=True runs onnx-simplifier to fold constants and remove dead nodes.
    """
    from ultralytics import YOLO

    if device == "auto":
        device = _best_device()

    #Half-precision ONNX only makes sense on CUDA
    if half and device != "cuda":
        print("[export/onnx] half=True is only effective on CUDA — disabling")
        half = False

    model = YOLO(str(weights))
    print(f"\n[export/onnx] Exporting {weights.name} → ONNX")
    print(f"  opset={opset}  dynamic={dynamic}  simplify={simplify}  half={half}")

    #Ultralytics writes the ONNX file next to the weights
    export_path = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        dynamic=dynamic,
        simplify=simplify,
        half=half,
        device=device,
    )

    src = Path(export_path)
    dest = weights.parent / weights.with_suffix(".onnx").name
    if src != dest:
        shutil.copy2(src, dest)

    print(f"[export/onnx] Saved → {dest}")
    _verify_onnx(dest)
    return dest


def _verify_onnx(path: Path):
    """Quick sanity check using onnx.checker (if installed)."""
    try:
        import onnx
        model = onnx.load(str(path))
        onnx.checker.check_model(model)
        print(f"[export/onnx] ✓ ONNX model verified (opset {model.opset_import[0].version})")
        size_mb = path.stat().st_size / 1e6
        print(f"[export/onnx]   File size: {size_mb:.1f} MB")
    except ImportError:
        print("[export/onnx] onnx package not installed — skipping verification")
        print("              Install with: pip install onnx onnxsim")
    except Exception as e:
        print(f"[export/onnx] Verification failed: {e}")


def export_torchscript(
    weights: Path,
    imgsz: int = 640,
    optimize: bool = False,      #torch.jit.optimize_for_inference — CPU only
) -> Path:
    """
    Export to TorchScript (.torchscript).

    TorchScript export is NOT supported on MPS — ultralytics will run it on CPU.
    optimize=True applies torch.jit.optimize_for_inference, useful for CPU deployment.
    """
    from ultralytics import YOLO
    import torch

    device = _best_device()

    #TorchScript export must happen on CPU (ultralytics limitation)
    export_device = "cpu"
    if device == "mps":
        print("[export/ts] TorchScript export is not MPS-compatible — using CPU")

    model = YOLO(str(weights))
    print(f"\n[export/ts] Exporting {weights.name} → TorchScript")
    print(f"  device={export_device}  optimize={optimize}")

    try:
        export_path = model.export(
            format="torchscript",
            imgsz=imgsz,
            optimize=optimize,
            device=export_device,
        )
        src = Path(export_path)
        dest = weights.parent / weights.with_suffix(".torchscript").name
        if src != dest:
            shutil.copy2(src, dest)

        size_mb = dest.stat().st_size / 1e6
        print(f"[export/ts] Saved → {dest}  ({size_mb:.1f} MB)")
        return dest

    except Exception as e:
        print(f"[export/ts] Failed: {e}")
        print("[export/ts] TorchScript export requires PyTorch and may fail for")
        print("            certain model configurations. Use ONNX instead.")
        return None


def print_usage_examples(onnx_path: Path | None, ts_path: Path | None):
    print("\n" + "=" * 60)
    print("  HOW TO USE EXPORTED MODELS")
    print("=" * 60)

    if onnx_path:
        print(f"""
  ONNX Runtime (Python):
    import onnxruntime as ort
    import numpy as np
    sess = ort.InferenceSession("{onnx_path}")
    #input: float32 NCHW, normalised [0,1]
    inp = np.random.randn(1, 3, 640, 640).astype(np.float32)
    out = sess.run(None, {{sess.get_inputs()[0].name: inp}})

  OpenCV DNN:
    net = cv2.dnn.readNetFromONNX("{onnx_path}")
    blob = cv2.dnn.blobFromImage(frame, 1/255, (640,640), swapRB=True)
    net.setInput(blob)
    preds = net.forward()

  Ultralytics (ONNX):
    from ultralytics import YOLO
    model = YOLO("{onnx_path}")
    results = model.predict(image)
""")

    if ts_path:
        print(f"""
  TorchScript:
    import torch
    model = torch.jit.load("{ts_path}")
    model.eval()
    #input: float32 NCHW tensor, normalised [0,1]
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        output = model(x)
""")
    print("=" * 60)


#CLI──

def main():
    parser = argparse.ArgumentParser(
        description="Export YOLOv8 pothole model to ONNX and TorchScript",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", default="models/pothole_yolov8.pt",
                        help="Path to trained .pt weights")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version (17 = ONNX Runtime ≥1.14)")
    parser.add_argument("--no-dynamic", dest="dynamic", action="store_false",
                        help="Disable dynamic axes in ONNX (fixed batch=1, imgsz=640)")
    parser.add_argument("--no-simplify", dest="simplify", action="store_false",
                        help="Skip onnx-simplifier pass")
    parser.add_argument("--half", action="store_true",
                        help="Export FP16 ONNX (CUDA only)")
    parser.add_argument("--no-torchscript", dest="torchscript", action="store_false",
                        help="Skip TorchScript export")
    parser.add_argument("--optimize-ts", action="store_true",
                        help="Apply torch.jit.optimize_for_inference (CPU TorchScript)")
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        print(f"[error] Weights not found: {weights}")
        raise SystemExit(1)

    onnx_path = export_onnx(
        weights=weights,
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=args.dynamic,
        simplify=args.simplify,
        half=args.half,
    )

    ts_path = None
    if args.torchscript:
        ts_path = export_torchscript(
            weights=weights,
            imgsz=args.imgsz,
            optimize=args.optimize_ts,
        )

    print_usage_examples(onnx_path, ts_path)


if __name__ == "__main__":
    main()
