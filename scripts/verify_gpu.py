"""
scripts/verify_gpu.py — Pre-training GPU readiness check

Run this before training to confirm the A40 environment is correctly configured.

Usage:
    python scripts/verify_gpu.py

Checks:
  - CUDA availability and device properties
  - VRAM free/total
  - PyTorch + Ultralytics versions
  - Dataset path and yaml validity
  - A small YOLO forward pass on GPU (smoke test)
  - nvidia-smi snapshot
"""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path("/workspace/Pothole-I")
DATASET_DIR  = PROJECT_ROOT / "Pothole-Detection-1"
DATA_YAML    = DATASET_DIR / "data.yaml"


def _section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check_cuda():
    _section("1 · CUDA / GPU")
    import torch

    if not torch.cuda.is_available():
        print("  [FAIL] CUDA not available.")
        print("         Check: nvidia-smi && nvcc --version")
        sys.exit(1)

    n = torch.cuda.device_count()
    print(f"  CUDA available    : YES")
    print(f"  Device count      : {n}")

    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        print(f"\n  [GPU {i}]  {props.name}")
        print(f"    VRAM total     : {total/1e9:.1f} GB")
        print(f"    VRAM free      : {free/1e9:.1f} GB")
        print(f"    VRAM used      : {(total-free)/1e9:.1f} GB")
        print(f"    CUDA capability: {props.major}.{props.minor}")
        print(f"    SM count       : {props.multi_processor_count}")

    print(f"\n  PyTorch version   : {torch.__version__}")
    print(f"  CUDA build version: {torch.version.cuda}")

    #Recommend batch size
    free_gb = torch.cuda.mem_get_info(0)[0] / 1e9
    if free_gb >= 40:
        rec_batch = 64
    elif free_gb >= 20:
        rec_batch = 32
    elif free_gb >= 10:
        rec_batch = 16
    else:
        rec_batch = 8
    print(f"\n  Recommended batch : {rec_batch}  (based on {free_gb:.1f} GB free)")
    return rec_batch


def check_packages():
    _section("2 · Package Versions")
    import torch
    packages = {
        "torch":         torch.__version__,
        "ultralytics":   None,
        "cv2":           None,
        "numpy":         None,
    }
    try:
        import ultralytics; packages["ultralytics"] = ultralytics.__version__
    except ImportError:
        print("  [FAIL] ultralytics not installed — run: pip install ultralytics")
        sys.exit(1)

    try:
        import cv2; packages["cv2"] = cv2.__version__
    except ImportError:
        print("  [WARN] opencv not found")

    try:
        import numpy; packages["numpy"] = numpy.__version__
    except ImportError:
        print("  [WARN] numpy not found")

    for name, ver in packages.items():
        status = f"{ver}" if ver else "[not installed]"
        print(f"  {name:<16}: {status}")


def check_dataset():
    _section("3 · Dataset")

    if not DATASET_DIR.exists():
        print(f"  [FAIL] Dataset directory not found: {DATASET_DIR}")
        sys.exit(1)

    if not DATA_YAML.exists():
        print(f"  [FAIL] data.yaml not found: {DATA_YAML}")
        sys.exit(1)

    import yaml
    with open(DATA_YAML) as f:
        cfg = yaml.safe_load(f)

    print(f"  data.yaml         : {DATA_YAML}")
    print(f"  Classes           : {cfg.get('nc')} → {cfg.get('names')}")

    #Note: Roboflow data.yaml uses relative paths ../train/images
    #train_gpu.py patches these to absolute paths automatically.
    #Verify the actual split directories exist.
    for split, subdir in [("train", "train"), ("valid", "valid"), ("test", "test")]:
        img_dir = DATASET_DIR / subdir / "images"
        lbl_dir = DATASET_DIR / subdir / "labels"
        if img_dir.exists():
            n_imgs = len(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
            n_lbls = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
            print(f"  [{split:5s}] images={n_imgs:4d}  labels={n_lbls:4d}  → {img_dir}")
        else:
            marker = "[WARN] missing (optional)" if split == "test" else "[FAIL] MISSING"
            print(f"  [{split:5s}] {marker}: {img_dir}")
            if split != "test":
                sys.exit(1)


def smoke_test_yolo():
    _section("4 · YOLO GPU Smoke Test")
    import torch
    import numpy as np
    from ultralytics import YOLO

    print("  Loading yolov8n.pt on cuda:0 ...")
    weights = PROJECT_ROOT / "yolov8n.pt"
    if not weights.exists():
        print("  [WARN] yolov8n.pt not found locally — will be downloaded at training time.")
        weights = "yolov8n.pt"

    model = YOLO(str(weights))
    model.to("cuda:0")

    dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    import time
    t0 = time.perf_counter()
    _ = model.predict(dummy, device="cuda:0", verbose=False, imgsz=640)
    ms = (time.perf_counter() - t0) * 1000

    allocated = torch.cuda.memory_allocated(0) / 1e6
    print(f"  Forward pass      : {ms:.1f} ms")
    print(f"  VRAM allocated    : {allocated:.0f} MB")
    print(f"  [OK] YOLO runs on CUDA.")

    #Flush
    torch.cuda.empty_cache()
    print(f"  VRAM after cache clear: {torch.cuda.memory_allocated(0)/1e6:.0f} MB")


def nvidia_smi():
    _section("5 · nvidia-smi")
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,driver_version,memory.total,memory.free,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 6:
                    name, driver, mem_total, mem_free, util, temp = parts
                    print(f"  GPU               : {name}")
                    print(f"  Driver            : {driver}")
                    print(f"  VRAM total        : {mem_total} MB")
                    print(f"  VRAM free         : {mem_free} MB")
                    print(f"  GPU utilization   : {util}%")
                    print(f"  Temperature       : {temp}°C")
        else:
            print(f"  [WARN] nvidia-smi error: {result.stderr}")
    except FileNotFoundError:
        print("  [WARN] nvidia-smi not in PATH")
    except subprocess.TimeoutExpired:
        print("  [WARN] nvidia-smi timed out")


def main():
    print("\n" + "=" * 60)
    print("  Pothole Detection — GPU Readiness Check")
    print("=" * 60)

    rec_batch = check_cuda()
    check_packages()
    check_dataset()
    smoke_test_yolo()
    nvidia_smi()

    print("\n" + "=" * 60)
    print("  ALL CHECKS PASSED — Ready to train.")
    print(f"  Recommended command:")
    print(f"    python training/train_gpu.py --batch {rec_batch}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
