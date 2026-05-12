import argparse
import sys
import os
from pathlib import Path
from collections import Counter, defaultdict


#1. Version checks─────────

def check_environment() -> dict:
    """Validate all required packages and return device info."""
    issues = []

    #Python
    if sys.version_info < (3, 9):
        issues.append(f"Python 3.9+ required, got {sys.version}")

    #torch
    try:
        import torch
        print(f"  torch        : {torch.__version__}")
    except ImportError:
        issues.append("torch not installed — run: pip install torch torchvision")

    #ultralytics
    try:
        import ultralytics
        print(f"  ultralytics  : {ultralytics.__version__}")
    except ImportError:
        issues.append("ultralytics not installed — run: pip install ultralytics>=8.2.0")

    #opencv
    try:
        import cv2
        print(f"  opencv       : {cv2.__version__}")
    except ImportError:
        issues.append("opencv not installed — run: pip install opencv-python-headless")

    #roboflow (optional — only needed at download time)
    try:
        import roboflow
        print(f"  roboflow     : {roboflow.__version__}")
    except ImportError:
        print("  roboflow     : not installed (only needed to re-download dataset)")

    if issues:
        print("\n[ERROR] Missing dependencies:")
        for i in issues:
            print(f"  - {i}")
        sys.exit(1)

    return _detect_device()


def _detect_device() -> dict:
    """Return best available compute device and its properties."""
    import torch

    if torch.cuda.is_available():
        device = "cuda"
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        info = {"device": device, "name": name, "vram_gb": round(vram, 1)}
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        info = {"device": device, "name": "Apple Silicon MPS", "vram_gb": None}
    else:
        device = "cpu"
        import psutil
        ram = psutil.virtual_memory().total / 1e9
        info = {"device": device, "name": "CPU", "vram_gb": None, "ram_gb": round(ram, 1)}

    return info


#2. Dataset validation─────

def validate_dataset(dataset_dir: Path) -> dict:
    """
    Check that a Roboflow YOLOv8 download has the expected structure:

        dataset_dir/
            data.yaml
            train/
                images/  *.jpg|*.png
                labels/  *.txt
            valid/
                images/
                labels/
            test/          ← optional
                images/
                labels/
    """
    import yaml

    print(f"\n[Dataset] Validating: {dataset_dir.resolve()}")

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}\n"
            "Run the Roboflow download snippet first."
        )

    #data.yaml
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found in {dataset_dir}")

    with open(yaml_path) as f:
        data_cfg = yaml.safe_load(f)

    nc = data_cfg.get("nc", 0)
    names = data_cfg.get("names", [])
    print(f"  Classes      : {nc} → {names}")

    splits = {}
    broken = defaultdict(list)

    for split in ("train", "valid", "test"):
        img_dir = dataset_dir / split / "images"
        lbl_dir = dataset_dir / split / "labels"

        if not img_dir.exists():
            if split == "test":
                print(f"  [{split}]       : not present (optional — skipped)")
            else:
                raise FileNotFoundError(f"Missing required split: {img_dir}")
            continue

        images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        labels = sorted(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []

        img_stems = {p.stem for p in images}
        lbl_stems = {p.stem for p in labels}

        no_label = img_stems - lbl_stems          #images without annotations
        no_image = lbl_stems - img_stems          #orphan label files
        empty_labels = [p for p in labels if p.stat().st_size == 0]

        splits[split] = {
            "images": len(images),
            "labels": len(labels),
            "no_label": len(no_label),
            "no_image": len(no_image),
            "empty_labels": len(empty_labels),
        }

        if no_label:
            broken[split].extend([f"no label for: {s}" for s in sorted(no_label)[:5]])
        if empty_labels:
            broken[split].extend([f"empty label: {p.name}" for p in empty_labels[:5]])

        print(
            f"  [{split:5s}]  images={len(images):4d}  labels={len(labels):4d}"
            f"  missing_labels={len(no_label):3d}  empty_labels={len(empty_labels):3d}"
        )

    #Class distribution
    print("\n  Class distribution (training set):")
    dist = _class_distribution(dataset_dir / "train" / "labels", nc)
    for cls_id, count in sorted(dist.items()):
        name = names[cls_id] if cls_id < len(names) else f"class_{cls_id}"
        bar = "#" * min(count // max(1, max(dist.values()) // 30), 30)
        print(f"    [{cls_id}] {name:20s} {count:5d}  {bar}")

    if any(broken.values()):
        print("\n  [WARN] Dataset issues found (first 5 per split):")
        for split, msgs in broken.items():
            for m in msgs:
                print(f"    {split}: {m}")

    return {
        "splits": splits,
        "nc": nc,
        "names": names,
        "yaml_path": str(yaml_path),
        "broken": dict(broken),
    }


def _class_distribution(label_dir: Path, nc: int) -> Counter:
    counter = Counter()
    if not label_dir.exists():
        return counter
    for lbl in label_dir.glob("*.txt"):
        with open(lbl) as f:
            for line in f:
                line = line.strip()
                if line:
                    cls_id = int(line.split()[0])
                    counter[cls_id] += 1
    return counter


#3. Batch size recommendation─

def recommend_batch_size(device_info: dict) -> int:
    """
    Heuristic batch-size recommendation.

    CUDA ≥40 GB  → 64  (A40 / A100 / H100)
    CUDA ≥24 GB  → 32  (RTX 3090 / A5000)
    CUDA ≥16 GB  → 16  (RTX 3080 / A4000)
    CUDA ≥8 GB   → 8   (RTX 3070 / T4)
    MPS          → 8   (unified memory — conservative)
    CPU          → 4
    """
    device = device_info["device"]
    vram = device_info.get("vram_gb") or 0

    if device == "cuda":
        if vram >= 40:
            return 64   #A40 / A100 / H100 (48-80 GB)
        elif vram >= 24:
            return 32   #RTX 3090 / A5000 (24 GB)
        elif vram >= 16:
            return 16   #RTX 3080 / A4000 (16 GB)
        elif vram >= 8:
            return 8    #RTX 3070 / T4 (8 GB)
        else:
            return 4
    elif device == "mps":
        return 8
    else:
        return 4


#CLI entry-point───────────

def main():
    parser = argparse.ArgumentParser(description="Validate training environment and dataset")
    parser.add_argument(
        "--dataset-dir",
        default="pothole-detection-1",
        help="Path to Roboflow YOLOv8 download directory (default: pothole-detection-1)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Pothole Detection — Training Environment Setup")
    print("=" * 60)

    print("\n[1] Package versions:")
    device_info = check_environment()

    print(f"\n[2] Compute device: {device_info['device'].upper()} — {device_info['name']}")
    if device_info.get("vram_gb"):
        print(f"    VRAM: {device_info['vram_gb']} GB")
    if device_info.get("ram_gb"):
        print(f"    RAM : {device_info['ram_gb']} GB")

    print("\n[3] Dataset validation:")
    dataset_dir = Path(args.dataset_dir)
    summary = validate_dataset(dataset_dir)

    batch = recommend_batch_size(device_info)
    print(f"\n[4] Recommended batch size for {device_info['device'].upper()}: {batch}")

    print("\n[5] Ready to train. Run:")
    print(f"    python training/train.py --dataset-dir {args.dataset_dir} --batch {batch}")
    print("=" * 60)


if __name__ == "__main__":
    main()
