import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml


#Image extensions recognised by ultralytics
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _check_structure(dataset_dir: Path) -> list[str]:
    """Return list of error strings for missing directories."""
    errors = []
    for split in ("train", "valid", "test"):
        img_dir   = dataset_dir / split / "images"
        label_dir = dataset_dir / split / "labels"
        if not img_dir.exists():
            errors.append(f"Missing: {img_dir}")
        if not label_dir.exists():
            errors.append(f"Missing: {label_dir}")
    return errors


def _check_yaml(dataset_dir: Path) -> tuple[dict | None, list[str]]:
    """Load and validate data.yaml.  Returns (cfg, errors)."""
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        return None, [f"data.yaml not found: {yaml_path}"]

    cfg = _load_yaml(yaml_path)
    errors = []

    nc = cfg.get("nc", None)
    if nc != 1:
        errors.append(f"data.yaml: nc={nc} — must be 1 (single-class pothole dataset)")

    names = cfg.get("names", [])
    if not names:
        errors.append("data.yaml: 'names' field is empty")
    elif names[0].lower() != "pothole":
        errors.append(
            f"data.yaml: names[0]='{names[0]}' — expected 'pothole'. "
            "Rename before training."
        )

    for key in ("train", "val", "test"):
        if key not in cfg:
            errors.append(f"data.yaml: missing key '{key}'")

    return cfg, errors


def _scan_split(
    dataset_dir: Path,
    split: str,
) -> tuple[dict, list[str], list[str]]:
    """
    Scan one split (train/valid/test).

    Returns:
        stats   — dict with counts and class distribution
        errors  — fatal problems (wrong class ID, corrupted labels, missing pairs)
        warnings — non-fatal issues (empty labels, very tiny boxes)
    """
    img_dir   = dataset_dir / split / "images"
    label_dir = dataset_dir / split / "labels"

    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    labels = sorted(label_dir.glob("*.txt"))

    label_stems = {p.stem for p in labels}
    image_stems = {p.stem for p in images}

    errors:   list[str] = []
    warnings: list[str] = []

    #Pairing check────────
    missing_labels  = image_stems - label_stems
    orphan_labels   = label_stems - image_stems
    if missing_labels:
        n = len(missing_labels)
        sample = sorted(missing_labels)[:3]
        errors.append(
            f"[{split}] {n} images have no label file "
            f"(e.g. {', '.join(s + '.txt' for s in sample)})"
        )
    if orphan_labels:
        n = len(orphan_labels)
        errors.append(f"[{split}] {n} label files have no matching image")

    #Label content check──
    class_counts: Counter = Counter()
    total_boxes = 0
    degenerate  = 0
    bad_class   = 0
    format_err  = 0
    empty_labels = 0

    for lp in labels:
        raw = lp.read_text().strip()
        if not raw:
            empty_labels += 1
            continue

        for line_no, line in enumerate(raw.splitlines(), 1):
            parts = line.strip().split()
            if len(parts) != 5:
                format_err += 1
                errors.append(
                    f"[{split}] {lp.name} line {line_no}: expected 5 fields, got {len(parts)}"
                )
                continue

            try:
                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
            except ValueError:
                format_err += 1
                errors.append(f"[{split}] {lp.name} line {line_no}: non-numeric field")
                continue

            total_boxes += 1
            class_counts[cls_id] += 1

            if cls_id != 0:
                bad_class += 1
                errors.append(
                    f"[{split}] {lp.name} line {line_no}: class_id={cls_id} "
                    f"(expected 0 for pothole)"
                )

            #Normalisation bounds check
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
                errors.append(
                    f"[{split}] {lp.name} line {line_no}: "
                    f"cx={cx:.4f}/cy={cy:.4f} out of [0,1]"
                )
            if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                degenerate += 1
                if w <= 0 or h <= 0:
                    errors.append(
                        f"[{split}] {lp.name} line {line_no}: "
                        f"zero-area bbox w={w:.4f} h={h:.4f}"
                    )
                else:
                    warnings.append(
                        f"[{split}] {lp.name} line {line_no}: "
                        f"bbox w={w:.4f} h={h:.4f} out of (0,1]"
                    )

            #Tiny box warning (< 4×4 px on 640-wide image)
            if w * 640 < 4 or h * 640 < 4:
                warnings.append(
                    f"[{split}] {lp.name}: very small box "
                    f"({w*640:.1f}×{h*640:.1f} px at imgsz=640) — "
                    f"consider imgsz=1280"
                )

    stats = {
        "split":        split,
        "images":       len(images),
        "labels":       len(labels),
        "total_boxes":  total_boxes,
        "empty_labels": empty_labels,
        "class_counts": dict(class_counts),
        "bad_class":    bad_class,
        "format_err":   format_err,
    }

    if empty_labels:
        warnings.append(
            f"[{split}] {empty_labels} label files are empty "
            "(images with no potholes — OK if intentional)"
        )

    return stats, errors, warnings


def validate(dataset_dir: Path) -> bool:
    """
    Run all checks.  Prints a formatted report.
    Returns True if the dataset is ready to train, False if errors found.
    """
    W = 64
    print("\n" + "=" * W)
    print("  POTHOLE DATASET VALIDATION")
    print("=" * W)
    print(f"  Dataset: {dataset_dir.resolve()}\n")

    all_errors:   list[str] = []
    all_warnings: list[str] = []

    #1. Directory structure
    struct_errors = _check_structure(dataset_dir)
    all_errors.extend(struct_errors)

    #2. data.yaml─────────
    cfg, yaml_errors = _check_yaml(dataset_dir)
    all_errors.extend(yaml_errors)
    if cfg:
        print(f"  data.yaml")
        print(f"    nc    : {cfg.get('nc')}  (expected 1)")
        print(f"    names : {cfg.get('names')}  (expected ['pothole'])")
        print(f"    train : {cfg.get('train')}")
        print(f"    val   : {cfg.get('val')}")
        print(f"    test  : {cfg.get('test')}\n")

    #3. Per-split scan────
    all_stats: list[dict] = []
    for split in ("train", "valid", "test"):
        img_dir = dataset_dir / split / "images"
        if not img_dir.exists():
            continue
        stats, errs, warns = _scan_split(dataset_dir, split)
        all_stats.append(stats)
        all_errors.extend(errs)
        all_warnings.extend(warns)

    #4. Summary table─────
    print(f"  {'Split':<8} {'Images':>7} {'Labels':>7} {'Boxes':>8} "
          f"{'Empty':>6} {'cls0':>6}")
    print("  " + "-" * 46)
    for s in all_stats:
        cls0 = s["class_counts"].get(0, 0)
        print(
            f"  {s['split']:<8} {s['images']:>7} {s['labels']:>7} "
            f"{s['total_boxes']:>8} {s['empty_labels']:>6} {cls0:>6}"
        )

    total_images = sum(s["images"] for s in all_stats)
    total_boxes  = sum(s["total_boxes"] for s in all_stats)
    print("  " + "-" * 46)
    print(f"  {'TOTAL':<8} {total_images:>7}          {total_boxes:>8}\n")

    #5. Class distribution
    all_classes: Counter = Counter()
    for s in all_stats:
        all_classes.update(s["class_counts"])

    if len(all_classes) == 1 and 0 in all_classes:
        print(f"  [OK] Single class only: class_id=0 (pothole) x {all_classes[0]} boxes")
    else:
        for cls_id, cnt in sorted(all_classes.items()):
            tag = "(POTHOLE)" if cls_id == 0 else "(UNEXPECTED — must be removed)"
            print(f"  class {cls_id}: {cnt} boxes {tag}")

    #6. Small-box advisory
    tiny_warns = [w for w in all_warnings if "very small box" in w]
    if tiny_warns:
        print(f"\n  [ADVISORY] {len(tiny_warns)} very small boxes detected.")
        print("  Recommendation: train with --imgsz 1280 (already the default).")
        print("  Small boxes at imgsz=640 are below 4px and may not be learnable.")

    #7. Results───────────
    other_warns = [w for w in all_warnings if "very small box" not in w]
    print("\n" + "=" * W)

    if all_errors:
        print(f"  ERRORS ({len(all_errors)} found):")
        for e in all_errors:
            print(f"    [x] {e}")
    if other_warns:
        print(f"\n  WARNINGS ({len(other_warns)} found):")
        for w in other_warns:
            print(f"    [!] {w}")
    if not all_errors:
        print("  [PASS] Dataset is valid and ready to train.")
        print()
        print("  Next step:")
        print("    python training/train_gpu.py")
        print()
        print("  Or for a fast baseline first:")
        print("    python training/train_gpu.py --model yolov8n.pt --imgsz 640 --batch 64 --epochs 100 --name pothole_baseline")

    print("=" * W + "\n")
    return len(all_errors) == 0


def main():
    parser = argparse.ArgumentParser(
        description="Validate pothole detection dataset before training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-dir",
        default="Pothole-Detection-1",
        help="Path to dataset root (contains data.yaml, train/, valid/, test/)",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        #Try RunPod absolute path
        rp_path = Path("/workspace/Pothole-I") / args.dataset_dir
        if rp_path.exists():
            dataset_dir = rp_path
        else:
            print(f"[error] Dataset directory not found: {dataset_dir}")
            sys.exit(1)

    ok = validate(dataset_dir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
