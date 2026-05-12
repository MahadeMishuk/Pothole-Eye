import argparse
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

POTHOLE_COLOR = (0, 0, 255)      #Red — high-visibility
OBJECT_COLOR  = (255, 200, 0)    #Cyan-ish — secondary detections
FONT          = cv2.FONT_HERSHEY_SIMPLEX


#Model wrapper────────────

class PotholeInferenceEngine:
    """Thin wrapper around YOLO with clean draw/annotate helpers."""

    def __init__(
        self,
        weights: str | Path,
        conf: float = 0.30,
        iou: float = 0.45,
        device: str = "auto",
        imgsz: int = 640,
    ):
        from ultralytics import YOLO
        import torch

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.model  = YOLO(str(weights))
        self.conf   = conf
        self.iou    = iou
        self.device = device
        self.imgsz  = imgsz
        self.names  = self.model.names  #{0: 'pothole', ...}

        print(f"[engine] Loaded {Path(weights).name} on {device.upper()}")
        print(f"[engine] Classes: {self.names}")

    def predict(self, frame_bgr: np.ndarray) -> List[dict]:
        """
        Run inference on a single BGR frame.

        Returns list of dicts:
            { x1, y1, x2, y2, conf, cls_id, cls_name, is_pothole }
        """
        results = self.model.predict(
            frame_bgr,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False,
            stream=False,
        )

        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0])
                cls_name = self.names.get(cls_id, f"cls_{cls_id}")
                detections.append({
                    "x1": float(x1), "y1": float(y1),
                    "x2": float(x2), "y2": float(y2),
                    "conf": float(box.conf[0]),
                    "cls_id": cls_id,
                    "cls_name": cls_name,
                    #Treat everything as a pothole if the model is single-class
                    "is_pothole": "pothole" in cls_name.lower() or len(self.names) == 1,
                })
        return detections

    def annotate(
        self,
        frame_bgr: np.ndarray,
        detections: List[dict],
        fps: Optional[float] = None,
    ) -> np.ndarray:
        """Draw bounding boxes and labels onto a copy of the frame."""
        out = frame_bgr.copy()
        h, w = out.shape[:2]

        for det in detections:
            x1, y1 = int(det["x1"]), int(det["y1"])
            x2, y2 = int(det["x2"]), int(det["y2"])
            conf    = det["conf"]
            label   = f"{det['cls_name']} {conf:.2f}"
            color   = POTHOLE_COLOR if det["is_pothole"] else OBJECT_COLOR
            thick   = 2 if det["is_pothole"] else 1

            #Box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)

            #Label background
            (tw, th), bl = cv2.getTextSize(label, FONT, 0.5, 1)
            ly = max(y1 - 4, th + 4)
            cv2.rectangle(out, (x1, ly - th - 4), (x1 + tw + 4, ly + bl), color, -1)
            cv2.putText(out, label, (x1 + 2, ly - 2), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        #FPS overlay
        if fps is not None:
            cv2.putText(out, f"FPS: {fps:.1f}", (8, 28), FONT, 0.7,
                        (0, 255, 0), 2, cv2.LINE_AA)

        #Detection count
        n = len(detections)
        cv2.putText(out, f"Detections: {n}", (8, 56), FONT, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)

        return out


#Image inference──────────

def run_image(engine: PotholeInferenceEngine, source: Path, out_dir: Path) -> Path:
    """Run inference on a single image and save annotated result."""
    frame = cv2.imread(str(source))
    if frame is None:
        raise ValueError(f"Could not read image: {source}")

    t0 = time.perf_counter()
    detections = engine.predict(frame)
    ms = (time.perf_counter() - t0) * 1000

    annotated = engine.annotate(frame, detections)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{source.stem}_pred{source.suffix}"
    cv2.imwrite(str(out_path), annotated)

    print(f"[image] {source.name}: {len(detections)} detections in {ms:.1f}ms → {out_path}")
    return out_path


def run_folder(engine: PotholeInferenceEngine, source: Path, out_dir: Path):
    """Run inference on all images in a folder."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in sorted(source.iterdir()) if p.suffix.lower() in exts]
    if not images:
        print(f"[folder] No images found in {source}")
        return
    print(f"[folder] Processing {len(images)} images...")
    for img in images:
        run_image(engine, img, out_dir)


#Video inference──────────

def run_video(
    engine: PotholeInferenceEngine,
    source: Path,
    out_dir: Path,
    show: bool = False,
    frame_skip: int = 1,           #process every Nth frame (1 = all)
) -> Path:
    """
    Run inference on a video file.

    frame_skip=2 halves processing time at cost of temporal resolution.
    """
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {source}")

    fps_in  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{source.stem}_pred.mp4"
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(str(out_path), fourcc, fps_in / frame_skip, (width, height))

    print(f"[video] {source.name}: {total} frames @ {fps_in:.1f}fps")

    frame_idx = 0
    fps_buf: List[float] = []
    last_detections: List[dict] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()

        if frame_idx % frame_skip == 0:
            last_detections = engine.predict(frame)
            elapsed = time.perf_counter() - t0
            fps_buf.append(1.0 / max(elapsed, 1e-6))
            if len(fps_buf) > 30:
                fps_buf.pop(0)
            avg_fps = sum(fps_buf) / len(fps_buf)

            annotated = engine.annotate(frame, last_detections, fps=avg_fps)
            writer.write(annotated)

            if show:
                cv2.imshow("Pothole Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        else:
            #Write un-annotated duplicate to keep video length consistent
            writer.write(frame)

        frame_idx += 1
        if frame_idx % 100 == 0:
            pct = frame_idx / max(total, 1) * 100
            print(f"  {frame_idx}/{total} ({pct:.0f}%) — avg {avg_fps:.1f} FPS")

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()

    print(f"[video] Saved → {out_path}")
    return out_path


#Webcam real-time loop────

def run_webcam(
    engine: PotholeInferenceEngine,
    camera_idx: int = 0,
    target_width: int = 640,
    frame_skip: int = 1,
):
    """
    Real-time webcam inference loop.

    Performance tips already applied:
    - Resize capture to target_width before inference (avoids upscaling overhead)
    - frame_skip=2 on CPU/MPS for smoother display
    - engine.device is resolved at init time

    Press Q to quit.
    """
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_idx}")

    #Set capture resolution (lower resolution = higher FPS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(target_width * 0.75))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  #minimize buffer lag

    print(f"\n[webcam] Starting on camera {camera_idx} — press Q to quit")

    fps_buf: List[float] = []
    last_detections: List[dict] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[webcam] Failed to grab frame")
            break

        t0 = time.perf_counter()

        if frame_idx % frame_skip == 0:
            last_detections = engine.predict(frame)

        elapsed = time.perf_counter() - t0
        fps_buf.append(1.0 / max(elapsed, 1e-6))
        if len(fps_buf) > 30:
            fps_buf.pop(0)
        avg_fps = sum(fps_buf) / len(fps_buf)

        annotated = engine.annotate(frame, last_detections, fps=avg_fps)
        cv2.imshow("Pothole Detection — Webcam (Q to quit)", annotated)

        frame_idx += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[webcam] Stopped.")


#CLI──

def main():
    parser = argparse.ArgumentParser(
        description="Pothole detection inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", default="models/pothole_yolov8.pt",
                        help="Path to trained .pt weights")
    parser.add_argument("--source", required=True,
                        help="Image path | video path | folder path | 'webcam'")
    parser.add_argument("--conf", type=float, default=0.30,
                        help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out-dir", default="outputs/inference",
                        help="Output directory for annotated images/videos")
    parser.add_argument("--show", action="store_true",
                        help="Display frames in a window during video inference")
    parser.add_argument("--frame-skip", type=int, default=1,
                        help="Process every Nth frame (video/webcam)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index for webcam mode")
    args = parser.parse_args()

    engine = PotholeInferenceEngine(
        weights=args.weights,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        imgsz=args.imgsz,
    )

    out_dir = Path(args.out_dir)
    source  = args.source

    if source.lower() == "webcam":
        run_webcam(engine, camera_idx=args.camera, frame_skip=args.frame_skip)

    else:
        src = Path(source)
        if not src.exists():
            print(f"[error] Source not found: {src}")
            raise SystemExit(1)

        if src.is_dir():
            run_folder(engine, src, out_dir)
        elif src.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}:
            run_video(engine, src, out_dir, show=args.show, frame_skip=args.frame_skip)
        else:
            run_image(engine, src, out_dir)


if __name__ == "__main__":
    main()
