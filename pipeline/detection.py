import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


def _resolve_device() -> str:
    env = os.environ.get("INFERENCE_DEVICE", "").strip()
    if env:
        return env
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


INFERENCE_DEVICE = _resolve_device()

RISK_COLORS = {
    "near":   (0, 0, 255),
    "medium": (0, 165, 255),
    "far":    (0, 255, 0),
    "object": (255, 200, 0),
}

SEVERITY_COLORS = {
    "L4_CRITICAL": (0, 0, 255),
    "L3_SEVERE":   (0, 80, 255),
    "L2_MODERATE": (0, 165, 255),
    "L1_COSMETIC": (0, 255, 180),
}


@dataclass
class Detection:
    """Single detected entity in a frame. Extended with fusion + severity fields."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_name: str
    is_pothole: bool = False

    #Distance / risk (set by depth module)
    distance_m:  Optional[float] = None
    risk_level:  Optional[str]   = None
    track_id:    Optional[int]   = None

    #Fusion fields (set by PerceptionFusion)
    depth_m:        float = 0.0
    road_ratio:     float = 1.0
    severity_score: float = 0.0
    severity_level: str   = "L1_COSMETIC"

    #Instance segmentation mask (set by RT-DETR-seg or YOLOv8-seg)
    mask: Optional[np.ndarray] = field(default=None, repr=False)

    #State machine state (set by PotholeStateMachine)
    track_state: str = "tentative"

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)

    @property
    def bbox_xyxy(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    @property
    def center(self) -> Tuple[int, int]:
        return int((self.x1 + self.x2) / 2), int((self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def as_xyxy_array(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)


class PotholeDetector:
    """
    Pothole-only detector. No COCO classes, no image-processing fallback.

    Priority:
      RT-DETR-L (trained on pothole dataset) → Custom pothole YOLO
    """

    def __init__(self, model_manager, config):
        self.mm     = model_manager
        self.config = config

    def detect(
        self,
        frame_bgr: np.ndarray,
        road_mask: Optional[np.ndarray] = None,
    ) -> List[Detection]:
        detections: List[Detection] = []

        #Strategy 1: RT-DETR-L (primary — transformer, anchor-free)
        if getattr(self.mm, "rtdetr", None) is not None:
            detections.extend(self._run_rtdetr(frame_bgr))
        #Strategy 2: Custom trained single-class pothole YOLO
        elif getattr(self.mm, "pothole_yolo", None) is not None:
            detections.extend(self._run_custom_pothole_yolo(frame_bgr))

        #Gate by road mask: discard potholes whose centre falls outside the road
        if road_mask is not None and detections:
            detections = self._filter_by_road_mask(detections, road_mask)

        #NMS on pothole detections only
        return _nms(detections, iou_threshold=self.config.NMS_THRESHOLD)

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """
        Batch inference: N frames → one GPU forward pass.
        Returns one detection list per frame.
        """
        if not frames:
            return []

        if getattr(self.mm, "rtdetr", None) is not None:
            return self._run_rtdetr_batch(frames)

        #Fallback: sequential per-frame using custom pothole YOLO
        return [self.detect(f) for f in frames]

    #RT-DETR 

    def _run_rtdetr(self, frame_bgr: np.ndarray) -> List[Detection]:
        try:
            results = self.mm.rtdetr(
                frame_bgr,
                conf    = self.config.POTHOLE_CONFIDENCE,
                imgsz   = self.config.MAX_FRAME_WIDTH,
                device  = INFERENCE_DEVICE,
                verbose = False,
                half    = ("cuda" in INFERENCE_DEVICE),
            )
            return self._parse_yolo_results(results, source="rtdetr")
        except Exception as e:
            logger.warning("RT-DETR inference error: %s", e)
            return []

    def _run_rtdetr_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        try:
            results = self.mm.rtdetr(
                frames,
                conf    = self.config.POTHOLE_CONFIDENCE,
                imgsz   = self.config.MAX_FRAME_WIDTH,
                device  = INFERENCE_DEVICE,
                verbose = False,
                half    = ("cuda" in INFERENCE_DEVICE),
                stream  = False,
            )
            output = []
            for r in results:
                output.append(self._parse_single_result(r, source="rtdetr"))
            #Pad if needed
            while len(output) < len(frames):
                output.append([])
            return output
        except Exception as e:
            logger.warning("RT-DETR batch error: %s — falling back to sequential", e)
            return [self.detect(f) for f in frames]

    #Custom pothole YOLO──

    def _run_custom_pothole_yolo(self, frame_bgr: np.ndarray) -> List[Detection]:
        try:
            results = self.mm.pothole_yolo(
                frame_bgr,
                conf    = self.config.POTHOLE_CONFIDENCE,
                device  = INFERENCE_DEVICE,
                verbose = False,
                half    = ("cuda" in INFERENCE_DEVICE),
            )
            return self._parse_yolo_results(results, source="pothole_yolo")
        except Exception as e:
            logger.warning("Custom pothole YOLO error: %s", e)
            return []

    #Result parsing────────

    def _parse_yolo_results(self, results, source: str) -> List[Detection]:
        detections: List[Detection] = []
        for r in results:
            detections.extend(self._parse_single_result(r, source))
        if "cuda" in INFERENCE_DEVICE:
            torch.cuda.empty_cache()
        return detections

    def _parse_single_result(self, r, source: str) -> List[Detection]:
        detections: List[Detection] = []
        if r.boxes is None:
            return detections

        for i, box in enumerate(r.boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf   = float(box.conf[0])
            cls_id = int(box.cls[0])

            #Both models are single-class (nc=1, class 0 = pothole).
            #is_pothole is always True; the flag is kept for downstream
            #compatibility with depth/fusion modules.
            cls_name = r.names.get(cls_id, "pothole")
            is_pothole = True  #single-class pothole model — every detection is a pothole

            #Extract instance mask if available (segmentation model)
            mask = None
            if hasattr(r, "masks") and r.masks is not None:
                try:
                    masks = r.masks.data
                    if i < masks.shape[0]:
                        m = masks[i].cpu().numpy()
                        mask = (m > 0.5).astype(np.uint8) * 255
                except Exception:
                    pass

            detections.append(Detection(
                x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                confidence = conf,
                class_name = cls_name,
                is_pothole = is_pothole,
                mask       = mask,
            ))

        return detections

    #Helpers 

    @staticmethod
    def _filter_by_road_mask(
        detections: List[Detection], road_mask: np.ndarray
    ) -> List[Detection]:
        h, w = road_mask.shape[:2]
        filtered = []
        for det in detections:
            if not det.is_pothole:
                filtered.append(det)
                continue
            cx, cy = det.center
            cx = max(0, min(cx, w - 1))
            cy = max(0, min(cy, h - 1))
            if road_mask[cy, cx] == 255:
                filtered.append(det)
        return filtered


#Standalone helpers────────

def _iou(a: Detection, b: Detection) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def _nms(detections: List[Detection], iou_threshold: float = 0.45) -> List[Detection]:
    if not detections:
        return []
    detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept, suppressed = [], set()
    for i, det in enumerate(detections):
        if i in suppressed:
            continue
        kept.append(det)
        for j in range(i + 1, len(detections)):
            if j not in suppressed and _iou(det, detections[j]) > iou_threshold:
                suppressed.add(j)
    return kept


def draw_detections(
    frame_bgr: np.ndarray,
    detections: List[Detection],
    show_distance: bool = True,
    show_severity: bool = True,
) -> np.ndarray:
    annotated = frame_bgr.copy()

    for det in detections:
        x1, y1, x2, y2 = det.bbox

        if det.is_pothole:
            #Colour by severity level if available, otherwise by risk
            if show_severity and det.severity_score > 0:
                colour = SEVERITY_COLORS.get(det.severity_level, RISK_COLORS["far"])
            else:
                colour = RISK_COLORS.get(det.risk_level or "far", RISK_COLORS["far"])
            thickness = 2
        else:
            colour    = RISK_COLORS["object"]
            thickness = 1

        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, thickness)

        #Draw instance mask overlay (semi-transparent fill)
        if det.is_pothole and det.mask is not None:
            try:
                mask_resized = cv2.resize(det.mask, (x2 - x1, y2 - y1),
                                          interpolation=cv2.INTER_NEAREST)
                overlay = annotated[y1:y2, x1:x2].copy()
                overlay[mask_resized > 0] = (
                    0.5 * np.array(colour) + 0.5 * overlay[mask_resized > 0]
                ).astype(np.uint8)
                annotated[y1:y2, x1:x2] = overlay
            except Exception:
                pass

        #Label
        label_parts = [f"{det.class_name} {det.confidence:.2f}"]
        if show_distance and det.is_pothole:
            dm = det.depth_m or det.distance_m
            if dm:
                label_parts.append(f"{dm:.1f}m")
        if show_severity and det.is_pothole and det.severity_score > 0:
            label_parts.append(det.severity_level)
        if det.track_id is not None:
            label_parts.insert(0, f"#{det.track_id}")

        emoji = ""
        if det.is_pothole:
            emoji = {"near": "!", "medium": "~", "far": "*"}.get(det.risk_level or "far", "*")
            label_parts.insert(0, f"[{emoji}]")

        label = " ".join(label_parts)
        (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        label_y = max(y1 - 4, lh + 4)
        cv2.rectangle(annotated,
                      (x1, label_y - lh - 4),
                      (x1 + lw + 4, label_y + baseline),
                      colour, -1)
        cv2.putText(annotated, label, (x1 + 2, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    return annotated
