"""
Perception Fusion Module

Combines detection, segmentation, and metric depth into a unified per-instance
representation. Three functions:

  1. Road-mask gating     — suppress detections not on road surface
  2. Depth-weighted conf  — discount far/uncertain detections
  3. Severity estimation  — physically-grounded score in [0, 1]

The fusion output is the input to the temporal tracking layer.
]]
import logging
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

#Normalized focal length assumption at 640px frame width.
#Used only for real-area conversion; depth itself is metric from DepthAnythingV2.
_FOCAL_NORM_PX = 500.0
_MAX_RELIABLE_DEPTH_M = 40.0


class PerceptionFusion:
    """
    Fuses RT-DETR detections with segmentation mask and metric depth map.

    All fusion is done in-place on Detection objects, adding/updating:
      - depth_m        (float, metres)
      - severity_score (float, 0-1)
      - severity_level (str, L1-L4)
      - road_ratio     (float, 0-1, fraction of bbox on road)

    Detections that do not pass the road-mask gate are removed from the list.
    """

    def __init__(self, road_ratio_min: float = 0.35):
        self._road_ratio_min = road_ratio_min

    def fuse(
        self,
        detections: list,
        seg_mask:   Optional[np.ndarray],
        depth_map:  Optional[np.ndarray],
        frame_bgr:  np.ndarray,
    ) -> list:
        """
        Args:
            detections: List of Detection objects from the detector
            seg_mask:   Binary H×W road mask (0/255) or None
            depth_map:  Float32 H×W metric depth in metres or None
            frame_bgr:  Original BGR frame (for geometry)

        Returns:
            Filtered + annotated list of Detection objects.
        """
        if not detections:
            return []

        h, w = frame_bgr.shape[:2]
        kept = []

        for det in detections:
            x1 = max(0, int(det.x1))
            y1 = max(0, int(det.y1))
            x2 = min(w - 1, int(det.x2))
            y2 = min(h - 1, int(det.y2))

            if x2 <= x1 or y2 <= y1:
                continue

            #Gate 1: Road mask coverage
            road_ratio = self._road_ratio(seg_mask, x1, y1, x2, y2)
            det.road_ratio = road_ratio

            if det.is_pothole and seg_mask is not None:
                if road_ratio < self._road_ratio_min:
                    continue  #hard reject — not on road surface

            #Gate 2: Metric depth extraction
            if depth_map is not None:
                depth_roi = depth_map[y1:y2, x1:x2]
                valid = depth_roi[(depth_roi > 0.1) & (depth_roi < 200.0)]
                if valid.size > 0:
                    det.depth_m = float(np.median(valid))
                    #Soft-discount far potholes (hard to characterise at distance)
                    if det.is_pothole and det.depth_m > _MAX_RELIABLE_DEPTH_M:
                        det.confidence *= 0.6
                else:
                    det.depth_m = det.distance_m or 10.0
            else:
                det.depth_m = det.distance_m or 10.0

            #Severity estimation (pothole-only) ─────────────────────────
            if det.is_pothole:
                det.severity_score, det.severity_level = self._severity(
                    depth_map  = depth_map,
                    depth_m    = det.depth_m,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence = det.confidence,
                    mask       = getattr(det, "mask", None),
                )

            kept.append(det)

        return kept

    #Helpers ─

    @staticmethod
    def _road_ratio(
        seg_mask: Optional[np.ndarray],
        x1: int, y1: int, x2: int, y2: int,
    ) -> float:
        if seg_mask is None:
            return 1.0  #no mask → don't gate
        roi = seg_mask[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0
        return float(np.count_nonzero(roi)) / roi.size

    @staticmethod
    def _severity(
        depth_map:  Optional[np.ndarray],
        depth_m:    float,
        x1: int, y1: int, x2: int, y2: int,
        confidence: float,
        mask:       Optional[np.ndarray],
    ):
        """
        Physically-grounded severity score.

        Components:
          depth_delta  — depth variance within bbox (proxy for crater depth)
          area_m2      — real-world area via pinhole projection
          shape_score  — mask compactness (irregular = structurally worse)
          confidence   — detector certainty
        """
        #1. Depth variance inside bbox (higher std → deeper crater)
        depth_delta = 0.0
        if depth_map is not None:
            roi = depth_map[y1:y2, x1:x2]
            valid = roi[(roi > 0.1) & (roi < 200.0)]
            if valid.size > 10:
                depth_delta = min(float(np.std(valid)) / 0.5, 1.0)

        #2. Real-world area via pinhole projection
        bbox_area_px = (x2 - x1) * (y2 - y1)
        pixel_size_m = depth_m / _FOCAL_NORM_PX
        real_area_m2 = bbox_area_px * (pixel_size_m ** 2)
        area_score   = min(real_area_m2 / 1.0, 1.0)

        #3. Shape compactness from instance mask (if available)
        shape_score = 0.4  #default: unknown
        if mask is not None:
            try:
                m = mask.astype(np.uint8)
                area = float(np.count_nonzero(m))
                contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours and area > 0:
                    perimeter = sum(cv2.arcLength(c, True) for c in contours)
                    if perimeter > 0:
                        circularity = (4 * np.pi * area) / (perimeter ** 2)
                        shape_score = float(np.clip(1.0 - circularity, 0.0, 1.0))
            except Exception:
                pass

        #4. Weighted combination
        score = (
            0.40 * depth_delta
            + 0.35 * area_score
            + 0.15 * shape_score
            + 0.10 * min(confidence, 1.0)
        )
        score = float(np.clip(score, 0.0, 1.0))
        level = _severity_level(score)
        return score, level


def _severity_level(score: float) -> str:
    if score < 0.25: return "L1_COSMETIC"
    if score < 0.50: return "L2_MODERATE"
    if score < 0.75: return "L3_SEVERE"
    return "L4_CRITICAL"
