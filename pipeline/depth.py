import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from pipeline.detection import Detection

logger = logging.getLogger(__name__)


class DepthEstimator:


    def __init__(self, model_manager, config):
        self.mm     = model_manager
        self.config = config

        #Cache last depth map for temporal smoothing
        self._cached_depth:  Optional[np.ndarray] = None
        self._cache_weight:  float = 0.0

    #Public API────────────

    def estimate(
        self,
        frame_bgr:  np.ndarray,
        detections: List[Detection],
    ) -> List[Detection]:

        if not detections:
            return detections

        depth_map = self.get_depth_map(frame_bgr)

        h, w = frame_bgr.shape[:2]
        for det in detections:
            dist = self._sample_depth(det, depth_map, w, h)
            det.depth_m   = round(dist, 2) if dist else 0.0
            det.distance_m = det.depth_m   #keep backward-compat alias
            det.risk_level = self._categorise(dist)

        return detections

    def get_depth_map(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    
        depth_map = None

        #Backend 1: Depth Anything V2 (metric, absolute)
        if getattr(self.mm, "depth_anything", None) is not None:
            depth_map = self._depth_anything_map(frame_bgr)

        #Backend 2: MiDaS (relative, calibrated via geometry)
        if depth_map is None and getattr(self.mm, "midas", None) is not None:
            depth_map = self._midas_map(frame_bgr)

        return depth_map

    #Depth Anything V2────

    def _depth_anything_map(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
      
        try:
            model, processor = self.mm.depth_anything
            h, w = frame_bgr.shape[:2]

            from PIL import Image
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)

            inputs = processor(images=pil_image, return_tensors="pt")
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            #Run in FP16 if on CUDA for speed
            if device.type == "cuda":
                model = model.half()
                inputs = {k: v.half() if v.is_floating_point() else v
                          for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = model(**inputs)

            depth = outputs.predicted_depth   #[1, H', W'] in metres

            depth_up = torch.nn.functional.interpolate(
                depth.unsqueeze(1).float(),
                size  = (h, w),
                mode  = "bicubic",
                align_corners = False,
            ).squeeze().cpu().numpy()

            #Clamp to sane outdoor range [0.1m, 200m]
            depth_up = np.clip(depth_up, 0.1, 200.0).astype(np.float32)
            return depth_up

        except Exception as e:
            logger.warning("Depth Anything V2 inference error: %s", e)
            return None

    #MiDaS (fallback)─────

    def _midas_map(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Run MiDaS small. Outputs relative inverse depth — larger = closer.
        We convert to approximate metric depth using geometric calibration.
        """
        try:
            transform = self.mm.midas_transform
            img_rgb   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            batch     = transform(img_rgb)

            with torch.inference_mode():
                pred = self.mm.midas(batch)
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1),
                    size  = frame_bgr.shape[:2],
                    mode  = "bicubic",
                    align_corners = False,
                ).squeeze()

            inv_depth = pred.cpu().numpy().astype(np.float32)

            #Convert relative inverse depth to approximate metric:
            #MiDaS: large value = close. Scale by empirical factor.
            d_min, d_max = inv_depth.min(), inv_depth.max()
            if d_max == d_min:
                return None

            norm  = (inv_depth - d_min) / (d_max - d_min)  #0=far, 1=near
            #Map to [0.5m, 80m] — road scene typical range
            depth_m = (1.0 - norm) * 79.5 + 0.5
            return depth_m.astype(np.float32)

        except Exception as e:
            logger.warning("MiDaS inference error: %s", e)
            return None

    #Depth sampling per detection

    def _sample_depth(
        self,
        det:       Detection,
        depth_map: Optional[np.ndarray],
        frame_w:   int,
        frame_h:   int,
    ) -> Optional[float]:
        """Sample median depth within bbox, fallback to geometric."""
        if depth_map is not None:
            x1 = max(0, int(det.x1))
            y1 = max(0, int(det.y1))
            x2 = min(frame_w - 1, int(det.x2))
            y2 = min(frame_h - 1, int(det.y2))
            roi = depth_map[y1:y2, x1:x2]
            valid = roi[(roi > 0.1) & (roi < 200.0)]
            if valid.size > 0:
                return float(np.median(valid))

        return self._geometric_distance(det, frame_w, frame_h)

    #Geometric fallback────

    def _geometric_distance(
        self, det: Detection, frame_w: int, frame_h: int
    ) -> Optional[float]:
        REAL_DIMS = {
            "pothole":    (0.5,),
            "car":        (1.8, 1.5),
            "truck":      (2.5, 3.0),
            "bus":        (2.5, 3.5),
            "motorcycle": (0.7, 1.0),
            "bicycle":    (0.5, 1.0),
            "person":     (0.5, 1.7),
        }

        f   = self.config.FOCAL_LENGTH_PX
        cls = det.class_name.lower()
        bbox_w = det.x2 - det.x1

        ref = REAL_DIMS.get(cls)
        if ref and bbox_w > 0:
            return max(0.5, (f * ref[0]) / bbox_w)

        #Y-position heuristic fallback
        cy     = (det.y1 + det.y2) / 2.0
        norm_y = cy / frame_h
        if norm_y < 0.5:
            return None
        frac = (norm_y - 0.5) / 0.5
        return round(50.0 * (1.0 - frac) ** 2 + 1.0, 1)

    #Risk categorisation──

    def _categorise(self, distance_m: Optional[float]) -> str:
        if distance_m is None:
            return "far"
        if distance_m <= self.config.NEAR_DISTANCE:
            return "near"
        if distance_m <= self.config.MEDIUM_DISTANCE:
            return "medium"
        return "far"


#Module-level helpers (kept for backward compat)

def risk_label(level: str) -> str:
    return {"near": "DANGER", "medium": "CAUTION", "far": "DETECTED"}.get(level, "DETECTED")


def risk_color_bgr(level: str) -> Tuple[int, int, int]:
    return {
        "near":   (0, 0, 255),
        "medium": (0, 165, 255),
        "far":    (0, 255, 0),
    }.get(level, (0, 255, 0))
