import logging
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _make_bytetrack_args(
    track_high_thresh: float = 0.45,
    track_low_thresh:  float = 0.10,
    new_track_thresh:  float = 0.50,
    track_buffer:      int   = 30,
    match_thresh:      float = 0.80,
) -> SimpleNamespace:
    #fuse_score: ultralytics ≥8.3 accesses self.args.fuse_score during matching.
    #False = pure IoU matching (no score fusion) — correct for single-class pothole model.
    return SimpleNamespace(
        track_high_thresh = track_high_thresh,
        track_low_thresh  = track_low_thresh,
        new_track_thresh  = new_track_thresh,
        track_buffer      = track_buffer,
        match_thresh      = match_thresh,
        fuse_score        = False,
    )


class PotholeByteTracker:
    """
    Thin adapter around ultralytics BYTETracker.

    Inputs:  list of Detection objects (potholes only)
    Outputs: list of TrackedDetection dicts with track_id + metadata
    """

    def __init__(
        self,
        track_high_thresh: float = 0.45,
        track_low_thresh:  float = 0.10,
        new_track_thresh:  float = 0.50,
        track_buffer:      int   = 30,
        match_thresh:      float = 0.80,
        frame_rate:        int   = 30,
    ):
        args = _make_bytetrack_args(
            track_high_thresh = track_high_thresh,
            track_low_thresh  = track_low_thresh,
            new_track_thresh  = new_track_thresh,
            track_buffer      = track_buffer,
            match_thresh      = match_thresh,
        )
        try:
            from ultralytics.trackers.byte_tracker import BYTETracker
            self._tracker = BYTETracker(args, frame_rate=frame_rate)
            self._available = True
            logger.info("ByteTracker initialized (high=%.2f low=%.2f buffer=%d)",
                        track_high_thresh, track_low_thresh, track_buffer)
        except ImportError as e:
            logger.warning("BYTETracker not available (%s) — falling back to SORT", e)
            self._tracker = None
            self._available = False
            self._sort_fallback = None

        self._ema_store: Dict[int, float] = {}
        self._alpha = 0.35

    def update(
        self,
        detections: list,
        frame_shape: tuple,
    ) -> List[dict]:
        """
        Update tracker with current frame's detections.

        Args:
            detections:  List of Detection objects (is_pothole=True)
            frame_shape: (H, W) of current frame

        Returns:
            List of dicts with keys:
              track_id, bbox_xyxy, confidence, confidence_ema,
              severity_score, severity_level, depth_m, road_ratio
        """
        if not detections:
            return []

        if not self._available:
            return self._sort_update(detections)

        #Ultralytics BYTETracker API (≥8.1) expects an object that exposes:
        #  .conf  — (N,)   confidence scores
        #  .xyxy  — (N,4)  boxes as [x1,y1,x2,y2]
        #  .xywh  — (N,4)  boxes as [cx,cy,w,h]  (computed property)
        #  .cls   — (N,)   class IDs
        #  [mask] — boolean / index filtering (returns a new _Dets)
        class _Dets:
            def __init__(self, conf, xyxy, cls):
                self.conf = conf
                self.xyxy = xyxy
                self.cls  = cls

            @property
            def xywh(self):
                #Convert xyxy → cx,cy,w,h — shape (N,4)
                x1, y1, x2, y2 = (
                    self.xyxy[:, 0], self.xyxy[:, 1],
                    self.xyxy[:, 2], self.xyxy[:, 3],
                )
                return np.stack(
                    [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1],
                    axis=1,
                )

            def __getitem__(self, mask):
                return _Dets(self.conf[mask], self.xyxy[mask], self.cls[mask])

            def __len__(self):
                return len(self.conf)

        det_input = _Dets(
            conf = np.array([d.confidence for d in detections], dtype=np.float32),
            xyxy = np.array([[d.x1, d.y1, d.x2, d.y2] for d in detections], dtype=np.float32),
            cls  = np.zeros(len(detections), dtype=np.float32),
        )

        try:
            tracks = self._tracker.update(det_input, frame_shape)
        except Exception as e:
            logger.warning("BYTETracker.update error: %s", e)
            return []

        if tracks is None or len(tracks) == 0:
            return []

        #Match tracks back to detections to carry over metadata
        result = []
        for track in tracks:
            #tracks: [x1, y1, x2, y2, track_id, conf, cls, ...]
            if len(track) < 5:
                continue

            tx1, ty1, tx2, ty2 = track[0], track[1], track[2], track[3]
            tid  = int(track[4])
            conf = float(track[5]) if len(track) > 5 else 0.5

            #Update EMA confidence
            if tid not in self._ema_store:
                self._ema_store[tid] = conf
            else:
                self._ema_store[tid] = (
                    self._alpha * conf
                    + (1.0 - self._alpha) * self._ema_store[tid]
                )

            #Find closest original detection to inherit metadata
            meta = self._match_metadata(detections, tx1, ty1, tx2, ty2)

            result.append({
                "track_id":       tid,
                "bbox_xyxy":      [float(tx1), float(ty1), float(tx2), float(ty2)],
                "confidence":     conf,
                "confidence_ema": self._ema_store[tid],
                "severity_score": meta.get("severity_score", 0.0),
                "severity_level": meta.get("severity_level", "L1_COSMETIC"),
                "depth_m":        meta.get("depth_m", 10.0),
                "road_ratio":     meta.get("road_ratio", 1.0),
            })

        return result

    @staticmethod
    def _match_metadata(detections, tx1, ty1, tx2, ty2) -> dict:
        """Find best-IoU detection to inherit non-tracking metadata."""
        best_iou = 0.0
        best_meta = {}
        for d in detections:
            ix1, iy1 = max(d.x1, tx1), max(d.y1, ty1)
            ix2, iy2 = min(d.x2, tx2), min(d.y2, ty2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            a1 = (d.x2 - d.x1) * (d.y2 - d.y1)
            a2 = (tx2 - tx1) * (ty2 - ty1)
            union = a1 + a2 - inter
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_meta = {
                    "severity_score": getattr(d, "severity_score", 0.0),
                    "severity_level": getattr(d, "severity_level", "L1_COSMETIC"),
                    "depth_m":        getattr(d, "depth_m", 10.0),
                    "road_ratio":     getattr(d, "road_ratio", 1.0),
                }
        return best_meta

    def _sort_update(self, detections: list) -> List[dict]:
        """SORT fallback when BYTETracker is unavailable."""
        if self._sort_fallback is None:
            try:
                from tracking.sort_tracker import Sort
                self._sort_fallback = Sort(max_age=30, min_hits=2, iou_threshold=0.3)
            except Exception:
                return []

        sort_input = np.array([
            [d.x1, d.y1, d.x2, d.y2, d.confidence]
            for d in detections
        ], dtype=np.float32)

        try:
            tracks = self._sort_fallback.update(sort_input)
        except Exception:
            return []

        result = []
        for t in tracks:
            tid = int(t[4])
            if tid not in self._ema_store:
                self._ema_store[tid] = 0.5
            meta = self._match_metadata(detections, t[0], t[1], t[2], t[3])
            result.append({
                "track_id":       tid,
                "bbox_xyxy":      [float(t[0]), float(t[1]), float(t[2]), float(t[3])],
                "confidence":     0.5,
                "confidence_ema": self._ema_store[tid],
                "severity_score": meta.get("severity_score", 0.0),
                "severity_level": meta.get("severity_level", "L1_COSMETIC"),
                "depth_m":        meta.get("depth_m", 10.0),
                "road_ratio":     meta.get("road_ratio", 1.0),
            })
        return result

    def reset(self):
        """Reset tracker state (call between video clips)."""
        self._ema_store.clear()
        if self._available and self._tracker:
            try:
                self._tracker.reset()
            except Exception:
                pass
