import logging
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from pipeline.detection import Detection, draw_detections
from pipeline.depth import risk_label, risk_color_bgr
from pipeline.segmentation import RoadSegmenter
from pipeline.detection import PotholeDetector
from pipeline.depth import DepthEstimator
from pipeline.fusion import PerceptionFusion
from pipeline.scheduler import AdaptiveFrameScheduler
from pipeline.ingestion import FramePacket

logger = logging.getLogger(__name__)


@dataclass
class PerceptionResult:
    """Output of processing one frame through the full GPU stack."""
    frame_idx:       int
    frame_bgr:       np.ndarray              #original input
    annotated_frame: np.ndarray              #with boxes / HUD
    detections:      List[Detection] = field(default_factory=list)
    seg_mask:        Optional[np.ndarray] = None
    depth_map:       Optional[np.ndarray] = None
    fps:             float = 0.0
    processing_ms:   float = 0.0
    pothole_count:   int   = 0
    alert_triggered: bool  = False
    alert_message:   str   = ""
    gps:             Optional[tuple] = None
    timestamp:       float = 0.0
    source:          str   = "stream"
    job_id:          Optional[str] = None


class GPUInferenceWorker:
    """
    Batched GPU inference worker.

    All model objects live inside this class — nothing else should
    instantiate models or call CUDA directly.
    """

    def __init__(self, model_manager, config):
        self.config    = config
        self.mm        = model_manager

        self.segmenter = RoadSegmenter(model_manager, config)
        self.detector  = PotholeDetector(model_manager, config)
        self.depth_est = DepthEstimator(model_manager, config)
        self.fusion    = PerceptionFusion(
            road_ratio_min=getattr(config, "FUSION_ROAD_RATIO_MIN", 0.35)
        )
        self.scheduler = AdaptiveFrameScheduler()

        #Cached outputs from last heavy model run
        self._cached_seg:   Optional[np.ndarray] = None
        self._cached_depth: Optional[np.ndarray] = None

        #FPS tracking
        self._fps_times: list = []
        self._fps_window: int = 30

        #Alert cooldown
        self._last_alert_ts: float = 0.0

        #Concurrency guard
        self._lock = threading.Lock()

    #Public API────────────

    def process_single(
        self,
        frame_bgr:    np.ndarray,
        frame_number: int = 0,
        gps:          Optional[tuple] = None,
    ) -> PerceptionResult:
        """Process one frame synchronously. Thread-safe via lock."""
        packet = FramePacket(
            frame_bgr = frame_bgr,
            frame_idx = frame_number,
            timestamp = time.monotonic(),
            gps       = gps,
        )
        results = self.process_batch_sync([packet])
        return results[0] if results else self._empty_result(frame_bgr, frame_number)

    def process_batch_sync(
        self,
        packets: List[FramePacket],
    ) -> List[PerceptionResult]:
        """
        Process a batch of frames. One GPU forward pass per model type.
        Thread-safe — acquires lock for duration of GPU work.
        """
        if not packets:
            return []

        with self._lock:
            return self._process_batch_impl(packets)

    #Core batch implementation─

    def _process_batch_impl(self, packets: List[FramePacket]) -> List[PerceptionResult]:
        results: List[PerceptionResult] = []

        #Batch detection (all frames → one forward pass if RT-DETR supports it)
        frames = [self._resize(p.frame_bgr) for p in packets]
        batch_detections = self.detector.detect_batch(frames)

        for i, (packet, frame, det_list) in enumerate(
            zip(packets, frames, batch_detections)
        ):
            t0 = time.perf_counter()
            decision = self.scheduler.tick()

            #Segmentation───
            if decision.run_segmentation:
                try:
                    self._cached_seg = self.segmenter.segment(frame)
                except Exception as e:
                    logger.warning("Segmentation error: %s", e)
            elif self._cached_seg is not None and decision.seg_frame_age > 0:
                self._cached_seg = self.segmenter.warp_mask(self._cached_seg, frame)

            #Depth─────
            if decision.run_depth:
                try:
                    self._cached_depth = self.depth_est.get_depth_map(frame)
                except Exception as e:
                    logger.warning("Depth error: %s", e)
            #depth cache is used as-is (no warping — depth changes slowly)

            #Depth assignment to existing detections
            if self._cached_depth is not None:
                h, w = frame.shape[:2]
                for det in det_list:
                    det.distance_m = self.depth_est._sample_depth(
                        det, self._cached_depth, w, h
                    )
                    det.depth_m    = det.distance_m or 0.0
                    det.risk_level = self.depth_est._categorise(det.distance_m)
            else:
                h, w = frame.shape[:2]
                for det in det_list:
                    dist = self.depth_est._geometric_distance(det, w, h)
                    det.distance_m = dist
                    det.depth_m    = dist or 0.0
                    det.risk_level = self.depth_est._categorise(dist)

            #Fusion (road mask gating + severity)─
            fused_detections = self.fusion.fuse(
                detections = det_list,
                seg_mask   = self._cached_seg,
                depth_map  = self._cached_depth,
                frame_bgr  = frame,
            )

            #Annotation─
            annotated = self._annotate(frame, fused_detections, self._cached_seg)

            t_end    = time.perf_counter()
            proc_ms  = (t_end - t0) * 1000.0
            fps      = self._update_fps(t_end)
            potholes = [d for d in fused_detections if d.is_pothole]
            alert_triggered, alert_msg = self._check_alert(potholes)

            self.scheduler.record_latency(proc_ms)

            results.append(PerceptionResult(
                frame_idx       = packet.frame_idx,
                frame_bgr       = packet.frame_bgr,
                annotated_frame = annotated,
                detections      = fused_detections,
                seg_mask        = self._cached_seg,
                depth_map       = self._cached_depth,
                fps             = fps,
                processing_ms   = round(proc_ms, 1),
                pothole_count   = len(potholes),
                alert_triggered = alert_triggered,
                alert_message   = alert_msg,
                gps             = packet.gps,
                timestamp       = packet.timestamp,
                source          = packet.source,
                job_id          = packet.job_id,
            ))

        return results

    #Annotation────────────

    def _annotate(
        self,
        frame_bgr:   np.ndarray,
        detections:  List[Detection],
        seg_mask:    Optional[np.ndarray],
    ) -> np.ndarray:
        annotated = frame_bgr.copy()

        if seg_mask is not None and self.config.SHOW_ROAD_MASK:
            annotated = RoadSegmenter.overlay_mask(
                annotated, seg_mask, alpha=self.config.ROAD_MASK_ALPHA
            )

        annotated = draw_detections(annotated, detections,
                                    show_distance=True, show_severity=True)

        near_potholes = [d for d in detections if d.is_pothole and d.risk_level == "near"]
        if near_potholes:
            h, w = annotated.shape[:2]
            overlay = annotated.copy()
            for thick in [6, 12, 18]:
                cv2.rectangle(overlay, (thick, thick),
                              (w - thick, h - thick), (0, 0, 255), thick)
            annotated = cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0)

        return annotated

    #Alert─

    def _check_alert(self, potholes: List[Detection]):
        now  = time.time()
        near = [d for d in potholes if d.risk_level == "near"]
        if not near:
            return False, ""
        if now - self._last_alert_ts < self.config.ALERT_COOLDOWN_SEC:
            return True, ""
        self._last_alert_ts = now
        closest = min(near, key=lambda d: d.depth_m or d.distance_m or 999.0)
        dist = closest.depth_m or closest.distance_m or 0
        return True, f"POTHOLE {dist:.1f}m ahead! [{closest.severity_level}]"

    #FPS tracking──────────

    def _update_fps(self, ts: float) -> float:
        self._fps_times.append(ts)
        if len(self._fps_times) > self._fps_window:
            self._fps_times.pop(0)
        if len(self._fps_times) < 2:
            return 0.0
        elapsed = self._fps_times[-1] - self._fps_times[0]
        return (len(self._fps_times) - 1) / elapsed if elapsed > 0 else 0.0

    #Resize

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        max_w = self.config.MAX_FRAME_WIDTH
        if w <= max_w:
            return frame
        scale = max_w / w
        return cv2.resize(frame, (max_w, int(h * scale)), interpolation=cv2.INTER_LINEAR)

    #Empty result

    @staticmethod
    def _empty_result(frame_bgr: np.ndarray, frame_idx: int) -> PerceptionResult:
        return PerceptionResult(
            frame_idx       = frame_idx,
            frame_bgr       = frame_bgr,
            annotated_frame = frame_bgr.copy(),
        )
