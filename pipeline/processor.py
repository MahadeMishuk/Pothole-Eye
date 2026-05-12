"""
Pipeline Processor — Backward-Compatible Wrapper

Delegates all real work to GPUInferenceWorker.
Preserves the FrameResult interface so all existing callers in app.py
continue to work without modification.
"""
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from pipeline.gpu_worker import GPUInferenceWorker, PerceptionResult
from pipeline.ingestion import FramePacket
from pipeline.detection import Detection

logger = logging.getLogger(__name__)


@dataclass
class FrameResult:
    """Public output contract — unchanged from v1 for caller compatibility."""
    frame:             np.ndarray                    #Annotated BGR frame
    raw_frame:         np.ndarray                    #Original BGR frame
    detections:        List[Detection] = field(default_factory=list)
    road_mask:         Optional[np.ndarray] = None
    fps:               float = 0.0
    frame_number:      int   = 0
    processing_time_ms: float = 0.0
    pothole_count:     int   = 0
    alert_triggered:   bool  = False
    alert_message:     str   = ""

    @classmethod
    def from_perception(cls, r: PerceptionResult, frame_number: int) -> "FrameResult":
        return cls(
            frame              = r.annotated_frame,
            raw_frame          = r.frame_bgr,
            detections         = r.detections,
            road_mask          = r.seg_mask,
            fps                = r.fps,
            frame_number       = frame_number,
            processing_time_ms = r.processing_ms,
            pothole_count      = r.pothole_count,
            alert_triggered    = r.alert_triggered,
            alert_message      = r.alert_message,
        )


class PipelineProcessor:
    """
    Thin wrapper around GPUInferenceWorker.

    Existing call sites:
        result = pipeline.process(frame_bgr, frame_number=n, gps=(lat, lon))
    continue to work identically.

    New batch path (for Celery video worker):
        results = pipeline.process_batch_sync(packets)
    """

    def __init__(self, model_manager, config):
        self.config = config
        self._worker = GPUInferenceWorker(model_manager, config)
        #Expose scheduler for external diagnostics
        self.scheduler = self._worker.scheduler

    def process(
        self,
        frame_bgr:    np.ndarray,
        frame_number: int = 0,
        gps:          Optional[Tuple[float, float]] = None,
    ) -> FrameResult:
        """Single-frame synchronous processing — maintains v1 API."""
        result = self._worker.process_single(frame_bgr, frame_number, gps)
        return FrameResult.from_perception(result, frame_number)

    def process_batch_sync(
        self,
        packets: List[FramePacket],
    ) -> List[PerceptionResult]:
        """Batch processing — used by Celery video upload worker."""
        return self._worker.process_batch_sync(packets)
