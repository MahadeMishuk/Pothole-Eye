"""
Adaptive Frame Scheduler
Decouples model cadence from frame cadence.

Heavy models (segmentation, depth) run every N frames; their outputs are
temporally propagated via optical flow until the next inference fires.
The interval adapts automatically based on measured GPU latency.
"""
import time
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class ScheduleDecision:
    run_segmentation: bool
    run_depth: bool
    run_detection: bool   #always True — detection is the core task
    seg_frame_age: int    #frames since last segmentation run
    depth_frame_age: int  #frames since last depth run
    frame_idx: int


class AdaptiveFrameScheduler:
    """
    Controls which models fire on each frame.

    Default cadence on A40:
      Detection:    every frame  (~18ms with RT-DETR-L FP16)
      Segmentation: every 3rd    (~22ms with YOLOv8l-seg FP16)
      Depth:        every 5th    (~18ms with DepthAnythingV2-Small FP16)

    The schedule auto-backs off when measured GPU latency exceeds budget,
    and tightens again when headroom returns.
    """

    _SEG_INTERVAL_BASE   = 3
    _DEPTH_INTERVAL_BASE = 5
    _SEG_INTERVAL_MIN    = 2
    _SEG_INTERVAL_MAX    = 10
    _DEPTH_INTERVAL_MIN  = 3
    _DEPTH_INTERVAL_MAX  = 15
    _TARGET_LATENCY_MS   = 33   #30 FPS target frame budget
    _LATENCY_WINDOW      = 30   #rolling window for latency EMA

    def __init__(self):
        self._frame_idx       = 0
        self._seg_interval    = self._SEG_INTERVAL_BASE
        self._depth_interval  = self._DEPTH_INTERVAL_BASE
        self._latency_window  = deque(maxlen=self._LATENCY_WINDOW)
        self._last_seg_idx    = -999
        self._last_depth_idx  = -999

    def tick(self) -> ScheduleDecision:
        """Advance one frame and return the schedule for this frame."""
        idx = self._frame_idx
        self._frame_idx += 1

        self._adapt_intervals()

        run_seg   = (idx - self._last_seg_idx)   >= self._seg_interval
        run_depth = (idx - self._last_depth_idx) >= self._depth_interval

        if run_seg:
            self._last_seg_idx = idx
        if run_depth:
            self._last_depth_idx = idx

        return ScheduleDecision(
            run_segmentation = run_seg,
            run_depth        = run_depth,
            run_detection    = True,
            seg_frame_age    = idx - self._last_seg_idx,
            depth_frame_age  = idx - self._last_depth_idx,
            frame_idx        = idx,
        )

    def record_latency(self, ms: float):
        """Feed back measured per-frame latency to drive adaptive scheduling."""
        self._latency_window.append(ms)

    def reset(self):
        self._frame_idx      = 0
        self._last_seg_idx   = -999
        self._last_depth_idx = -999
        self._latency_window.clear()

    @property
    def seg_interval(self) -> int:
        return self._seg_interval

    @property
    def depth_interval(self) -> int:
        return self._depth_interval

    def _adapt_intervals(self):
        if len(self._latency_window) < 10:
            return

        avg_ms = float(np.mean(self._latency_window))
        budget = self._TARGET_LATENCY_MS

        if avg_ms > budget * 1.4:
            #Overloaded: run heavy models less often
            self._seg_interval   = min(self._seg_interval   + 1, self._SEG_INTERVAL_MAX)
            self._depth_interval = min(self._depth_interval + 1, self._DEPTH_INTERVAL_MAX)
        elif avg_ms < budget * 0.65:
            #Plenty of headroom: tighten cadence for higher quality
            self._seg_interval   = max(self._seg_interval   - 1, self._SEG_INTERVAL_MIN)
            self._depth_interval = max(self._depth_interval - 1, self._DEPTH_INTERVAL_MIN)
