import queue
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class FramePacket:
    """Atomic unit passed through the pipeline."""
    frame_bgr:  np.ndarray       #H × W × 3 BGR uint8
    frame_idx:  int
    timestamp:  float            #time.monotonic() at capture
    gps:        Optional[tuple] = None   #(lat, lon) or None
    source:     str = "stream"   #"stream" | "upload" | "browser"
    job_id:     Optional[str] = None    #for upload tracking


class RingFrameBuffer:
    

    def __init__(self, capacity: int = 64):
        self._q = queue.Queue(maxsize=capacity)
        self.dropped_frames = 0
        self._capacity = capacity

    def push(self, packet: FramePacket) -> bool:
        """
        Add frame to buffer. Returns True if accepted, False if dropped.
        Non-blocking — if full, evicts oldest frame to make room.
        """
        if self._q.full():
            try:
                self._q.get_nowait()   #evict oldest
                self.dropped_frames += 1
            except queue.Empty:
                pass

        try:
            self._q.put_nowait(packet)
            return True
        except queue.Full:
            self.dropped_frames += 1
            return False

    def pop(self, timeout: float = 0.05) -> Optional[FramePacket]:
        """Block for up to `timeout` seconds waiting for a frame."""
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def pop_batch(self, n: int = 4, timeout: float = 0.05) -> List[FramePacket]:
        """
        Drain up to N frames from the buffer.
        Blocks up to `timeout` seconds for the first frame,
        then drains remaining frames without blocking.
        """
        batch: List[FramePacket] = []

        #Wait for first frame
        first = self.pop(timeout=timeout)
        if first is None:
            return batch
        batch.append(first)

        for _ in range(n - 1):
            try:
                batch.append(self._q.get_nowait())
            except queue.Empty:
                break

        return batch

    def qsize(self) -> int:
        return self._q.qsize()

    def clear(self):
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    def __len__(self) -> int:
        return self._q.qsize()
