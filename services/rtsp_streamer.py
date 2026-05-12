import logging
import subprocess
import threading
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class NVENCRTSPStreamer:
    """
    Pushes annotated BGR frames to MediaMTX via FFmpeg using NVENC H.264.

    Usage:
        streamer = NVENCRTSPStreamer("rtsp://mediamtx:8554/live", 1280, 720, 25)
        streamer.start()
        streamer.push_frame(frame_bgr)   #called from pipeline thread
        streamer.stop()
    """

    def __init__(
        self,
        rtsp_url:  str = "rtsp://mediamtx:8554/live",
        width:     int = 1280,
        height:    int = 720,
        fps:       int = 25,
        bitrate:   str = "4M",
        use_nvenc: bool = True,
    ):
        self._rtsp_url  = rtsp_url
        self._width     = width
        self._height    = height
        self._fps       = fps
        self._bitrate   = bitrate
        self._use_nvenc = use_nvenc
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._running = False

    def start(self):
        """Launch FFmpeg subprocess. Idempotent."""
        if self._running:
            return

        codec_args = self._nvenc_args() if self._use_nvenc else self._sw_args()

        cmd = [
            "ffmpeg",
            "-loglevel", "warning",
            "-f",       "rawvideo",
            "-pix_fmt", "bgr24",
            "-s",       f"{self._width}x{self._height}",
            "-r",       str(self._fps),
            "-i",       "pipe:0",
            *codec_args,
            "-f",       "rtsp",
            "-rtsp_transport", "tcp",
            self._rtsp_url,
        ]

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin  = subprocess.PIPE,
                stdout = subprocess.DEVNULL,
                stderr = subprocess.PIPE,
            )
            self._running = True
            logger.info("RTSP streamer started → %s (codec=%s)",
                        self._rtsp_url,
                        "h264_nvenc" if self._use_nvenc else "libx264")
        except FileNotFoundError:
            logger.error("FFmpeg not found — RTSP streaming disabled")
            self._running = False
        except Exception as e:
            logger.error("FFmpeg launch failed: %s", e)
            self._running = False

    def push_frame(self, frame_bgr: np.ndarray):
        """Write one raw BGR frame to FFmpeg stdin. Non-blocking on error."""
        if not self._running or self._proc is None:
            return
        if self._proc.poll() is not None:
            logger.warning("FFmpeg process died — attempting restart")
            self._running = False
            self.start()
            return

        resized = (
            cv2.resize(frame_bgr, (self._width, self._height))
            if frame_bgr.shape[1] != self._width or frame_bgr.shape[0] != self._height
            else frame_bgr
        )

        try:
            with self._lock:
                self._proc.stdin.write(resized.tobytes())
        except BrokenPipeError:
            logger.warning("RTSP streamer: broken pipe (FFmpeg exited)")
            self._running = False
        except Exception as e:
            logger.debug("RTSP push error: %s", e)

    def stop(self):
        """Terminate FFmpeg subprocess."""
        self._running = False
        if self._proc:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
        logger.info("RTSP streamer stopped.")

    def _nvenc_args(self) -> list:
        return [
            "-c:v",     "h264_nvenc",
            "-preset",  "p2",          #low-latency preset (NVENC SDK v9+)
            "-tune",    "ll",
            "-zerolatency", "1",
            "-b:v",     self._bitrate,
            "-maxrate", self._bitrate,
            "-bufsize",  "1M",
            "-g",       str(self._fps),    #keyframe every 1 second
            "-bf",      "0",               #no B-frames for low latency
            "-pix_fmt", "yuv420p",
        ]

    def _sw_args(self) -> list:
        return [
            "-c:v",    "libx264",
            "-preset", "ultrafast",
            "-tune",   "zerolatency",
            "-b:v",    self._bitrate,
            "-g",      str(self._fps),
            "-pix_fmt", "yuv420p",
        ]

    @property
    def is_running(self) -> bool:
        return self._running and self._proc is not None and self._proc.poll() is None
