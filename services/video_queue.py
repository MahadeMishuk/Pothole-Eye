import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

#Celery app is created lazily so importing this module doesn't require Redis
_celery_app = None


def get_celery_app():
    global _celery_app
    if _celery_app is not None:
        return _celery_app

    from celery import Celery
    broker  = os.environ.get("CELERY_BROKER_URL",  "redis://redis:6379/0")
    backend = os.environ.get("CELERY_RESULT_BACKEND", "redis://redis:6379/1")

    app = Celery("pothole_video", broker=broker, backend=backend)
    app.conf.update(
        result_expires       = 86400,   #24h
        task_serializer      = "json",
        result_serializer    = "json",
        accept_content       = ["json"],
        worker_prefetch_multiplier = 1,  #one task at a time on GPU worker
        task_acks_late       = True,     #only ack after completion
    )
    _celery_app = app
    return app


def make_video_task():
    """Factory: returns the Celery task bound to the current app."""
    celery = get_celery_app()

    @celery.task(bind=True, max_retries=2, name="process_uploaded_video")
    def process_uploaded_video(self, video_path: str, job_id: str):
        """
        Async video processing task.

        Batches frames (BATCH_SIZE=8) for efficient GPU utilisation.
        Writes progress to Redis every 10 processed frames.
        Stores annotated output at /app/outputs/<job_id>.mp4
        """
        _set_progress(job_id, 0, "processing")

        try:
            return _run_video_job(self, video_path, job_id)
        except Exception as exc:
            logger.error("Video job %s failed: %s", job_id, exc)
            _set_progress(job_id, 0, f"error: {exc}")
            raise self.retry(exc=exc, countdown=10)

    return process_uploaded_video


def submit_video_job(video_path: str) -> str:
    """
    Submit an uploaded video for background processing.
    Returns job_id that the caller uses to poll progress.
    """
    job_id = uuid.uuid4().hex[:12]
    task   = make_video_task()
    task.delay(video_path, job_id)
    logger.info("Submitted video job %s: %s", job_id, video_path)
    return job_id


def get_job_progress(job_id: str) -> dict:
    """
    Poll job progress from Redis.
    Returns: {progress: 0-100, status: str, output: str|None}
    """
    try:
        import redis
        r = redis.from_url(
            os.environ.get("CELERY_BROKER_URL", "redis://redis:6379/0")
        )
        progress = int(r.get(f"job:{job_id}:progress") or 0)
        status   = (r.get(f"job:{job_id}:status") or b"processing").decode()
        output   = r.get(f"job:{job_id}:output")
        output   = output.decode() if output else None
        return {"progress": progress, "status": status, "output": output}
    except Exception as e:
        logger.warning("Redis progress fetch failed: %s", e)
        return {"progress": 0, "status": "unknown", "output": None}


#Internal implementation───

def _run_video_job(task, video_path: str, job_id: str) -> dict:
    #Import here to avoid circular import at module load time
    from flask import current_app
    from config import config as cfg

    #Get pipeline — singleton, already loaded if main app is running
    from app import get_pipeline, app as flask_app
    pipeline = get_pipeline()

    from pipeline.ingestion import FramePacket

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps_src      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_src        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_src        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    max_w  = cfg.MAX_FRAME_WIDTH
    scale  = min(1.0, max_w / max(w_src, 1))
    out_w  = int(w_src * scale)
    out_h  = int(h_src * scale) + 60  #+60 for HUD panel

    out_path = str(Path(cfg.OUTPUT_FOLDER) / f"{job_id}.mp4")
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    skip     = max(1, cfg.FRAME_SKIP)
    out_fps  = fps_src / skip

    writer = cv2.VideoWriter(out_path, fourcc, out_fps, (out_w, out_h))

    BATCH_SIZE = 8
    frame_batch  = []
    frame_idx    = 0
    processed    = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % skip == 0:
            frame_batch.append((frame_idx, frame))

        if len(frame_batch) == BATCH_SIZE:
            _process_and_write(pipeline, flask_app, frame_batch, writer, job_id)
            processed += len(frame_batch)
            frame_batch = []

            if processed % 40 == 0:
                progress = min(99, int(processed / max(total_frames / skip, 1) * 100))
                _set_progress(job_id, progress, "processing")

        frame_idx += 1

    #Flush remaining frames
    if frame_batch:
        _process_and_write(pipeline, flask_app, frame_batch, writer, job_id)
        processed += len(frame_batch)

    cap.release()
    writer.release()

    _set_progress(job_id, 100, "done")
    _set_output(job_id, f"{job_id}.mp4")

    logger.info("Video job %s complete — %d frames processed → %s",
                job_id, processed, out_path)
    return {"job_id": job_id, "output": out_path, "frames_processed": processed}


def _process_and_write(pipeline, flask_app, frame_batch, writer, job_id):
    from pipeline.ingestion import FramePacket
    packets = [
        FramePacket(
            frame_bgr = frame,
            frame_idx = idx,
            timestamp = time.monotonic(),
            source    = "upload",
            job_id    = job_id,
        )
        for idx, frame in frame_batch
    ]

    with flask_app.app_context():
        results = pipeline.process_batch_sync(packets)
        for r in results:
            if r.annotated_frame is not None:
                writer.write(r.annotated_frame)


def _set_progress(job_id: str, progress: int, status: str):
    try:
        import redis
        r = redis.from_url(
            os.environ.get("CELERY_BROKER_URL", "redis://redis:6379/0")
        )
        r.setex(f"job:{job_id}:progress", 86400, str(progress))
        r.setex(f"job:{job_id}:status",   86400, status)
    except Exception:
        pass


def _set_output(job_id: str, filename: str):
    try:
        import redis
        r = redis.from_url(
            os.environ.get("CELERY_BROKER_URL", "redis://redis:6379/0")
        )
        r.setex(f"job:{job_id}:output", 86400, filename)
    except Exception:
        pass
