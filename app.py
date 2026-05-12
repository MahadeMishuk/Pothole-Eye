import base64
import logging
import os
import threading
import time
import uuid

from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename

#App Factory
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

from config import Config

Config.init_dirs()

app = Flask(__name__)
app.config.from_object(Config)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0 

from database.models import db
db.init_app(app)

socketio = SocketIO(
    app,
    async_mode="threading",
    cors_allowed_origins="*",
    max_http_buffer_size=10 * 1024 * 1024,
)

#Model + Pipeline (lazy-loaded on first use)
_model_manager: Optional[object] = None
_pipeline:      Optional[object] = None
_pothole_memory: Optional[object] = None
_db_ops:        Optional[object] = None
_reporter:      Optional[object] = None
_alert_manager: Optional[object] = None
_rtsp_streamer: Optional[object] = None
_init_lock = threading.Lock()


def get_pipeline():
    global _model_manager, _pipeline, _pothole_memory, _db_ops
    global _reporter, _alert_manager, _rtsp_streamer

    with _init_lock:
        if _pipeline is not None:
            return _pipeline

        logger.info("Initialising ML pipeline …")
        from config import config
        from models.model_manager import ModelManager
        from pipeline.processor import PipelineProcessor
        from tracking.pothole_memory import PotholeMemory
        from database.operations import PotholeDB
        from reporting.reporter import MDOTReporter
        from alerts.alert_manager import AlertManager

        _model_manager = ModelManager(config)
        _model_manager.initialize()

        _pipeline       = PipelineProcessor(_model_manager, config)
        _db_ops         = PotholeDB(spatial_threshold_m=config.SPATIAL_DEDUP_THRESHOLD_M)
        _pothole_memory = PotholeMemory(
            config, _db_ops, config.SNAPSHOTS_FOLDER,
            on_pothole_added=lambda data: socketio.emit("new_pothole", data),
        )
        _reporter       = MDOTReporter(config, _db_ops, str(Path("static")))
        _alert_manager  = AlertManager(config, socketio)

        app.extensions["pothole_db"]       = _db_ops
        app.extensions["pothole_reporter"] = _reporter
        app.extensions["alert_manager"]    = _alert_manager

        #Start RTSP streamer if enabled (camera live feed → MediaMTX → WebRTC)
        if config.RTSP_ENABLED:
            try:
                from services.rtsp_streamer import NVENCRTSPStreamer
                _rtsp_streamer = NVENCRTSPStreamer(
                    rtsp_url    = config.RTSP_URL,
                    width       = config.STREAM_WIDTH,
                    height      = config.STREAM_HEIGHT,
                    fps         = config.STREAM_FPS,
                )
                _rtsp_streamer.start()
                logger.info("RTSP streamer started → %s", config.RTSP_URL)
            except Exception as exc:
                logger.warning("RTSP streamer init failed (continuing without): %s", exc)
                _rtsp_streamer = None

        logger.info("Pipeline initialised.")
        return _pipeline


#App State
class AppState:
    camera_active         = False
    browser_camera_active = False
    frame_count           = 0
    current_gps: Optional[tuple] = None
    gps_source: str = "none"
    _camera_thread: Optional[threading.Thread] = None
    _frame_lock = threading.Lock()


state = AppState()

#In-process upload job store (thread-safe, no Redis required)
#{job_id: {"progress": 0-100, "status": str, "output": str|None}}
_upload_jobs: dict = {}
_upload_jobs_lock = threading.Lock()


#Blueprint
from api.routes import api_bp
app.register_blueprint(api_bp)


#DB Init + Schema Migration

def _migrate_db():
    """
    Idempotent migration: add v2 columns to any existing potholes / detection_events
    tables that were created before the v2 model was introduced.

    SQLAlchemy's db.create_all() only creates NEW tables; it never alters existing
    ones.  Without this, every Pothole.query call raises
      OperationalError: no such column: potholes.severity_score
    which @handle_errors swallows and the UI silently shows 0 potholes.
    """
    from sqlalchemy import text, inspect as sa_inspect
    from database.models import db as _db_obj

    inspector = sa_inspect(_db_obj.engine)

    #potholes table — v2 additions
    try:
        pothole_cols = {c["name"] for c in inspector.get_columns("potholes")}
    except Exception:
        return  #table not yet created — create_all() will handle it

    potholes_v2 = [
        ("severity_score", "ALTER TABLE potholes ADD COLUMN severity_score REAL    DEFAULT 0.0"),
        ("severity_level", "ALTER TABLE potholes ADD COLUMN severity_level TEXT    DEFAULT 'L1_COSMETIC'"),
        ("track_state",    "ALTER TABLE potholes ADD COLUMN track_state    TEXT    DEFAULT 'confirmed'"),
        ("cluster_id",     "ALTER TABLE potholes ADD COLUMN cluster_id     TEXT"),
        ("source",         "ALTER TABLE potholes ADD COLUMN source         TEXT    DEFAULT 'camera'"),
    ]

    #detection_events table — v2 additions
    try:
        evt_cols = {c["name"] for c in inspector.get_columns("detection_events")}
    except Exception:
        evt_cols = set()

    events_v2 = [
        ("latitude",  "ALTER TABLE detection_events ADD COLUMN latitude  REAL"),
        ("longitude", "ALTER TABLE detection_events ADD COLUMN longitude REAL"),
    ]

    with _db_obj.engine.connect() as conn:
        for col, sql in potholes_v2:
            if col not in pothole_cols:
                conn.execute(text(sql))
                conn.commit()
                logger.info("DB migration: added potholes.%s", col)

        for col, sql in events_v2:
            if col not in evt_cols:
                conn.execute(text(sql))
                conn.commit()
                logger.info("DB migration: added detection_events.%s", col)


with app.app_context():
    db.create_all()
    _migrate_db()
    logger.info("Database tables created/verified.")


#Utility

def _encode_frame(frame: np.ndarray, quality: int = None) -> str:
    q = quality or Config.JPEG_QUALITY
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _allowed_file(filename: str) -> bool:
    ext = Path(filename).suffix.lstrip(".").lower()
    return ext in Config.ALLOWED_EXTENSIONS


#Shared frame processor───

def _process_and_emit(frame: np.ndarray, sid: Optional[str] = None):
    """Run CV pipeline on one BGR frame, emit via SocketIO, push to RTSP."""
    pipeline = get_pipeline()
    if not state._frame_lock.acquire(blocking=False):
        return

    try:
        with app.app_context():
            result = pipeline.process(
                frame,
                frame_number=state.frame_count,
                gps=state.current_gps,
            )
            state.frame_count += 1

            if _pothole_memory:
                _pothole_memory.update(
                    result.detections,
                    result.raw_frame,
                    state.frame_count,
                    gps=state.current_gps,
                )
            if _alert_manager:
                _alert_manager.process(result.detections)

        #Push annotated frame to RTSP streamer (for WebRTC path)
        if _rtsp_streamer and _rtsp_streamer.is_running:
            _rtsp_streamer.push_frame(result.frame)

        payload = {
            "image":         _encode_frame(result.frame),
            "fps":           round(result.fps, 1),
            "pothole_count": result.pothole_count,
            "processing_ms": round(result.processing_time_ms, 1),
            "alert":         result.alert_triggered,
            "alert_message": result.alert_message,
            "frame_number":  state.frame_count,
        }
        if sid:
            socketio.emit("frame", payload, to=sid)
        else:
            socketio.emit("frame", payload)
    finally:
        state._frame_lock.release()


#Server-side camera thread

def _camera_worker(stream_url: Optional[str] = None):
    """
    Capture frames from either a local device (stream_url=None → index 0) or a
    remote RTSP/HTTP stream.  For RTSP we use CAP_FFMPEG and reduce the internal
    buffer to 1 frame so we always decode the most recent frame, not a buffered one.
    If the stream drops we attempt one reconnect before emitting camera_error.
    """
    source = stream_url or 0
    is_remote = isinstance(source, str)

    def _open():
        if is_remote:
            c = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            c = cv2.VideoCapture(source)
        return c

    cap = _open()

    if not cap.isOpened():
        if is_remote:
            msg = f"Cannot connect to stream: {source}"
            fallback = None
        else:
            msg = "Server camera not found. Use Browser Camera mode instead."
            fallback = "browser"
        logger.warning("Camera open failed: %s", msg)
        socketio.emit("camera_error", {"message": msg, "fallback": fallback})
        state.camera_active = False
        return

    mode = f"RTSP/HTTP ({source})" if is_remote else "device camera"
    logger.info("Server camera started: %s", mode)
    frame_idx         = 0
    skip              = Config.FRAME_SKIP
    consecutive_fails = 0

    while state.camera_active:
        ret, frame = cap.read()
        if not ret:
            consecutive_fails += 1
            #After ~1.5 s of failures on a remote stream, attempt one reconnect
            if is_remote and consecutive_fails >= 30:
                logger.warning("Stream lost — attempting reconnect: %s", source)
                socketio.emit("camera_status", {"status": "reconnecting"})
                cap.release()
                time.sleep(2)
                cap = _open()
                consecutive_fails = 0
                if not cap.isOpened():
                    socketio.emit("camera_error", {
                        "message": f"Stream reconnect failed: {source}",
                        "fallback": None,
                    })
                    break
            else:
                time.sleep(0.05)
            continue

        consecutive_fails = 0
        frame_idx += 1
        if frame_idx % skip != 0:
            time.sleep(0.001)
            continue

        _process_and_emit(frame)
        time.sleep(0.001)

    cap.release()
    logger.info("Server camera stopped.")


#HTTP Routes 

@app.route("/")
def index():
    return render_template(
        "index.html",
        mapbox_token=Config.MAPBOX_ACCESS_TOKEN,
        mapbox_style=Config.MAPBOX_STYLE,
        default_lat=Config.DEFAULT_LAT,
        default_lng=Config.DEFAULT_LNG,
        default_zoom=Config.DEFAULT_ZOOM,
        cache_bust=int(time.time()),
    )


@app.route("/upload", methods=["POST"])
def upload_video():
    """
    Accept a video file, save it, and start a background thread for processing.
    Returns job_id immediately. Progress available via:
      - REST:     GET /api/jobs/<job_id>/progress
      - SocketIO: 'upload_frame' events (preview) + 'upload_complete' event
    """
    if "video" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["video"]
    if not f.filename or not _allowed_file(f.filename):
        return jsonify({"error": "Invalid file type"}), 400

    uid        = uuid.uuid4().hex[:8]
    filename   = f"{uid}_{secure_filename(f.filename)}"
    input_path = os.path.join(Config.UPLOAD_FOLDER, filename)
    f.save(input_path)

    if not os.path.exists(input_path) or os.path.getsize(input_path) == 0:
        return jsonify({"error": "File save failed"}), 500

    job_id = uid
    with _upload_jobs_lock:
        _upload_jobs[job_id] = {"progress": 0, "status": "processing", "output": None}

    t = threading.Thread(target=_upload_worker, args=(input_path, job_id), daemon=True)
    t.start()

    logger.info("Upload job %s started: %s (%.1f MB)",
                job_id, filename, os.path.getsize(input_path) / 1e6)
    return jsonify({"status": "queued", "job_id": job_id})


@app.route("/api/jobs/<job_id>/progress")
def job_progress(job_id: str):
    """REST progress endpoint — works with both in-process and Celery jobs."""
    with _upload_jobs_lock:
        job = _upload_jobs.get(job_id)
    if job:
        return jsonify(job)
    return jsonify({"progress": 0, "status": "not_found", "output": None}), 404


def _set_job(job_id: str, progress: int, status: str, output: Optional[str] = None):
    with _upload_jobs_lock:
        _upload_jobs[job_id] = {"progress": progress, "status": status, "output": output}


def _upload_worker(video_path: str, job_id: str):
    """
    Unified video ingestion worker — same GPU pipeline as the live camera path.
    Emits SocketIO 'upload_frame' events for real-time preview and
    'upload_complete' on finish. Also updates _upload_jobs for REST polling.
    """
    pipeline = get_pipeline()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logger.error("Upload job %s: cannot open %s", job_id, video_path)
        _set_job(job_id, 0, "error: cannot open video")
        socketio.emit("upload_error", {"message": "Cannot read video file"})
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps_src      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w_src        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_src        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale  = min(1.0, Config.MAX_FRAME_WIDTH / max(w_src, 1))
    out_w  = int(w_src * scale)
    out_h  = int(h_src * scale) + 60   #+60 for HUD panel
    skip   = max(1, Config.FRAME_SKIP)
    out_fps = fps_src / skip

    output_name = f"output_{job_id}.mp4"
    output_path = os.path.join(Config.OUTPUT_FOLDER, output_name)
    fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
    writer  = cv2.VideoWriter(output_path, fourcc, out_fps, (out_w, out_h))

    logger.info("Upload job %s: %d frames, %.1f fps, %dx%d → %dx%d",
                job_id, total_frames, fps_src, w_src, h_src, out_w, out_h)

    frame_idx = 0
    processed = 0

    while True:
        #Honour cancellation from stop_upload SocketIO event
        with _upload_jobs_lock:
            if _upload_jobs.get(job_id, {}).get("status") == "cancelled":
                logger.info("Upload job %s cancelled by client", job_id)
                break

        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % skip != 0:
            continue

        try:
            with app.app_context():
                result = pipeline.process(
                    frame,
                    frame_number=frame_idx,
                    gps=None,
                )
                if _pothole_memory:
                    _pothole_memory.update(
                        result.detections, result.raw_frame, frame_idx,
                        gps=None,
                    )
        except Exception as exc:
            logger.warning("Upload job %s frame %d pipeline error: %s", job_id, frame_idx, exc)
            continue

        writer.write(result.frame)
        processed += 1

        progress = min(99, int(frame_idx / total_frames * 100))
        _set_job(job_id, progress, "processing")

        #Emit preview frame every 8 processed frames to avoid flooding SocketIO
        if processed % 8 == 0:
            socketio.emit("upload_frame", {
                "image":        _encode_frame(result.frame),
                "progress":     progress,
                "fps":          round(result.fps, 1),
                "pothole_count": result.pothole_count,
                "frame_number": frame_idx,
                "total_frames": total_frames,
            })

    cap.release()
    writer.release()

    _set_job(job_id, 100, "done", output_name)

    socketio.emit("upload_complete", {
        "output_file":     output_name,
        "download_url":    f"/outputs/{output_name}",
        "frames_processed": processed,
    })
    logger.info("Upload job %s complete: %d frames → %s", job_id, processed, output_path)


@app.route("/api/stream/webrtc")
def stream_webrtc_info():
    """
    Return the MediaMTX WebRTC WHEP endpoint so the browser can connect.
    The browser JS opens an RTCPeerConnection to this URL.
    """
    host = request.host.split(":")[0]  #RunPod public hostname
    port = Config.MEDIAMTX_WEBRTC_PORT
    whep_url = f"http://{host}:{port}/live/whep"
    return jsonify({
        "whep_url": whep_url,
        "enabled": Config.RTSP_ENABLED and _rtsp_streamer is not None,
    })


@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(Config.OUTPUT_FOLDER, filename)


@app.route("/static/snapshots/<path:filename>")
def serve_snapshot(filename):
    return send_from_directory(Config.SNAPSHOTS_FOLDER, filename)


@app.route("/gps", methods=["POST"])
@app.route("/api/location", methods=["POST"])
def update_location():
    data   = request.get_json(silent=True) or {}
    lat    = data.get("latitude") or data.get("lat")
    lon    = data.get("longitude") or data.get("lng") or data.get("lon")
    source = data.get("source", "browser")

    if lat is not None and lon is not None:
        state.current_gps = (float(lat), float(lon))
        state.gps_source  = source
        return jsonify({"status": "ok", "lat": float(lat), "lng": float(lon),
                        "source": source})
    return jsonify({"error": "Missing lat/lng coordinates"}), 400


@app.route("/api/location", methods=["GET"])
def get_location():
    if state.current_gps:
        return jsonify({"lat": state.current_gps[0], "lng": state.current_gps[1],
                        "source": state.gps_source})
    return jsonify({"lat": None, "lng": None, "source": "none"})


#SocketIO Events───────────

@socketio.on("connect")
def on_connect():
    logger.info("Client connected: %s", request.sid)
    emit("status", {
        "camera_active":        state.camera_active,
        "browser_camera_active": state.browser_camera_active,
        "rtsp_streaming":       _rtsp_streamer is not None and
                                (_rtsp_streamer.is_running if _rtsp_streamer else False),
    })


@socketio.on("disconnect")
def on_disconnect():
    logger.info("Client disconnected: %s", request.sid)


@socketio.on("start_camera")
def on_start_camera(data=None):
    if state.camera_active:
        emit("status", {"camera_active": True, "mode": "server"})
        return

    data       = data or {}
    stream_url = data.get("url") or None   #None → local device (index 0)

    state.camera_active = True
    state.frame_count   = 0
    t = threading.Thread(target=_camera_worker, args=(stream_url,), daemon=True)
    state._camera_thread = t
    t.start()
    emit("status", {"camera_active": True, "mode": "server",
                    "stream_url": stream_url or "device"})


@socketio.on("stop_camera")
def on_stop_camera():
    state.camera_active         = False
    state.browser_camera_active = False
    emit("status", {"camera_active": False, "browser_camera_active": False})


@socketio.on("start_browser_camera")
def on_start_browser_camera():
    state.camera_active         = False
    state.browser_camera_active = True
    state.frame_count           = 0
    emit("status", {"browser_camera_active": True, "mode": "browser"})


@socketio.on("stop_browser_camera")
def on_stop_browser_camera():
    state.browser_camera_active = False
    emit("status", {"browser_camera_active": False})


@socketio.on("browser_frame")
def on_browser_frame(data):
    if not state.browser_camera_active:
        return

    b64 = data.get("image", "")
    if not b64:
        return
    if "," in b64:
        b64 = b64.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(b64)
        nparr     = np.frombuffer(img_bytes, dtype=np.uint8)
        frame     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as exc:
        logger.debug("browser_frame decode error: %s", exc)
        return

    if frame is None or frame.size == 0:
        return

    threading.Thread(
        target=_process_and_emit,
        args=(frame, request.sid),
        daemon=True,
    ).start()


@socketio.on("stop_upload")
def on_stop_upload():
    #Mark any in-progress job as cancelled so the worker loop can check
    with _upload_jobs_lock:
        for jid, job in _upload_jobs.items():
            if job["status"] == "processing":
                job["status"] = "cancelled"
    emit("status", {"upload_processing": False})


@socketio.on("get_map_data")
def on_get_map_data():
    if _db_ops:
        potholes     = _db_ops.get_all_potholes(500)
        gps_potholes = [p for p in potholes if p.get("latitude")]
        emit("map_data", {"potholes": gps_potholes})


@socketio.on("init_pipeline")
def on_init_pipeline():
    try:
        get_pipeline()
        emit("pipeline_ready", {"status": "ready"})
    except Exception as exc:
        emit("pipeline_ready", {"status": "error", "message": str(exc)})


@socketio.on("set_confidence")
def on_set_confidence(data=None):
    """Dynamically update the detection confidence threshold from the UI slider."""
    data = data or {}
    try:
        threshold = float(data.get("threshold", Config.POTHOLE_CONFIDENCE))
        threshold = max(0.1, min(0.95, threshold))
        Config.POTHOLE_CONFIDENCE = threshold
        emit("config_updated", {"confidence_threshold": round(threshold, 2)})
        logger.info("Confidence threshold updated to %.2f", threshold)
    except (TypeError, ValueError) as exc:
        logger.warning("Invalid confidence value: %s", exc)


#Entry Point 

if __name__ == "__main__":
    logger.info("Starting AI Eyes on the Road v2 …")
    logger.info("Dashboard: http://%s:%d", Config.HOST, Config.PORT)
    socketio.run(
        app,
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG,
        use_reloader=False,
        allow_unsafe_werkzeug=True,
    )
