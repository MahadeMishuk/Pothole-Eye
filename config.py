import os
from pathlib import Path

BASE_DIR = Path(__file__).parent


class Config:

    SECRET_KEY = os.environ.get("SECRET_KEY", "pothole-detection-secret-2024")
    DEBUG = os.environ.get("DEBUG", "False").lower() == "true"
    HOST = os.environ.get("HOST", "0.0.0.0")
    PORT = int(os.environ.get("PORT", "5000"))

    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL", f"sqlite:///{BASE_DIR}/database/potholes.db"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    DETECTION_BACKEND = os.environ.get("DETECTION_BACKEND", "rtdetr")
    RTDETR_MODEL = os.environ.get("RTDETR_MODEL", "rtdetr-l.pt")
    YOLO_MODEL = os.environ.get("YOLO_MODEL", "yolov8m.pt")
    POTHOLE_MODEL_PATH = os.environ.get(
        "POTHOLE_MODEL_PATH", str(BASE_DIR / "models" / "pothole_yolov8.pt")
    )

    SEGMENTATION_BACKEND = os.environ.get("SEGMENTATION_BACKEND", "yolov8seg")
    YOLOSEG_MODEL = os.environ.get("YOLOSEG_MODEL", "yolov8l-seg.pt")
    USE_SEGMENTATION = os.environ.get("USE_SEGMENTATION", "True").lower() == "true"

    DEPTH_BACKEND = os.environ.get("DEPTH_BACKEND", "depthanything")
    DEPTH_ANYTHING_MODEL = os.environ.get(
        "DEPTH_ANYTHING_MODEL",
        "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
    )
    USE_DEPTH_MIDAS = os.environ.get("USE_DEPTH_MIDAS", "True").lower() == "true"

    DETECTION_CONFIDENCE = float(os.environ.get("DETECTION_CONFIDENCE", "0.25"))
    POTHOLE_CONFIDENCE = float(os.environ.get("POTHOLE_CONFIDENCE", "0.25"))
    NMS_THRESHOLD = float(os.environ.get("NMS_THRESHOLD", "0.45"))
    POTHOLE_MIN_AREA = int(os.environ.get("POTHOLE_MIN_AREA", "800"))
    POTHOLE_MAX_AREA = int(os.environ.get("POTHOLE_MAX_AREA", "50000"))

    NEAR_DISTANCE = float(os.environ.get("NEAR_DISTANCE", "5.0"))
    MEDIUM_DISTANCE = float(os.environ.get("MEDIUM_DISTANCE", "15.0"))
    FOCAL_LENGTH_PX = float(os.environ.get("FOCAL_LENGTH_PX", "800.0"))
    POTHOLE_REAL_WIDTH_M = float(os.environ.get("POTHOLE_REAL_WIDTH_M", "0.5"))

    FUSION_ROAD_RATIO_MIN = float(os.environ.get("FUSION_ROAD_RATIO_MIN", "0.35"))

    ALERT_COOLDOWN_SEC = float(os.environ.get("ALERT_COOLDOWN_SEC", "3.0"))
    ALERT_DISTANCE_THRESHOLD = float(
        os.environ.get("ALERT_DISTANCE_THRESHOLD", "5.0")
    )

    BYTETRACK_HIGH_THRESH = float(
        os.environ.get("BYTETRACK_HIGH_THRESH", "0.45")
    )
    BYTETRACK_LOW_THRESH = float(
        os.environ.get("BYTETRACK_LOW_THRESH", "0.10")
    )
    BYTETRACK_NEW_THRESH = float(
        os.environ.get("BYTETRACK_NEW_THRESH", "0.25")
    )
    BYTETRACK_MATCH_THRESH = float(
        os.environ.get("BYTETRACK_MATCH_THRESH", "0.40")
    )
    TRACKER_MAX_AGE = int(os.environ.get("TRACKER_MAX_AGE", "30"))
    BYTETRACK_FRAME_RATE = int(os.environ.get("BYTETRACK_FRAME_RATE", "8"))

    TRACKER_MIN_HITS = int(os.environ.get("TRACKER_MIN_HITS", "2"))
    TRACKER_IOU_THRESHOLD = float(
        os.environ.get("TRACKER_IOU_THRESHOLD", "0.3")
    )

    SPATIAL_DEDUP_THRESHOLD_M = float(
        os.environ.get("SPATIAL_DEDUP_THRESHOLD_M", "5.0")
    )

    CELERY_BROKER_URL = os.environ.get(
        "CELERY_BROKER_URL", "redis://redis:6379/0"
    )
    CELERY_RESULT_BACKEND = os.environ.get(
        "CELERY_RESULT_BACKEND", "redis://redis:6379/1"
    )

    RTSP_ENABLED = os.environ.get("RTSP_ENABLED", "True").lower() == "true"
    RTSP_URL = os.environ.get("RTSP_URL", "rtsp://mediamtx:8554/live")
    MEDIAMTX_HOST = os.environ.get("MEDIAMTX_HOST", "mediamtx")
    MEDIAMTX_WEBRTC_PORT = int(
        os.environ.get("MEDIAMTX_WEBRTC_PORT", "8889")
    )
    STREAM_WIDTH = int(os.environ.get("STREAM_WIDTH", "1280"))
    STREAM_HEIGHT = int(os.environ.get("STREAM_HEIGHT", "720"))
    STREAM_FPS = int(os.environ.get("STREAM_FPS", "25"))

    MAPBOX_ACCESS_TOKEN = os.environ.get("MAPBOX_ACCESS_TOKEN", "")
    MAPBOX_STYLE = os.environ.get(
        "MAPBOX_STYLE", "mapbox://styles/mapbox/dark-v11"
    )
    DEFAULT_LAT = float(os.environ.get("DEFAULT_LAT", "39.2904"))
    DEFAULT_LNG = float(os.environ.get("DEFAULT_LNG", "-76.6122"))
    DEFAULT_ZOOM = float(os.environ.get("DEFAULT_ZOOM", "13"))

    MDOT_REPORT_EMAIL = os.environ.get(
        "MDOT_REPORT_EMAIL", "SHA.CustomerService@maryland.gov"
    )
    PERSONAL_REPORT_EMAIL = os.environ.get(
        "PERSONAL_REPORT_EMAIL", "hasanmahade237@gmail.com"
    )
    REPORT_FROM_EMAIL = os.environ.get("REPORT_FROM_EMAIL", "")
    SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
    SMTP_USERNAME = os.environ.get("SMTP_USERNAME", "")
    SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
    REPORT_FOLLOWUP_DAYS = int(
        os.environ.get("REPORT_FOLLOWUP_DAYS", "7")
    )

    FRAME_SKIP = int(os.environ.get("FRAME_SKIP", "1"))
    MAX_FRAME_WIDTH = int(os.environ.get("MAX_FRAME_WIDTH", "960"))
    JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "75"))

    UPLOAD_FOLDER = str(BASE_DIR / "uploads")
    OUTPUT_FOLDER = str(BASE_DIR / "outputs")
    SNAPSHOTS_FOLDER = str(BASE_DIR / "static" / "snapshots")
    MODELS_FOLDER = str(BASE_DIR / "models")
    MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "500"))
    MAX_CONTENT_LENGTH = MAX_UPLOAD_MB * 1024 * 1024
    ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm", "m4v"}

    COCO_ROAD_CLASSES = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
        0: "person",
        1: "bicycle",
        9: "traffic light",
        11: "stop sign",
    }

    CITYSCAPES_ROAD_CLASS = 0
    CITYSCAPES_SIDEWALK_CLASS = 1

    USE_LANE_DETECTION = (
        os.environ.get("USE_LANE_DETECTION", "True").lower() == "true"
    )
    LANE_MASK_COMBINE = os.environ.get("LANE_MASK_COMBINE", "union")
    SHOW_LANE_OVERLAY = (
        os.environ.get("SHOW_LANE_OVERLAY", "True").lower() == "true"
    )

    SHOW_ROAD_MASK = os.environ.get("SHOW_ROAD_MASK", "False").lower() == "true"
    ROAD_MASK_ALPHA = float(os.environ.get("ROAD_MASK_ALPHA", "0.25"))

    @classmethod
    def init_dirs(cls):
        for d in [
            cls.UPLOAD_FOLDER,
            cls.OUTPUT_FOLDER,
            cls.SNAPSHOTS_FOLDER,
            cls.MODELS_FOLDER,
        ]:
            Path(d).mkdir(parents=True, exist_ok=True)


config = Config()