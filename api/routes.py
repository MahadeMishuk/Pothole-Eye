"""
REST API Blueprint

Endpoints:
  GET  /api/potholes              — list all potholes (paginated, flat JSON)
  GET  /api/potholes/<id>         — single pothole detail
  POST /api/potholes/<id>/report  — trigger MDOT report
  POST /api/potholes/<id>/mark-crossed
  DELETE /api/potholes/<id>       — delete record
  GET  /api/map-data              — GeoJSON FeatureCollection (Mapbox source)
  GET  /api/mapbox-token          — serve Mapbox public token securely
  POST /api/location              — ingest GPS from browser / IP fallback
  GET  /api/location              — return current server GPS state
  GET  /api/gps-status            — detailed GPS status + source info
  GET  /api/stats                 — detection statistics
  GET  /api/alerts                — recent alert history
  GET  /api/performance           — CPU / memory metrics
  GET  /api/health                — liveness check
"""
import logging
import os
import time
from functools import wraps

import psutil
from flask import Blueprint, jsonify, request, current_app

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__, url_prefix="/api")


#Helpers─

def _db():
    return current_app.extensions.get("pothole_db")


def _reporter():
    return current_app.extensions.get("pothole_reporter")


def _alert_mgr():
    return current_app.extensions.get("alert_manager")


def handle_errors(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as exc:
            logger.error("API error in %s: %s", f.__name__, exc, exc_info=True)
            return jsonify({"error": str(exc)}), 500
    return wrapper


#Mapbox Token Endpoint────

@api_bp.route("/mapbox-token", methods=["GET"])
def mapbox_token():
    """
    Serves the Mapbox public access token from the server environment.

    Why an endpoint instead of hardcoding:
      - Token lives only in .env / server environment — not in committed JS files.
      - For production: add auth middleware here to restrict who can fetch it.
      - Mapbox pk.* tokens are public by design but should still be env-managed
        so they can be rotated without code changes.
    """
    from config import Config
    token = Config.MAPBOX_ACCESS_TOKEN
    if not token:
        return jsonify({"error": "MAPBOX_ACCESS_TOKEN not configured"}), 503
    return jsonify({
        "token": token,
        "style": Config.MAPBOX_STYLE,
        "default_lat": Config.DEFAULT_LAT,
        "default_lng": Config.DEFAULT_LNG,
        "default_zoom": Config.DEFAULT_ZOOM,
    })


#Pothole CRUD────────────

@api_bp.route("/potholes", methods=["GET"])
@handle_errors
def list_potholes():
    """
    Flat JSON array format — compatible with direct Mapbox source ingestion.

    ?limit=N  (default 200, max 1000)
    ?format=flat  → array of {id, lat, lng, confidence, timestamp, distance, ...}
    ?format=full  → object with {potholes: [...], total: N}  (default)
    """
    db = _db()
    if not db:
        return jsonify({"potholes": [], "total": 0})

    limit = min(int(request.args.get("limit", 200)), 1000)
    fmt   = request.args.get("format", "full")
    rows  = db.get_all_potholes(limit=limit)

    if fmt == "flat":
        flat = [
            {
                "id":         r["id"],
                "lat":        r.get("latitude"),
                "lng":        r.get("longitude"),
                "confidence": r.get("confidence", 0.0),
                "timestamp":  r.get("last_seen") or r.get("first_seen"),
                "distance":   r.get("estimated_distance_m"),
                "risk_level": r.get("risk_level", "far"),
                "detection_count": r.get("detection_count", 1),
                "snapshot_path":   r.get("snapshot_path"),
                "crossed":    r.get("crossed", False),
                "reported":   r.get("reported", False),
            }
            for r in rows
            if r.get("latitude") is not None   #only GPS-tagged records
        ]
        return jsonify(flat)

    return jsonify({"potholes": rows, "total": len(rows)})


@api_bp.route("/potholes/<int:pid>", methods=["GET"])
@handle_errors
def get_pothole(pid):
    db = _db()
    ph = db.get_pothole_by_id(pid) if db else None
    if not ph:
        return jsonify({"error": "Not found"}), 404
    return jsonify(ph)


@api_bp.route("/potholes/<int:pid>", methods=["DELETE"])
@handle_errors
def delete_pothole(pid):
    db = _db()
    if db and db.delete_pothole(pid):
        return jsonify({"deleted": pid})
    return jsonify({"error": "Not found"}), 404


@api_bp.route("/potholes/<int:pid>/mark-crossed", methods=["POST"])
@handle_errors
def mark_crossed(pid):
    db = _db()
    if db and db.mark_crossed(pid):
        return jsonify({"success": True, "pothole_id": pid})
    return jsonify({"error": "Not found"}), 404


@api_bp.route("/potholes/<int:pid>/location", methods=["PATCH"])
@handle_errors
def assign_location(pid):
    """Assign GPS coordinates to a pothole that was detected without GPS."""
    from database.models import db as _sqldb, Pothole
    data = request.get_json() or {}
    lat  = data.get("lat")
    lng  = data.get("lng")
    if lat is None or lng is None:
        return jsonify({"error": "lat and lng required"}), 400
    ph = Pothole.query.get(pid)
    if not ph:
        return jsonify({"error": "Not found"}), 404
    ph.latitude  = float(lat)
    ph.longitude = float(lng)
    _sqldb.session.commit()
    return jsonify({"id": pid, "lat": float(lat), "lng": float(lng)})


#MDOT Reporting───────────

@api_bp.route("/potholes/<int:pid>/report", methods=["POST"])
@handle_errors
def report_pothole(pid):
    db = _db()
    reporter = _reporter()
    ph = db.get_pothole_by_id(pid) if db else None
    if not ph:
        return jsonify({"error": "Pothole not found"}), 404
    if not reporter:
        return jsonify({"error": "Reporter not configured"}), 503
    result = reporter.submit_initial_report(ph)
    return jsonify(result)


@api_bp.route("/potholes/<int:pid>/personal-report", methods=["POST"])
@handle_errors
def personal_report(pid):
    """Send personal alert email (with snapshot) to the configured personal address."""
    from config import Config
    db       = _db()
    reporter = _reporter()
    ph       = db.get_pothole_by_id(pid) if db else None
    if not ph:
        return jsonify({"error": "Pothole not found"}), 404
    if not reporter:
        return jsonify({"error": "Reporter not configured"}), 503
    to_email = Config.PERSONAL_REPORT_EMAIL
    result   = reporter.send_personal_alert(ph, to_email)
    return jsonify(result)


@api_bp.route("/potholes/<int:pid>/report-preview", methods=["GET"])
@handle_errors
def report_preview(pid):
    db = _db()
    reporter = _reporter()
    ph = db.get_pothole_by_id(pid) if db else None
    if not ph:
        return jsonify({"error": "Not found"}), 404
    if not reporter:
        return jsonify({"error": "Reporter not configured"}), 503
    return jsonify({"report_text": reporter.generate_report_text(ph)})


#Map Data (GeoJSON)───────

@api_bp.route("/map-data", methods=["GET"])
@handle_errors
def map_data():
    """
    GeoJSON FeatureCollection — consumed directly by Mapbox GL JS as a source.

    Each feature's geometry is a Point [lng, lat].
    Properties carry all pothole metadata needed for popup rendering.
    Only potholes that have GPS coordinates are included.
    """
    db = _db()
    potholes = db.get_all_potholes(500) if db else []

    features = []
    for ph in potholes:
        lat = ph.get("latitude")
        lon = ph.get("longitude")
        if lat is None or lon is None:
            continue
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "id":                 ph["id"],
                "confidence":         ph.get("confidence", 0.0),
                "risk_level":         ph.get("risk_level", "far"),
                "first_seen":         ph.get("first_seen"),
                "last_seen":          ph.get("last_seen"),
                "detection_count":    ph.get("detection_count", 1),
                "estimated_distance_m": ph.get("estimated_distance_m"),
                "snapshot_path":      ph.get("snapshot_path"),
                "crossed":            ph.get("crossed", False),
                "reported":           ph.get("reported", False),
            },
        })

    return jsonify({
        "type":     "FeatureCollection",
        "features": features,
        "count":    len(features),
    })


#Stats & Metrics──────────

@api_bp.route("/stats", methods=["GET"])
@handle_errors
def get_stats():
    db = _db()
    stats = db.get_stats() if db else {}
    return jsonify(stats)


@api_bp.route("/performance", methods=["GET"])
@handle_errors
def get_performance():
    proc = psutil.Process(os.getpid())
    mem  = proc.memory_info()
    return jsonify({
        "cpu_percent":    proc.cpu_percent(interval=0.1),
        "memory_rss_mb":  round(mem.rss / 1024 / 1024, 1),
        "memory_vms_mb":  round(mem.vms / 1024 / 1024, 1),
        "threads":        proc.num_threads(),
        "uptime_s":       round(time.time() - proc.create_time(), 1),
    })


#Alerts

@api_bp.route("/alerts", methods=["GET"])
@handle_errors
def get_alerts():
    mgr = _alert_mgr()
    alerts = mgr.get_recent_alerts(20) if mgr else []
    return jsonify({"alerts": alerts})


#GPS Status ─

@api_bp.route("/gps-status", methods=["GET"])
@handle_errors
def gps_status():
    """
    Returns current GPS state: source, coordinates, and whether location
    services are active. Useful for the dashboard to display GPS health.
    """
    from app import state as app_state
    gps = app_state.current_gps
    return jsonify({
        "available":  gps is not None,
        "lat":        gps[0] if gps else None,
        "lng":        gps[1] if gps else None,
        "source":     app_state.gps_source,
        "camera_mode": (
            "browser" if app_state.browser_camera_active
            else "server" if app_state.camera_active
            else "idle"
        ),
    })


#Health

@api_bp.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "timestamp": time.time()})
