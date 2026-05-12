import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional

from .models import db, Pothole, DetectionEvent, Report

logger = logging.getLogger(__name__)


def _haversine_distance_m(lat1, lon1, lat2, lon2) -> float:
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class PotholeDB:
    """High-level data-access layer. All methods require Flask app context."""

    def __init__(self, spatial_threshold_m: float = 5.0):
        #Tightened from 10m to 5m — DBSCAN handles larger-scale clustering
        self.spatial_threshold_m = spatial_threshold_m

    #Spatial dedup─────────

    def find_nearby_pothole(
        self,
        lat: Optional[float],
        lon: Optional[float],
    ) -> Optional[Pothole]:
        if lat is None or lon is None:
            return None
        potholes = Pothole.query.filter(
            Pothole.latitude.isnot(None),
            Pothole.longitude.isnot(None),
        ).all()
        for ph in potholes:
            dist = _haversine_distance_m(lat, lon, ph.latitude, ph.longitude)
            if dist <= self.spatial_threshold_m:
                return ph
        return None

    #Upsert (v2 — with severity and track_state)

    def upsert_pothole(
        self,
        confidence:     float,
        bbox:           Optional[tuple] = None,
        distance_m:     Optional[float] = None,
        risk_level:     Optional[str]   = None,
        latitude:       Optional[float] = None,
        longitude:      Optional[float] = None,
        track_id:       Optional[int]   = None,
        snapshot_path:  Optional[str]   = None,
        source:         str = "camera",
        frame_number:   Optional[int]   = None,
        severity_score: float = 0.0,
        severity_level: str   = "L1_COSMETIC",
        track_state:    str   = "confirmed",
        cluster_id:     Optional[str] = None,
    ) -> Pothole:
        existing = self.find_nearby_pothole(latitude, longitude)

        if existing:
            existing.last_seen       = datetime.utcnow()
            existing.detection_count += 1
            existing.confidence      = max(existing.confidence, confidence)
            if distance_m is not None:
                existing.estimated_distance_m = distance_m
            if risk_level:
                existing.risk_level = risk_level
            if snapshot_path:
                existing.snapshot_path = snapshot_path
            if track_id is not None:
                existing.track_id = track_id
            #Update severity to maximum observed
            if severity_score > (existing.severity_score or 0.0):
                existing.severity_score = severity_score
                existing.severity_level = severity_level
            if track_state:
                existing.track_state = track_state
            if cluster_id:
                existing.cluster_id = cluster_id
            pothole = existing
        else:
            pothole = Pothole(
                track_id       = track_id,
                confidence     = confidence,
                estimated_distance_m = distance_m,
                risk_level     = risk_level,
                latitude       = latitude,
                longitude      = longitude,
                source         = source,
                snapshot_path  = snapshot_path,
                severity_score = severity_score,
                severity_level = severity_level,
                track_state    = track_state,
                cluster_id     = cluster_id,
            )
            if bbox and len(bbox) == 4:
                pothole.bbox_x1, pothole.bbox_y1, \
                pothole.bbox_x2, pothole.bbox_y2 = bbox
            db.session.add(pothole)

        try:
            db.session.flush()
            event = DetectionEvent(
                pothole_id   = pothole.id,
                confidence   = confidence,
                distance_m   = distance_m,
                frame_number = frame_number,
                latitude     = latitude,
                longitude    = longitude,
            )
            db.session.add(event)
            db.session.commit()
        except Exception as exc:
            db.session.rollback()
            logger.error("DB upsert failed: %s", exc)
            raise

        return pothole

    #Read─

    def get_all_potholes(self, limit: int = 500) -> List[Dict[str, Any]]:
        rows = Pothole.query.order_by(Pothole.last_seen.desc()).limit(limit).all()
        return [r.to_dict() for r in rows]

    def get_clustered_potholes(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Return pothole records grouped by cluster_id for map rendering.
        Records without cluster_id are returned as singletons.
        """
        from database.geo_clustering import cluster_potholes
        rows    = Pothole.query.filter(
            Pothole.latitude.isnot(None),
            Pothole.longitude.isnot(None),
        ).order_by(Pothole.last_seen.desc()).limit(limit).all()

        records = [r.to_dict() for r in rows]
        clusters = cluster_potholes(records)

        return [
            {
                "cluster_id":    c.cluster_id,
                "center_lat":    c.center_lat,
                "center_lon":    c.center_lon,
                "severity_max":  c.severity_max,
                "severity_avg":  c.severity_avg,
                "severity_level": c.severity_level,
                "count":         c.count,
                "confidence":    c.confidence,
                "member_ids":    c.member_ids,
            }
            for c in clusters
        ]

    def get_pothole_by_id(self, pothole_id: int) -> Optional[Dict[str, Any]]:
        ph = Pothole.query.get(pothole_id)
        return ph.to_dict() if ph else None

    def get_stats(self) -> Dict[str, Any]:
        total    = Pothole.query.count()
        critical = Pothole.query.filter_by(severity_level="L4_CRITICAL").count()
        severe   = Pothole.query.filter_by(severity_level="L3_SEVERE").count()
        moderate = Pothole.query.filter_by(severity_level="L2_MODERATE").count()
        cosmetic = Pothole.query.filter_by(severity_level="L1_COSMETIC").count()
        near     = Pothole.query.filter_by(risk_level="near").count()
        crossed  = Pothole.query.filter_by(crossed=True).count()
        reported = Pothole.query.filter_by(reported=True).count()
        return {
            "total":    total,
            "critical": critical,
            "severe":   severe,
            "moderate": moderate,
            "cosmetic": cosmetic,
            "near":     near,
            "crossed":  crossed,
            "reported": reported,
        }

    #Mutations─────────────

    def mark_crossed(self, pothole_id: int) -> bool:
        ph = Pothole.query.get(pothole_id)
        if ph:
            ph.crossed = True
            db.session.commit()
            return True
        return False

    def patch_snapshot(self, pothole_id: int, snapshot_path: str):
        """Set snapshot_path on a record only when it was previously unset."""
        ph = Pothole.query.get(pothole_id)
        if ph and not ph.snapshot_path:
            ph.snapshot_path = snapshot_path
            db.session.commit()

    def upgrade_to_mature(
        self,
        pothole_id:     int,
        confidence:     float,
        severity_score: float,
        severity_level: str,
        distance_m:     Optional[float],
        snapshot_path:  Optional[str],
        lat:            Optional[float],
        lon:            Optional[float],
        risk_level:     Optional[str],
    ):
        """Upgrade an existing CONFIRMED record with mature-state data."""
        ph = Pothole.query.get(pothole_id)
        if not ph:
            return
        ph.track_state = "mature"
        ph.confidence  = max(ph.confidence, confidence)
        if severity_score > (ph.severity_score or 0.0):
            ph.severity_score = severity_score
            ph.severity_level = severity_level
        if distance_m is not None:
            ph.estimated_distance_m = distance_m
        if risk_level:
            ph.risk_level = risk_level
        if snapshot_path and not ph.snapshot_path:
            ph.snapshot_path = snapshot_path
        if lat is not None and ph.latitude is None:
            ph.latitude  = lat
            ph.longitude = lon
        ph.last_seen = datetime.utcnow()
        db.session.commit()

    def update_cluster_id(self, pothole_id: int, cluster_id: str):
        ph = Pothole.query.get(pothole_id)
        if ph:
            ph.cluster_id = cluster_id
            db.session.commit()

    def delete_pothole(self, pothole_id: int) -> bool:
        ph = Pothole.query.get(pothole_id)
        if ph:
            db.session.delete(ph)
            db.session.commit()
            return True
        return False

    #Reports ─

    def create_report(
        self,
        pothole_id:       int,
        recipient_email:  str,
        subject:          str,
        report_type:      str = "initial",
        status:           str = "sent",
        reference_number: Optional[str] = None,
        error_message:    Optional[str] = None,
    ) -> Report:
        report = Report(
            pothole_id       = pothole_id,
            recipient_email  = recipient_email,
            subject          = subject,
            report_type      = report_type,
            status           = status,
            reference_number = reference_number,
            error_message    = error_message,
        )
        db.session.add(report)
        ph = Pothole.query.get(pothole_id)
        if ph and status in ("sent", "dry_run"):
            ph.reported = True
        db.session.commit()
        return report

    def get_unreported_potholes(self) -> List[Pothole]:
        return Pothole.query.filter_by(reported=False).all()

    def get_unresolved_reports(self, days_old: int = 7) -> List[Report]:
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(days=days_old)
        return Report.query.filter(
            Report.status == "sent",
            Report.submitted_at <= cutoff,
        ).all()
