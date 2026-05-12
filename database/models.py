"""
Database Models — SQLAlchemy ORM schema (v2)

New columns vs v1:
  severity_score  — float [0,1] from PerceptionFusion
  severity_level  — str  (L1_COSMETIC | L2_MODERATE | L3_SEVERE | L4_CRITICAL)
  track_state     — str  (tentative | confirmed | mature | archived)
  cluster_id      — str  (DBSCAN cluster identifier)
"""
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Pothole(db.Model):
    __tablename__ = "potholes"

    id              = db.Column(db.Integer, primary_key=True, autoincrement=True)
    track_id        = db.Column(db.Integer, nullable=True, index=True)

    #Temporal
    first_seen      = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    last_seen       = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    detection_count = db.Column(db.Integer, default=1, nullable=False)

    #Spatial
    latitude        = db.Column(db.Float, nullable=True)
    longitude       = db.Column(db.Float, nullable=True)

    #Detection
    confidence             = db.Column(db.Float,   nullable=False)
    estimated_distance_m   = db.Column(db.Float,   nullable=True)
    risk_level             = db.Column(db.String(10), nullable=True)
    source                 = db.Column(db.String(20), default="camera")

    #Severity (new in v2)
    severity_score  = db.Column(db.Float,   nullable=True, default=0.0)
    severity_level  = db.Column(db.String(20), nullable=True, default="L1_COSMETIC")

    #State machine (new in v2)
    track_state     = db.Column(db.String(20), nullable=True, default="confirmed")

    #Geo cluster (new in v2)
    cluster_id      = db.Column(db.String(64), nullable=True, index=True)

    #Snapshot
    snapshot_path   = db.Column(db.String(256), nullable=True)

    #Status flags
    crossed         = db.Column(db.Boolean, default=False, nullable=False)
    reported        = db.Column(db.Boolean, default=False, nullable=False)
    repaired        = db.Column(db.Boolean, default=False, nullable=False)

    #Bounding box (normalised 0–1)
    bbox_x1 = db.Column(db.Float, nullable=True)
    bbox_y1 = db.Column(db.Float, nullable=True)
    bbox_x2 = db.Column(db.Float, nullable=True)
    bbox_y2 = db.Column(db.Float, nullable=True)

    events  = db.relationship("DetectionEvent", backref="pothole", lazy="dynamic")
    reports = db.relationship("Report",          backref="pothole", lazy="dynamic")

    def to_dict(self):
        return {
            "id":                   self.id,
            "track_id":             self.track_id,
            "first_seen":           self.first_seen.isoformat() if self.first_seen else None,
            "last_seen":            self.last_seen.isoformat()  if self.last_seen  else None,
            "detection_count":      self.detection_count,
            "latitude":             self.latitude,
            "longitude":            self.longitude,
            "confidence":           round(self.confidence, 3),
            "estimated_distance_m": self.estimated_distance_m,
            "risk_level":           self.risk_level,
            "severity_score":       round(self.severity_score or 0.0, 3),
            "severity_level":       self.severity_level or "L1_COSMETIC",
            "track_state":          self.track_state or "confirmed",
            "cluster_id":           self.cluster_id,
            "source":               self.source,
            "snapshot_path":        self.snapshot_path,
            "crossed":              self.crossed,
            "reported":             self.reported,
            "repaired":             self.repaired,
        }

    def __repr__(self):
        return (f"<Pothole id={self.id} sev={self.severity_level} "
                f"conf={self.confidence:.2f} loc=({self.latitude},{self.longitude})>")


class DetectionEvent(db.Model):
    __tablename__ = "detection_events"

    id          = db.Column(db.Integer, primary_key=True, autoincrement=True)
    pothole_id  = db.Column(db.Integer, db.ForeignKey("potholes.id"),
                            nullable=False, index=True)
    timestamp   = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    confidence  = db.Column(db.Float, nullable=False)
    distance_m  = db.Column(db.Float, nullable=True)
    frame_number = db.Column(db.Integer, nullable=True)
    latitude    = db.Column(db.Float, nullable=True)
    longitude   = db.Column(db.Float, nullable=True)

    def to_dict(self):
        return {
            "id":           self.id,
            "pothole_id":   self.pothole_id,
            "timestamp":    self.timestamp.isoformat(),
            "confidence":   round(self.confidence, 3),
            "distance_m":   self.distance_m,
            "frame_number": self.frame_number,
        }


class Report(db.Model):
    __tablename__ = "reports"

    id               = db.Column(db.Integer, primary_key=True, autoincrement=True)
    pothole_id       = db.Column(db.Integer, db.ForeignKey("potholes.id"), nullable=False)
    submitted_at     = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    report_type      = db.Column(db.String(20), default="initial")
    recipient_email  = db.Column(db.String(256), nullable=False)
    subject          = db.Column(db.String(256), nullable=True)
    status           = db.Column(db.String(20), default="sent")
    error_message    = db.Column(db.Text, nullable=True)
    reference_number = db.Column(db.String(64), nullable=True)

    def to_dict(self):
        return {
            "id":               self.id,
            "pothole_id":       self.pothole_id,
            "submitted_at":     self.submitted_at.isoformat(),
            "report_type":      self.report_type,
            "recipient_email":  self.recipient_email,
            "status":           self.status,
            "reference_number": self.reference_number,
        }
