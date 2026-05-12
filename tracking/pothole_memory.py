import logging
import os
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from pipeline.detection import Detection
from tracking.bytetrack_adapter import PotholeByteTracker
from tracking.state_machine import PotholeStateMachine, PotholeState, PotholeTrack

logger = logging.getLogger(__name__)


class PotholeMemory:
    """
    Integrates ByteTrack + PotholeStateMachine + DB persistence.

    Public API unchanged from previous SORT-based version — all callers
    in app.py continue to work without modification.
    """

    def __init__(self, config, db_ops, snapshot_dir: str, on_pothole_added=None):
        self.config            = config
        self.db                = db_ops
        self.snapshot_dir      = snapshot_dir
        self._on_pothole_added = on_pothole_added  #callback(data) fired after every DB write
        os.makedirs(snapshot_dir, exist_ok=True)

        self.tracker = PotholeByteTracker(
            track_high_thresh = getattr(config, "BYTETRACK_HIGH_THRESH",  0.45),
            track_low_thresh  = getattr(config, "BYTETRACK_LOW_THRESH",   0.10),
            new_track_thresh  = getattr(config, "BYTETRACK_NEW_THRESH",   0.25),
            track_buffer      = getattr(config, "TRACKER_MAX_AGE",        30),
            match_thresh      = getattr(config, "BYTETRACK_MATCH_THRESH", 0.40),
            frame_rate        = getattr(config, "BYTETRACK_FRAME_RATE",   8),
        )

        self.state_machine = PotholeStateMachine(
            on_confirm          = self._on_track_confirmed,
            on_mature           = self._on_track_matured,
            confirm_age         = getattr(config, "STATE_CONFIRM_AGE",        None),
            confirm_ema_thresh  = getattr(config, "STATE_CONFIRM_EMA_THRESH", None),
            lost_patience_tent  = getattr(config, "STATE_LOST_PATIENCE_TENT", None),
            lost_patience_conf  = getattr(config, "STATE_LOST_PATIENCE_CONF", None),
        )

        self._snapped: set = set()

    #Public API────────────

    def update(
        self,
        detections:   List[Detection],
        frame_bgr:    np.ndarray,
        frame_number: int,
        gps:          Optional[Tuple[float, float]] = None,
    ) -> List[Detection]:
        """
        Update tracker and state machine with current frame detections.
        Assigns track_id + track_state to each pothole Detection.
        Returns the same list (mutated).
        """
        pothole_dets = [d for d in detections if d.is_pothole]

        if not pothole_dets:
            #Still need to tick state machine for missed-frame counters
            self.state_machine.update(
                active_track_ids = [],
                track_metadata   = {},
                frame_gps        = gps,
            )
            return detections

        #ByteTrack update──
        h, w = frame_bgr.shape[:2]
        tracked = self.tracker.update(pothole_dets, (h, w))

        #Build metadata dict for state machine
        active_ids: List[int]       = []
        meta: Dict[int, dict]       = {}

        for td in tracked:
            tid = td["track_id"]
            active_ids.append(tid)
            meta[tid] = {
                "confidence_ema": td["confidence_ema"],
                "severity_score": td.get("severity_score", 0.0),
                "severity_level": td.get("severity_level", "L1_COSMETIC"),
                "depth_m":        td.get("depth_m", 10.0),
                "road_ratio":     td.get("road_ratio", 1.0),
                "bbox_xyxy":      td["bbox_xyxy"],
            }

        #State machine update
        track_snapshot = self.state_machine.update(
            active_track_ids = active_ids,
            track_metadata   = meta,
            frame_gps        = gps,
        )

        #Annotate detections with track_id and track_state
        for det in pothole_dets:
            #Match detection to track by best IoU
            best_iou, best_tid = 0.0, None
            for td in tracked:
                bx1, by1, bx2, by2 = td["bbox_xyxy"]
                ix1, iy1 = max(det.x1, bx1), max(det.y1, by1)
                ix2, iy2 = min(det.x2, bx2), min(det.y2, by2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                a1 = (det.x2 - det.x1) * (det.y2 - det.y1)
                a2 = (bx2 - bx1) * (by2 - by1)
                union = a1 + a2 - inter
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou, best_tid = iou, td["track_id"]

            if best_tid is not None and best_iou > 0.1:
                det.track_id = best_tid
                track = track_snapshot.get(best_tid)
                if track:
                    det.track_state    = track.state.name.lower()
                    det.severity_score = track.severity_score
                    det.severity_level = track.severity_level

        #Snapshot for newly confirmed tracks
        for tid in active_ids:
            track = track_snapshot.get(tid)
            if track and track.state in (PotholeState.CONFIRMED, PotholeState.MATURE):
                if tid not in self._snapped:
                    det_for_snap = next(
                        (d for d in pothole_dets if d.track_id == tid), None
                    )
                    if det_for_snap:
                        snap = self._save_snapshot(frame_bgr, det_for_snap, tid, track)
                        if snap:
                            track.snapshot_path = snap
                            self._snapped.add(tid)
                            #Link snapshot to existing DB record if already written
                            if track.db_id and self.db:
                                try:
                                    self.db.patch_snapshot(track.db_id, snap)
                                except Exception as e:
                                    logger.warning("patch_snapshot error for track %d: %s", tid, e)

        return detections

    def get_active_tracks(self) -> Dict[int, PotholeTrack]:
        return {
            tid: t for tid, t in self.state_machine._tracks.items()
            if t.state not in (PotholeState.LOST, PotholeState.ARCHIVED)
        }

    def mark_crossed(self, track_id: int):
        track = self.state_machine.get_track(track_id)
        if track and track.db_id and self.db:
            try:
                self.db.mark_crossed(track.db_id)
            except Exception as e:
                logger.warning("mark_crossed DB error: %s", e)

    def reset(self):
        """Reset tracker state between video clips."""
        self.tracker.reset()
        self.state_machine._tracks.clear()
        self._snapped.clear()

    #State machine callbacks───

    def _risk_level(self, depth_m: Optional[float]) -> str:
        if depth_m is None:
            return "far"
        if depth_m < 5:
            return "near"
        if depth_m < 15:
            return "medium"
        return "far"

    def _on_track_confirmed(self, track: PotholeTrack):
        """Called when TENTATIVE → CONFIRMED (age=3). Write to DB immediately so
        fast-moving detections (~0.375s at 8fps) are never missed."""
        if track._db_written or self.db is None:
            return
        try:
            lat = track.gps[0] if track.gps else None
            lon = track.gps[1] if track.gps else None
            ph  = self.db.upsert_pothole(
                confidence     = track.confidence_ema,
                distance_m     = track.depth_m,
                risk_level     = self._risk_level(track.depth_m),
                latitude       = lat,
                longitude      = lon,
                track_id       = track.track_id,
                snapshot_path  = track.snapshot_path,
                source         = "camera",
                severity_score = track.severity_score,
                severity_level = track.severity_level,
                track_state    = "confirmed",
            )
            track.db_id       = ph.id
            track._db_written = True
            logger.info("Track %d written to DB at CONFIRMED (id=%d, severity=%s)",
                        track.track_id, ph.id, track.severity_level)
            if self._on_pothole_added:
                try:
                    self._on_pothole_added({
                        "id":         ph.id,
                        "latitude":   lat,
                        "longitude":  lon,
                        "risk_level": self._risk_level(track.depth_m),
                        "confidence": round(track.confidence_ema, 3),
                    })
                except Exception:
                    pass
        except Exception as e:
            logger.warning("DB persist error for track %d (CONFIRMED): %s", track.track_id, e)

    def _on_track_matured(self, track: PotholeTrack):
        """Called when CONFIRMED → MATURE. Upgrade existing record with better
        GPS / severity data accumulated over 10 frames."""
        if self.db is None:
            return
        try:
            lat = track.gps[0] if track.gps else None
            lon = track.gps[1] if track.gps else None
            if track._db_written and track.db_id:
                self.db.upgrade_to_mature(
                    pothole_id     = track.db_id,
                    confidence     = track.confidence_ema,
                    severity_score = track.severity_score,
                    severity_level = track.severity_level,
                    distance_m     = track.depth_m,
                    snapshot_path  = track.snapshot_path,
                    lat            = lat,
                    lon            = lon,
                    risk_level     = self._risk_level(track.depth_m),
                )
                logger.info("Track %d upgraded to MATURE in DB (id=%d)", track.track_id, track.db_id)
            else:
                ph = self.db.upsert_pothole(
                    confidence     = track.confidence_ema,
                    distance_m     = track.depth_m,
                    risk_level     = self._risk_level(track.depth_m),
                    latitude       = lat,
                    longitude      = lon,
                    track_id       = track.track_id,
                    snapshot_path  = track.snapshot_path,
                    source         = "camera",
                    severity_score = track.severity_score,
                    severity_level = track.severity_level,
                    track_state    = "mature",
                )
                track.db_id       = ph.id
                track._db_written = True
                logger.info("Track %d written to DB at MATURE (id=%d, severity=%s)",
                            track.track_id, ph.id, track.severity_level)
        except Exception as e:
            logger.warning("DB persist error for track %d (MATURE): %s", track.track_id, e)

    #Snapshot 

    def _save_snapshot(
        self,
        frame_bgr:   np.ndarray,
        det:         Detection,
        track_id:    int,
        track:       PotholeTrack,
    ) -> Optional[str]:
        try:
            x1, y1, x2, y2 = det.bbox
            h, w = frame_bgr.shape[:2]
            bw, bh = x2 - x1, y2 - y1

            pad_x = max(40, int(bw * 0.60))
            pad_y = max(40, int(bh * 0.60))
            cx1 = max(0, x1 - pad_x); cy1 = max(0, y1 - pad_y)
            cx2 = min(w, x2 + pad_x); cy2 = min(h, y2 + pad_y)
            crop = frame_bgr[cy1:cy2, cx1:cx2].copy()

            rx1, ry1 = x1 - cx1, y1 - cy1
            rx2, ry2 = x2 - cx1, y2 - cy1
            sev_colors = {
                "L4_CRITICAL": (0, 0, 255),
                "L3_SEVERE":   (0, 80, 255),
                "L2_MODERATE": (0, 165, 255),
                "L1_COSMETIC": (0, 255, 180),
            }
            box_color = sev_colors.get(track.severity_level, (0, 220, 0))
            cv2.rectangle(crop, (rx1, ry1), (rx2, ry2), box_color, 2)

            bar_h = 28
            bar   = np.zeros((bar_h, crop.shape[1], 3), dtype=np.uint8)
            bar[:] = (25, 25, 25)
            ts    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            dist  = f"{track.depth_m:.1f}m" if track.depth_m else "?"
            label = (
                f"ID#{track_id}  conf:{track.confidence_ema:.2f}  "
                f"dist:{dist}  sev:{track.severity_level}  {ts}"
            )
            cv2.putText(bar, label, (6, 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)
            snapshot = np.vstack([bar, crop])

            fname = f"pothole_{track_id}_{int(time.time())}.jpg"
            fpath = os.path.join(self.snapshot_dir, fname)
            cv2.imwrite(fpath, snapshot, [cv2.IMWRITE_JPEG_QUALITY, 95])
            return f"snapshots/{fname}"
        except Exception as e:
            logger.warning("Snapshot save error: %s", e)
            return None
