import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class PotholeState(Enum):
    TENTATIVE = auto()
    CONFIRMED = auto()
    MATURE    = auto()
    LOST      = auto()
    ARCHIVED  = auto()


@dataclass
class PotholeTrack:
    """Per-track state, updated every frame."""
    track_id:       int
    state:          PotholeState = PotholeState.TENTATIVE
    track_age:      int          = 0     #frames since first detection
    frames_missed:  int          = 0
    confidence_ema: float        = 0.0
    severity_score: float        = 0.0
    severity_level: str          = "L1_COSMETIC"
    depth_m:        float        = 0.0
    road_ratio:     float        = 1.0
    gps:            Optional[tuple] = None
    gps_locked:     bool         = False
    first_seen:     float        = field(default_factory=time.monotonic)
    last_seen:      float        = field(default_factory=time.monotonic)
    bbox_xyxy:      Optional[list] = None
    db_id:          Optional[int]  = None
    snapshot_path:  Optional[str]  = None
    _db_written:    bool           = False


class PotholeStateMachine:
    """
    Manages the state of all active pothole tracks.

    Threading: all methods are called from the pipeline thread.
    No locking is needed — update() is the single writer.
    """

    #Class-level defaults — can be overridden per-instance via __init__ kwargs
    CONFIRM_AGE          = 2     #2 frames @ 8 FPS = ~250 ms (was 3)
    CONFIRM_EMA_THRESH   = 0.25  #below POTHOLE_CONFIDENCE (0.30) so all valid detections can confirm (was 0.35)
    MATURE_AGE           = 10
    LOST_PATIENCE_TENT   = 5     #tolerate 5 dropped frames before losing a TENTATIVE track (was 3)
    LOST_PATIENCE_CONF   = 8     #(was 5)
    ARCHIVE_PATIENCE     = 30
    PURGE_AFTER_ARCHIVE_S = 300
    EMA_ALPHA            = 0.35

    def __init__(self, on_confirm=None, on_mature=None,
                 confirm_age=None, confirm_ema_thresh=None,
                 lost_patience_tent=None, lost_patience_conf=None):
        """
        Args:
            on_confirm:          callback(track) fired when TENTATIVE→CONFIRMED
            on_mature:           callback(track) fired when CONFIRMED→MATURE
            confirm_age:         override CONFIRM_AGE (frames)
            confirm_ema_thresh:  override CONFIRM_EMA_THRESH
            lost_patience_tent:  override LOST_PATIENCE_TENT
            lost_patience_conf:  override LOST_PATIENCE_CONF
        """
        self._tracks: Dict[int, PotholeTrack] = {}
        self._on_confirm = on_confirm
        self._on_mature  = on_mature
        #Allow per-instance overrides so config can tune without subclassing
        if confirm_age         is not None: self.CONFIRM_AGE         = confirm_age
        if confirm_ema_thresh  is not None: self.CONFIRM_EMA_THRESH  = confirm_ema_thresh
        if lost_patience_tent  is not None: self.LOST_PATIENCE_TENT  = lost_patience_tent
        if lost_patience_conf  is not None: self.LOST_PATIENCE_CONF  = lost_patience_conf

    def update(
        self,
        active_track_ids: List[int],
        track_metadata:   Dict[int, dict],
        frame_gps:        Optional[tuple] = None,
    ) -> Dict[int, PotholeTrack]:
        """
        Update all tracks with current frame's tracker output.

        Args:
            active_track_ids: IDs returned by ByteTrack this frame
            track_metadata:   {track_id: {confidence_ema, severity_score,
                                          severity_level, depth_m, road_ratio,
                                          bbox_xyxy}}
            frame_gps:        (lat, lon) or None

        Returns:
            Current snapshot of all non-purged tracks.
        """
        active_set = set(active_track_ids)

        #Update active tracks
        for tid in active_track_ids:
            meta = track_metadata.get(tid, {})

            if tid not in self._tracks:
                self._tracks[tid] = PotholeTrack(track_id=tid)

            track = self._tracks[tid]
            track.track_age      += 1
            track.frames_missed   = 0
            track.last_seen       = time.monotonic()

            #EMA confidence accumulation
            raw_conf = float(meta.get("confidence_ema", 0.0))
            if track.track_age == 1:
                track.confidence_ema = raw_conf
            else:
                track.confidence_ema = (
                    self.EMA_ALPHA * raw_conf
                    + (1.0 - self.EMA_ALPHA) * track.confidence_ema
                )

            track.severity_score = max(
                track.severity_score,
                float(meta.get("severity_score", 0.0))
            )
            track.severity_level = meta.get("severity_level", track.severity_level)
            track.depth_m        = float(meta.get("depth_m", track.depth_m))
            track.road_ratio     = float(meta.get("road_ratio", track.road_ratio))
            track.bbox_xyxy      = meta.get("bbox_xyxy", track.bbox_xyxy)

            if frame_gps and not track.gps_locked:
                track.gps = frame_gps

            self._transition_active(track)

        #Increment missed counter for inactive tracks─────
        for tid, track in self._tracks.items():
            if tid not in active_set and track.state not in (
                PotholeState.LOST, PotholeState.ARCHIVED
            ):
                track.frames_missed += 1
                self._transition_missed(track)

        #Purge old archived tracks─
        now = time.monotonic()
        to_purge = [
            tid for tid, t in self._tracks.items()
            if t.state == PotholeState.ARCHIVED
            and (now - t.last_seen) > self.PURGE_AFTER_ARCHIVE_S
        ]
        for tid in to_purge:
            del self._tracks[tid]

        return dict(self._tracks)

    def get_confirmed_tracks(self) -> List[PotholeTrack]:
        return [
            t for t in self._tracks.values()
            if t.state in (PotholeState.CONFIRMED, PotholeState.MATURE)
        ]

    def get_track(self, track_id: int) -> Optional[PotholeTrack]:
        return self._tracks.get(track_id)

    #Transitions───────

    def _transition_active(self, track: PotholeTrack):
        if track.state == PotholeState.TENTATIVE:
            if (track.track_age >= self.CONFIRM_AGE
                    and track.confidence_ema >= self.CONFIRM_EMA_THRESH):
                track.state = PotholeState.CONFIRMED
                track.gps_locked = track.gps is not None
                logger.debug("Track %d: TENTATIVE → CONFIRMED (age=%d ema=%.3f)",
                             track.track_id, track.track_age, track.confidence_ema)
                if self._on_confirm:
                    try:
                        self._on_confirm(track)
                    except Exception as e:
                        logger.warning("on_confirm callback error: %s", e)

        elif track.state == PotholeState.CONFIRMED:
            if track.track_age >= self.MATURE_AGE:
                track.state = PotholeState.MATURE
                logger.info("Track %d: CONFIRMED → MATURE (severity=%s depth=%.1fm)",
                            track.track_id, track.severity_level, track.depth_m)
                if self._on_mature:
                    try:
                        self._on_mature(track)
                    except Exception as e:
                        logger.warning("on_mature callback error: %s", e)

    def _transition_missed(self, track: PotholeTrack):
        patience = {
            PotholeState.TENTATIVE: self.LOST_PATIENCE_TENT,
            PotholeState.CONFIRMED: self.LOST_PATIENCE_CONF,
            PotholeState.MATURE:    self.ARCHIVE_PATIENCE,
        }.get(track.state, 5)

        if track.frames_missed > patience:
            if track.state == PotholeState.MATURE:
                track.state = PotholeState.ARCHIVED
                logger.debug("Track %d: MATURE → ARCHIVED", track.track_id)
            else:
                track.state = PotholeState.LOST
                logger.debug("Track %d: → LOST (missed=%d)", track.track_id, track.frames_missed)
