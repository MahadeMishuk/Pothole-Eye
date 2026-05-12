import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Deque

logger = logging.getLogger(__name__)


@dataclass
class AlertEvent:
    timestamp: float
    risk_level: str
    distance_m: Optional[float]
    message: str
    pothole_count: int


class AlertManager:
    """
    Manages alert state, throttling, and notification dispatch.
    Audio is handled client-side (Web Audio API) via SocketIO signal.
    """

    #Risk level priority
    _PRIORITY = {"near": 3, "medium": 2, "far": 1}

    def __init__(self, config, socketio=None):
        self.config = config
        self.socketio = socketio
        self._cooldowns: dict[str, float] = {}   #risk_level → last alert time
        self._history: Deque[AlertEvent] = deque(maxlen=100)
        self._active_alert: Optional[str] = None  #current highest risk level

    #Public API────────

    def process(self, detections: list) -> Optional[AlertEvent]:
        """
        Evaluate detections, trigger alert if warranted.

        Args:
            detections: List of Detection objects from pipeline

        Returns:
            AlertEvent if a new alert was triggered, else None
        """
        potholes = [d for d in detections if getattr(d, "is_pothole", False)]
        if not potholes:
            self._active_alert = None
            return None

        #Find highest risk pothole
        highest = max(
            potholes,
            key=lambda d: (self._PRIORITY.get(d.risk_level or "far", 1),
                           -(d.distance_m or 999))
        )

        risk = highest.risk_level or "far"
        dist = highest.distance_m

        #Only alert for near/medium
        if risk == "far":
            return None

        #Throttle
        now = time.time()
        last = self._cooldowns.get(risk, 0.0)
        cooldown = self.config.ALERT_COOLDOWN_SEC
        if now - last < cooldown:
            return None

        self._cooldowns[risk] = now
        self._active_alert = risk

        msg = self._build_message(risk, dist, len(potholes))
        event = AlertEvent(
            timestamp=now,
            risk_level=risk,
            distance_m=dist,
            message=msg,
            pothole_count=len(potholes),
        )
        self._history.appendleft(event)

        logger.info("ALERT [%s] %s", risk.upper(), msg)

        #Emit via SocketIO if available
        if self.socketio:
            self._emit(event)

        return event

    @property
    def active_alert(self) -> Optional[str]:
        return self._active_alert

    def get_recent_alerts(self, n: int = 10) -> List[dict]:
        return [
            {
                "timestamp": a.timestamp,
                "risk_level": a.risk_level,
                "distance_m": a.distance_m,
                "message": a.message,
                "pothole_count": a.pothole_count,
            }
            for a in list(self._history)[:n]
        ]

    def clear_active(self):
        self._active_alert = None

    #Private───────────

    @staticmethod
    def _build_message(risk: str, dist: Optional[float], count: int) -> str:
        prefix = {
            "near":   "⚠ DANGER",
            "medium": "⚡ CAUTION",
        }.get(risk, "ℹ ALERT")

        dist_str = f"{dist:.1f}m" if dist else "nearby"
        cnt_str = f"({count} detected)" if count > 1 else ""
        return f"{prefix} — Pothole {dist_str} ahead! {cnt_str}".strip()

    def _emit(self, event: AlertEvent):
        try:
            self.socketio.emit("alert", {
                "risk_level": event.risk_level,
                "distance_m": event.distance_m,
                "message": event.message,
                "pothole_count": event.pothole_count,
                "timestamp": event.timestamp,
                "play_sound": True,
            })
        except Exception as exc:
            logger.warning("SocketIO emit error: %s", exc)
