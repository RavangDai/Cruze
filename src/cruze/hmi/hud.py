"""
OpenCV HUD overlay.

Subscribes to:
  Channel.PERCEPTION_FRAME   — raw frame
  Channel.PERCEPTION_TRACKS  — list[Track]
  Channel.TELEMETRY_VEHICLE_STATE — VehicleState
  Channel.REASONING_EVENT    — DrivingEvent (for alert banner)

Draws on each frame and calls cv2.imshow().  Falls back gracefully when
opencv-python is not installed (just skips drawing).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import TYPE_CHECKING, Any

from cruze.core.bus import Channel, EventBus
from cruze.core.types import DrivingEvent, EventLevel, Frame, Track, VehicleState

if TYPE_CHECKING:
    from cruze.core.config import Config

logger = logging.getLogger(__name__)

# Alert banner stays visible for this many seconds.
_ALERT_TTL_S = 4.0

_LEVEL_COLOUR = {
    EventLevel.INFO:     (200, 200, 200),  # grey
    EventLevel.NOTICE:   (0, 200, 255),    # amber
    EventLevel.WARNING:  (0, 100, 255),    # orange
    EventLevel.CRITICAL: (0, 0, 255),      # red
}

_CLASS_COLOUR = {
    "car":           (0, 255, 0),
    "truck":         (0, 200, 100),
    "bus":           (0, 180, 180),
    "person":        (255, 128, 0),
    "motorcycle":    (255, 255, 0),
    "bicycle":       (128, 255, 0),
    "stop_sign":     (0, 0, 255),
    "traffic_light": (0, 255, 255),
    "unknown":       (128, 128, 128),
}


class HUDService:
    def __init__(self, cfg: "Config", bus: EventBus) -> None:
        self._cfg = cfg
        self._bus = bus
        self._latest_tracks: list[Track] = []
        self._latest_state = VehicleState()
        self._alert_queue: deque[tuple[DrivingEvent, float]] = deque(maxlen=3)
        self._running = False

    async def run(self) -> None:
        self._running = True

        if not self._cfg.hmi.enabled:
            logger.info("HUD: disabled in config")
            while self._running:
                await asyncio.sleep(1.0)
            return

        try:
            import cv2 as _cv2  # type: ignore
            self._cv2 = _cv2
        except ImportError:
            logger.warning("HUD: opencv-python not installed — overlay disabled")
            while self._running:
                await asyncio.sleep(1.0)
            return

        frame_q = self._bus.subscribe(Channel.PERCEPTION_FRAME, maxsize=2)
        track_q = self._bus.subscribe(Channel.PERCEPTION_TRACKS, maxsize=2)
        state_q = self._bus.subscribe(Channel.TELEMETRY_VEHICLE_STATE, maxsize=2)
        event_q = self._bus.subscribe(Channel.REASONING_EVENT, maxsize=4)

        async def drain_aux() -> None:
            while self._running:
                for q, attr in [(track_q, "_latest_tracks"), (state_q, "_latest_state")]:
                    if not q.empty():
                        setattr(self, attr, q.get_nowait())
                if not event_q.empty():
                    ev: DrivingEvent = event_q.get_nowait()
                    self._alert_queue.append((ev, time.monotonic()))
                await asyncio.sleep(0.01)

        asyncio.ensure_future(drain_aux())

        while self._running:
            try:
                frame: Frame = await asyncio.wait_for(frame_q.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue

            drawn = self._draw(frame)
            self._cv2.imshow(self._cfg.hmi.window_title, drawn)
            key = self._cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("HUD: quit key pressed")
                break

        self._cv2.destroyAllWindows()

    async def stop(self) -> None:
        self._running = False

    def _draw(self, frame: Frame) -> Any:
        import numpy as np  # type: ignore
        img = frame.image.copy()
        h, w = img.shape[:2]
        alpha = self._cfg.hmi.overlay_alpha

        overlay = img.copy()

        # Draw tracks.
        for track in self._latest_tracks:
            b = track.bbox
            colour = _CLASS_COLOUR.get(track.cls.value, (128, 128, 128))
            self._cv2.rectangle(overlay, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), colour, 2)

            label_parts = [track.cls.value]
            if self._cfg.hmi.show_track_ids:
                label_parts.append(f"#{track.track_id}")
            if self._cfg.hmi.show_distance and track.distance_m is not None:
                label_parts.append(f"{track.distance_m:.0f}m")
            label = " ".join(label_parts)
            self._cv2.putText(overlay, label, (int(b.x1), int(b.y1) - 6),
                         self._cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

        # Speed display (bottom-left).
        speed_mps = self._latest_state.speed_mps
        if speed_mps is not None:
            speed_mph = speed_mps * 2.237
            speed_str = f"{speed_mph:.0f} mph"
        else:
            speed_str = "-- mph"

        posted = self._latest_state.posted_speed_limit_mps
        if posted is not None:
            speed_str += f"  /  limit {posted * 2.237:.0f}"

        self._cv2.putText(overlay, speed_str, (10, h - 20),
                     self._cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Alert banner.
        now = time.monotonic()
        active_alerts = [(ev, ts) for ev, ts in self._alert_queue if now - ts < _ALERT_TTL_S]
        for i, (ev, _ts) in enumerate(reversed(active_alerts)):
            colour = _LEVEL_COLOUR.get(ev.level, (200, 200, 200))
            banner = f"{ev.kind.upper()}: {ev.level.value}"
            y = 40 + i * 36
            self._cv2.rectangle(overlay, (0, y - 28), (w, y + 8), colour, -1)
            self._cv2.putText(overlay, banner, (10, y),
                         self._cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        self._cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        return img
