"""
OpenCV HUD overlay.

Subscribes to:
  Channel.PERCEPTION_FRAME   — raw frame
  Channel.PERCEPTION_TRACKS  — list[Track]
  Channel.PERCEPTION_LANES   — Lanes (per-frame lane overlay)
  Channel.TELEMETRY_VEHICLE_STATE — VehicleState
  Channel.REASONING_EVENT    — DrivingEvent (for alert banner)

Draws on each frame and calls cv2.imshow().  Falls back gracefully when
opencv-python is not installed (just skips drawing). Runs only when
hmi.backend is "opencv" or "both" (the web dashboard covers "web").
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import TYPE_CHECKING, Any

from cruze.core.bus import Channel, EventBus
from cruze.core.types import (
    DrivingEvent,
    EventLevel,
    Frame,
    LaneLine,
    Lanes,
    Scene,
    Track,
    TrafficLightState,
    VehicleState,
)

if TYPE_CHECKING:
    from cruze.core.config import Config

logger = logging.getLogger(__name__)

# Alert banner stays visible for this many seconds.
_ALERT_TTL_S = 4.0

# Lanes unreceived for longer than this aren't drawn — lane detection stopped
# or is disabled. Measured from message ARRIVAL, not the frame capture
# timestamp: capture-to-publish latency alone can exceed this on a loaded CPU.
_LANES_TTL_S = 1.0

# Scenes older than this can't colour the corridor — reasoning stalled, so
# fall back to the calm green rather than freezing an old urgency level.
_SCENE_TTL_S = 1.0

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

# Traffic-light lamp state → BGR. Overrides _CLASS_COLOUR for lights so the
# box itself reads red/amber/green at a glance.
_LIGHT_COLOUR = {
    TrafficLightState.RED:     (0, 0, 255),
    TrafficLightState.YELLOW:  (0, 200, 255),
    TrafficLightState.GREEN:   (0, 255, 0),
    TrafficLightState.UNKNOWN: (160, 160, 160),
}

# Lane glow: thick dark pass under a thin bright core.
_LANE_GLOW_COLOUR = (0, 120, 0)
_LANE_CORE_COLOUR = (120, 255, 160)

# IDM urgency colours (BGR), mirroring the web dashboard's hex bands:
# red #ff3b30, orange #ff7a00, amber #ffb000.
_URGENCY_RED = (48, 59, 255)
_URGENCY_ORANGE = (0, 122, 255)
_URGENCY_AMBER = (0, 176, 255)


class HUDService:
    def __init__(self, cfg: "Config", bus: EventBus) -> None:
        self._cfg = cfg
        self._bus = bus
        self._latest_tracks: list[Track] = []
        self._latest_state = VehicleState()
        self._latest_lanes: Lanes | None = None
        self._lanes_seen_at = 0.0  # monotonic arrival time of the last Lanes
        self._latest_scene: Scene | None = None  # carries the IDM urgency scalar
        self._alert_queue: deque[tuple[DrivingEvent, float]] = deque(maxlen=3)
        self._running = False

    async def run(self) -> None:
        self._running = True

        if not self._cfg.hmi.enabled or self._cfg.hmi.backend not in ("opencv", "both"):
            logger.info("HUD: disabled (backend=%s)", self._cfg.hmi.backend)
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
        lane_q = self._bus.subscribe(Channel.PERCEPTION_LANES, maxsize=2)
        state_q = self._bus.subscribe(Channel.TELEMETRY_VEHICLE_STATE, maxsize=2)
        scene_q = self._bus.subscribe(Channel.REASONING_SCENE, maxsize=2)
        event_q = self._bus.subscribe(Channel.REASONING_EVENT, maxsize=4)

        async def drain_aux() -> None:
            while self._running:
                for q, attr in [(track_q, "_latest_tracks"), (state_q, "_latest_state"),
                                (lane_q, "_latest_lanes"), (scene_q, "_latest_scene")]:
                    if not q.empty():
                        setattr(self, attr, q.get_nowait())
                        if attr == "_latest_lanes":
                            self._lanes_seen_at = time.monotonic()
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

        # Urgency corridor + lane boundaries (under the track boxes).
        if self._cfg.hmi.show_lanes and self._latest_lanes is not None:
            if time.monotonic() - self._lanes_seen_at < _LANES_TTL_S:
                lanes = self._latest_lanes
                core = self._urgency_colour_bgr()
                left_pts = _boundary_points(lanes.left, lanes.left_poly)
                right_pts = _boundary_points(lanes.right, lanes.right_poly)

                # Corridor fill; skip when the extrapolated tops cross
                # (self-intersecting bowtie). Fill at ~1/3 intensity so the
                # road stays visible under the single addWeighted pass.
                if (left_pts is not None and right_pts is not None
                        and left_pts[-1][0] < right_pts[-1][0]):
                    ring = np.array(left_pts + right_pts[::-1], np.int32)
                    dim = tuple(c // 3 for c in core)
                    self._cv2.fillPoly(overlay, [ring], dim)

                for pts in (left_pts, right_pts):
                    if pts is None:
                        continue
                    arr = np.array(pts, np.int32)
                    # Thick dark pass + thin bright core = cheap glow.
                    self._cv2.polylines(overlay, [arr], False, _LANE_GLOW_COLOUR, 9)
                    self._cv2.polylines(overlay, [arr], False, core, 3)

        # Draw tracks.
        for track in self._latest_tracks:
            b = track.bbox
            if track.light_state is not None:
                colour = _LIGHT_COLOUR[track.light_state]
            else:
                colour = _CLASS_COLOUR.get(track.cls.value, (128, 128, 128))

            # Instance mask under the box; addWeighted below supplies the
            # translucency.
            if self._cfg.hmi.show_masks and track.mask_xy is not None:
                pts = np.array(track.mask_xy, dtype=np.int32)
                self._cv2.fillPoly(overlay, [pts], colour)

            self._cv2.rectangle(overlay, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), colour, 2)

            label_parts = [track.cls.value]
            if self._cfg.hmi.show_track_ids:
                label_parts.append(f"#{track.track_id}")
            if track.light_state is not None:
                label_parts.append(track.light_state.value.upper())
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

    def _urgency_colour_bgr(self) -> tuple[int, int, int]:
        """Corridor/lane colour from the IDM urgency scalar; mirrors the web
        dashboard's colour bands. Calm/unknown/stale → phosphor green."""
        scene = self._latest_scene
        accel = None
        if scene is not None and time.monotonic() - scene.timestamp < _SCENE_TTL_S:
            accel = scene.required_accel_mps2
        if accel is None:
            return _LANE_CORE_COLOUR
        r = self._cfg.reasoning
        if accel < -r.idm_hard_decel_mps2:
            return _URGENCY_RED
        if accel < -r.idm_advise_decel_mps2:
            return _URGENCY_ORANGE
        if accel < -1.0:  # firmer-than-comfort decel band (matches render.js)
            return _URGENCY_AMBER
        return _LANE_CORE_COLOUR


def _boundary_points(
    line: LaneLine | None,
    poly: tuple[tuple[float, float], ...] | None,
) -> list[tuple[int, int]] | None:
    """Integer pixel points for one boundary: curved polyline when present,
    else the straight segment's two endpoints."""
    if poly is not None:
        return [(int(x), int(y)) for x, y in poly]
    if line is not None:
        return [(int(line.x1), int(line.y1)), (int(line.x2), int(line.y2))]
    return None
