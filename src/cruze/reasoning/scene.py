"""
Scene assembler.

Subscribes to:
  Channel.PERCEPTION_TRACKS    — list[Track]
  Channel.TELEMETRY_VEHICLE_STATE — VehicleState

On each tracks update, joins the latest vehicle state and identifies the
lead vehicle (nearest, same approximate lane, classified as a vehicle).

Publishes:
  Channel.REASONING_SCENE — Scene
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from typing import TYPE_CHECKING

from cruze.core.bus import Channel, EventBus
from cruze.core.types import Lanes, ObjectClass, Scene, Track, VehicleState
from cruze.reasoning import threat

if TYPE_CHECKING:
    from cruze.core.config import Config

logger = logging.getLogger(__name__)

# Object classes treated as vehicles for lead selection and absolute-speed
# annotation. Bicycles are deliberately excluded: their depth/closing-speed
# estimates are too noisy for a meaningful ego-minus-closing speed.
_VEHICLE_CLASSES = {
    ObjectClass.CAR,
    ObjectClass.TRUCK,
    ObjectClass.BUS,
    ObjectClass.MOTORCYCLE,
}

# Fraction of frame width that counts as "roughly centered" (same lane proxy).
# Without lane detection, we use horizontal position in the image as a proxy.
# A track whose centre x is within this fraction of the image centre is
# considered to be in the ego lane.
_LANE_CENTRE_FRACTION = 0.35

# Lanes unreceived for longer than this are dropped from the Scene: the lane
# detector stopped or is disabled, and stale geometry would mislead. Measured
# from MESSAGE ARRIVAL, not the frame capture timestamp — on a loaded CPU the
# capture-to-publish latency alone can exceed this window while the lanes are
# still the freshest output available (they ship with the same frame's tracks).
_LANES_MAX_AGE_S = 0.5


class SceneAssembler:
    """
    Joins tracks + vehicle state into Scene snapshots.

    Parameters
    ----------
    cfg:
        Full application config.
    bus:
        Shared event bus.
    image_width:
        Camera frame width in pixels; used for lead-vehicle lane heuristic.
    """

    def __init__(self, cfg: "Config", bus: EventBus, image_width: int = 1280) -> None:
        self._cfg = cfg
        self._bus = bus
        self._image_width = image_width
        self._latest_vehicle_state = VehicleState()
        self._latest_lanes: Lanes | None = None
        self._lanes_seen_at = 0.0  # monotonic arrival time of the last Lanes
        self._running = False

    async def run(self) -> None:
        self._running = True
        tracks_q = self._bus.subscribe(Channel.PERCEPTION_TRACKS, maxsize=4)
        state_q = self._bus.subscribe(Channel.TELEMETRY_VEHICLE_STATE, maxsize=4)
        lanes_q = self._bus.subscribe(Channel.PERCEPTION_LANES, maxsize=2)
        logger.info("SceneAssembler started")

        async def drain_aux() -> None:
            """Keep vehicle state and lanes up to date in the background."""
            while self._running:
                drained = False
                if not state_q.empty():
                    self._latest_vehicle_state = state_q.get_nowait()
                    drained = True
                if not lanes_q.empty():
                    self._latest_lanes = lanes_q.get_nowait()
                    self._lanes_seen_at = time.monotonic()
                    drained = True
                if not drained:
                    await asyncio.sleep(0.01)
                else:
                    await asyncio.sleep(0)

        asyncio.ensure_future(drain_aux())

        while self._running:
            try:
                tracks: list[Track] = await asyncio.wait_for(tracks_q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            scene = self._build_scene(tracks)
            await self._bus.publish(Channel.REASONING_SCENE, scene)

    async def stop(self) -> None:
        self._running = False

    def _build_scene(self, tracks: list[Track]) -> Scene:
        tracks = self._with_absolute_speed(tracks)
        lead = self._find_lead(tracks)
        now = time.monotonic()
        lanes = self._latest_lanes
        if lanes is not None and now - self._lanes_seen_at > _LANES_MAX_AGE_S:
            lanes = None
        scene = Scene(
            timestamp=now,
            tracks=tuple(tracks),
            vehicle_state=self._latest_vehicle_state,
            lead_track=lead,
            lanes=lanes,
        )
        # Attach the IDM urgency scalar here so the HUD corridor colour and
        # the event engine's brake warnings derive from the same number.
        return dataclasses.replace(
            scene,
            required_accel_mps2=threat.idm_required_accel(scene, self._cfg.reasoning),
        )

    def _with_absolute_speed(self, tracks: list[Track]) -> list[Track]:
        """
        Attach absolute ground speed: ego speed − closing speed.

        Valid for same-direction traffic (the dominant case for vehicles
        ahead); oncoming vehicles read high. Clamped at 0 — a negative value
        would mean the target is reversing toward us, which at our accuracy
        is indistinguishable from noise.
        """
        ego = self._latest_vehicle_state.speed_mps
        if ego is None:
            return tracks
        out: list[Track] = []
        for t in tracks:
            if t.cls in _VEHICLE_CLASSES and t.closing_speed_mps is not None:
                out.append(
                    dataclasses.replace(t, speed_mps=max(0.0, ego - t.closing_speed_mps))
                )
            else:
                out.append(t)
        return out

    def _find_lead(self, tracks: list[Track]) -> Track | None:
        """
        Identify the lead vehicle: closest vehicle-class track that appears
        roughly centred in the frame (same-lane proxy).
        """
        cx_image = self._image_width / 2
        margin = self._image_width * _LANE_CENTRE_FRACTION

        candidates = [
            t for t in tracks
            if t.cls in _VEHICLE_CLASSES
            and t.distance_m is not None
            and abs(t.bbox.cx - cx_image) < margin
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda t: t.distance_m)  # type: ignore[return-value]
