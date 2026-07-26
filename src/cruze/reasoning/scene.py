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
from cruze.core.types import EgoEstimate, Lanes, ObjectClass, Scene, Track, VehicleState
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

# Half-width of the ego lane in metres, used when a track has a ground-plane
# position. A US lane is ~3.7 m; the extra margin past 1.85 m accepts a vehicle
# straddling the line or sitting slightly off-centre, which is exactly the
# case a following-distance warning needs to catch.
_EGO_LANE_HALF_WIDTH_M = 2.4

# A box within this many pixels of the left or right frame edge is treated as
# running off it. A couple of pixels of slack absorbs detector jitter on a box
# that genuinely ends at the boundary.
_EDGE_MARGIN_PX = 3.0

# Lanes unreceived for longer than this are dropped from the Scene: the lane
# detector stopped or is disabled, and stale geometry would mislead. Measured
# from MESSAGE ARRIVAL, not the frame capture timestamp — on a loaded CPU the
# capture-to-publish latency alone can exceed this window while the lanes are
# still the freshest output available (they ship with the same frame's tracks).
_LANES_MAX_AGE_S = 0.5

# EgoEstimate older than this is dropped from the Scene — same rationale and
# window as lanes: stale neural-net geometry would mislead downstream.
_EGO_MAX_AGE_S = 0.5


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
        self._latest_ego: EgoEstimate | None = None
        self._ego_seen_at = 0.0  # monotonic arrival time of the last EgoEstimate
        self._running = False

    async def run(self) -> None:
        self._running = True
        tracks_q = self._bus.subscribe(Channel.PERCEPTION_TRACKS, maxsize=4)
        state_q = self._bus.subscribe(Channel.TELEMETRY_VEHICLE_STATE, maxsize=4)
        lanes_q = self._bus.subscribe(Channel.PERCEPTION_LANES, maxsize=2)
        ego_q = self._bus.subscribe(Channel.PERCEPTION_EGO, maxsize=2)
        logger.info("SceneAssembler started")

        async def drain_aux() -> None:
            """Keep vehicle state, lanes and the ego estimate up to date.

            Waits on the three queues rather than polling them: the previous
            10 ms poll loop woke 100 times a second to find nothing, on the
            same event loop the perception publishes run on.
            """
            pending = {
                asyncio.ensure_future(q.get()): q
                for q in (state_q, lanes_q, ego_q)
            }
            try:
                while self._running:
                    done, _ = await asyncio.wait(
                        pending.keys(),
                        timeout=1.0,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in done:
                        queue = pending.pop(task)
                        message = task.result()
                        if queue is state_q:
                            self._latest_vehicle_state = message
                        elif queue is lanes_q:
                            self._latest_lanes = message
                            self._lanes_seen_at = time.monotonic()
                        else:
                            self._latest_ego = message
                            self._ego_seen_at = time.monotonic()
                        pending[asyncio.ensure_future(queue.get())] = queue
            finally:
                for task in pending:
                    task.cancel()

        aux_task = asyncio.ensure_future(drain_aux())
        self._aux_task = aux_task

        while self._running:
            try:
                tracks: list[Track] = await asyncio.wait_for(tracks_q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            scene = self._build_scene(tracks)
            await self._bus.publish(Channel.REASONING_SCENE, scene)

    async def stop(self) -> None:
        self._running = False
        aux_task = getattr(self, "_aux_task", None)
        if aux_task is not None:
            aux_task.cancel()

    def _build_scene(self, tracks: list[Track]) -> Scene:
        tracks = self._with_absolute_speed(tracks)
        lead = self._find_lead(tracks)
        now = time.monotonic()
        lanes = self._latest_lanes
        if lanes is not None and now - self._lanes_seen_at > _LANES_MAX_AGE_S:
            lanes = None
        ego = self._latest_ego
        if ego is not None and now - self._ego_seen_at > _EGO_MAX_AGE_S:
            ego = None
        scene = Scene(
            timestamp=now,
            # Every track in a batch carries the frame it was produced from;
            # with none, fall back to whichever net output is still fresh so
            # the dashboard can still pair an empty scene with its frame.
            frame_id=self._batch_frame_id(tracks, lanes, ego),
            tracks=tuple(tracks),
            vehicle_state=self._latest_vehicle_state,
            lead_track=lead,
            lanes=lanes,
            ego_path=ego.ego_path if ego else None,
            road_curvature_1pm=ego.road_curvature_1pm if ego else None,
            cipo_distance_m=ego.cipo_distance_m if ego else None,
            cipo_flag=ego.cipo_flag if ego else None,
        )
        # Attach the IDM urgency scalar here so the HUD corridor colour and
        # the event engine's brake warnings derive from the same number.
        return dataclasses.replace(
            scene,
            required_accel_mps2=threat.idm_required_accel(scene, self._cfg.reasoning),
        )

    @staticmethod
    def _batch_frame_id(
        tracks: list[Track], lanes: Lanes | None, ego: EgoEstimate | None
    ) -> int:
        """Frame this scene describes. The tracker stamps one frame_id across a
        whole batch, so any track answers; lanes and the ego estimate ship with
        the same frame and cover the empty-batch case."""
        for track in tracks:
            return track.frame_id
        if ego is not None:
            return ego.frame_id
        if lanes is not None:
            return lanes.frame_id
        return 0

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

    @property
    def _frame_width(self) -> int:
        """Width of the frames actually arriving. CameraService corrects the
        shared config once it sees a real frame, so read it rather than the
        value captured at construction."""
        return self._cfg.camera.width or self._image_width

    def _is_edge_truncated(self, t: Track) -> bool:
        """True when the box runs off the left or right of the frame.

        Such a box has no meaningful centre — the object continues past the
        edge — so both its lateral position and its ground-plane range are
        measured against the frame boundary rather than the object. A vehicle
        cut off at the side is beside us, not ahead, and treating it as the
        lead produces a stream of phantom collision warnings.

        Only the sides are tested. A genuine lead in close traffic legitimately
        touches the bottom edge, and excluding those would drop the real target
        exactly when a forward-collision warning matters most.
        """
        return t.bbox.x1 <= _EDGE_MARGIN_PX or t.bbox.x2 >= self._frame_width - _EDGE_MARGIN_PX

    def _in_ego_lane(self, t: Track) -> bool:
        """Same-lane test, in metres where the geometry allows it.

        A track's ground position gives its lateral offset directly, so the
        test is the physical one: is it inside the ego lane. The pixel-fraction
        proxy below is the fallback, and it is only as good as the configured
        image width — which is a request, not a fact, for a capture device and
        simply wrong for a replayed file of another resolution.
        """
        if t.ground_xz_m is not None:
            return abs(t.ground_xz_m[0]) < _EGO_LANE_HALF_WIDTH_M
        cx_image = self._frame_width / 2
        return abs(t.bbox.cx - cx_image) < self._frame_width * _LANE_CENTRE_FRACTION

    def _find_lead(self, tracks: list[Track]) -> Track | None:
        """
        Identify the lead vehicle: the closest vehicle-class track in the ego
        lane.
        """
        candidates = [
            t for t in tracks
            if t.cls in _VEHICLE_CLASSES
            and t.distance_m is not None
            and not self._is_edge_truncated(t)
            and self._in_ego_lane(t)
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda t: t.distance_m)  # type: ignore[return-value]
