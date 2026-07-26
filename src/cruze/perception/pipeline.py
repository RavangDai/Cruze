"""
Perception service — VisionPilot-aligned staging of camera frames:

  Frame → Pre-Processing → AI Networks → Fusion & Tracking → publish

Two cadences share one frame:

  Hot path   (perception.target_hz, default 15 Hz)
      shared crop-2:1 preprocess → AutoSpeed (vehicles/CIPO) + AutoSteer
      (ego path) → depth → tracker. This is what the dashboard tracks.

  Context path (perception.context_hz, default 3 Hz)
      YOLO restricted to person / traffic light / stop sign, plus the HSV lamp
      classifier and classical lane fit. These classes move slowly in image
      space, and keeping a ~45 ms detector off the hot path is most of the
      pipeline's headroom. Results are cached and merged into every hot-path
      frame until the next context pass.

The camera keeps publishing at its own rate so the dashboard video stays
smooth; frames arriving faster than target_hz are dropped here rather than
queued. Whatever is in the queue is drained to the newest frame first — a
drop-oldest queue otherwise hands the pipeline a frame that is already several
periods stale, which shows up directly as overlay lag.

Publishes:
  Channel.PERCEPTION_DETECTIONS — list[Detection] per processed frame
  Channel.PERCEPTION_LANES      — Lanes (when enabled)
  Channel.PERCEPTION_TRACKS     — list[Track] per processed frame
  Channel.PERCEPTION_EGO        — EgoEstimate (when the nets are enabled)
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from cruze.core.bus import Channel, EventBus
from cruze.core.types import Detection, EgoEstimate, Frame, Lanes, ObjectClass
from cruze.perception import depth as depth_mod
from cruze.perception import detect_fusion
from cruze.perception import lane as lane_mod
from cruze.perception import lights as lights_mod
from cruze.perception.detector import Detector
from cruze.perception.tracker import Tracker

if TYPE_CHECKING:
    from cruze.core.config import Config

logger = logging.getLogger(__name__)

# Classes the context detector contributes. AutoSpeed covers the vehicle
# classes on every frame, so re-detecting them here would only produce
# duplicates for the fusion step to discard.
CONTEXT_CLASSES = (
    ObjectClass.PERSON,
    ObjectClass.BICYCLE,
    ObjectClass.TRAFFIC_LIGHT,
    ObjectClass.STOP_SIGN,
)

# A cached context detection older than this is dropped rather than drawn.
# At 3 Hz a pedestrian box is refreshed every ~330 ms; past ~2x that interval
# the cache is stale enough that holding it would put a box where nothing is.
_CONTEXT_MAX_AGE_S = 0.7


class _ScalarKalman:
    """Constant-position scalar Kalman filter for lane cross-track error."""

    # Process noise driver: lateral drift during an actual departure
    # approaches ~1 m/s — the filter must track that ramp with lag well
    # under the 0.5 m LDW threshold.
    _DRIFT_MPS = 1.0
    # Measurement noise: Hough-fit jitter + pixel quantization projected to
    # metres at the bottom image row ≈ ±0.15 m.
    _MEAS_STD_M = 0.15
    # A one-frame jump beyond this is the detector re-latching onto an
    # adjacent lane (US lane ≈ 3.7 m wide), not drift — snap, don't blend.
    _REINIT_INNOVATION_M = 1.0
    # No measurement for this long → state is stale; reinitialize. Matches
    # the SceneAssembler's lane freshness window.
    _STALE_S = 0.5

    def __init__(self) -> None:
        self.x: float | None = None
        self.p = 0.0

    def update(self, z: float, dt: float) -> float:
        if (
            self.x is None
            or dt > self._STALE_S
            or abs(z - self.x) > self._REINIT_INNOVATION_M
        ):
            self.x = z
            self.p = self._MEAS_STD_M ** 2
            return z
        p_pred = self.p + (self._DRIFT_MPS * dt) ** 2
        r = self._MEAS_STD_M ** 2
        gain = p_pred / (p_pred + r)
        self.x = self.x + gain * (z - self.x)
        self.p = (1.0 - gain) * p_pred
        return self.x


class PerceptionService:
    """
    Subscribes to PERCEPTION_FRAME, runs the two-cadence pipeline, and
    publishes detections, lanes, tracks and the ego estimate.

    Parameters
    ----------
    cfg:
        Full application config.
    bus:
        Shared event bus.
    detector:
        Pre-loaded Detector instance for the context pass
        (use perception.detector.load()).
    vision_nets:
        VisionNets bundle for the hot path; None disables the AI-network stage.
    """

    def __init__(self, cfg: "Config", bus: EventBus, detector: Detector,
                 vision_nets=None) -> None:
        self._cfg = cfg
        self._bus = bus
        self._detector = detector
        self._vision_nets = vision_nets
        # When no net supplies vehicle boxes (AutoSpeed off, or the whole
        # AI-network stage disabled), the general detector is the only thing
        # that sees a car — so it stays on the hot path at full rate and the
        # context cadence is bypassed. Graceful degradation over a profile that
        # would otherwise detect no vehicles at all.
        # getattr default: a nets bundle that does not declare a vehicle source
        # is treated as not having one, which keeps the general detector at
        # full rate. Wrong in the safe direction.
        self._nets_own_vehicles = getattr(vision_nets, "has_vehicle_source", False)
        self._tracker = Tracker(
            iou_threshold=cfg.perception.iou_threshold,
            max_age=cfg.perception.max_track_age,
        )
        self._running = False
        # Inference gets a dedicated single worker. On the shared default
        # executor it competes with the dashboard's JPEG encode, so a frame
        # can sit behind an encode that only exists to feed a browser.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="perception")
        # Smoothed lane cross-track error (per-stream state, like the tracker).
        self._cte_kalman = _ScalarKalman()
        self._last_cte_ts = 0.0
        # Cached output of the low-rate context pass.
        self._context_dets: tuple[Detection, ...] = ()
        self._context_at = 0.0
        self._last_context_run = 0.0
        # Metrics
        self._frame_count = 0
        self._skipped_count = 0
        self._drop_count = 0
        self._total_latency_ms = 0.0

    async def run(self) -> None:
        self._running = True
        queue = self._bus.subscribe(Channel.PERCEPTION_FRAME, maxsize=4)
        target_hz = self._cfg.perception.target_hz
        min_interval = 1.0 / target_hz if target_hz > 0 else 0.0
        logger.info(
            "PerceptionService started (target %.1f Hz, context %.1f Hz)",
            target_hz, self._cfg.perception.context_hz,
        )
        last_processed = 0.0

        while self._running:
            try:
                frame: Frame = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            frame = _drain_to_newest(queue, frame)

            now = time.monotonic()
            if min_interval and now - last_processed < min_interval:
                self._skipped_count += 1
                continue
            last_processed = now

            t0 = time.monotonic()
            try:
                await self._process_frame(frame)
            except Exception:
                logger.exception("PerceptionService: error processing frame %d", frame.frame_id)

            latency_ms = (time.monotonic() - t0) * 1000
            self._total_latency_ms += latency_ms
            self._frame_count += 1

            budget = self._cfg.perception.latency_budget_ms
            if latency_ms > budget:
                self._drop_count += 1
                logger.debug(
                    "Perception over budget: %.1f ms > %.1f ms (frame %d)",
                    latency_ms, budget, frame.frame_id,
                )

    async def stop(self) -> None:
        self._running = False
        self._executor.shutdown(wait=False)
        if self._frame_count:
            avg = self._total_latency_ms / self._frame_count
            logger.info(
                "PerceptionService stopped: %d frames processed (%d skipped by rate gate), "
                "%.1f ms avg latency, %d over budget",
                self._frame_count, self._skipped_count, avg, self._drop_count,
            )

    # --- stages ------------------------------------------------------------

    def _run_context(self, frame: Frame) -> tuple[tuple[Detection, ...], lane_mod.LaneResult | None]:
        """Low-rate pass: context classes + lamp state + classical lane fit."""
        detections = self._detector.detect(frame)
        detections = lights_mod.annotate_lights(detections, frame.image)
        lane_result = (
            lane_mod.detect_lanes(frame.image)
            if self._cfg.perception.lane_detection_enabled
            else None
        )
        return tuple(detections), lane_result

    def _analyze(
        self, frame: Frame, run_context: bool
    ) -> tuple[tuple[Detection, ...], lane_mod.LaneResult | None, EgoEstimate | None, bool]:
        """All CPU-bound work for one frame — runs in the perception executor."""
        ego = self._vision_nets.infer(frame) if self._vision_nets is not None else None
        if not run_context:
            return (), None, ego, False
        context_dets, lane_result = self._run_context(frame)
        return context_dets, lane_result, ego, True

    async def _process_frame(self, frame: Frame) -> None:
        now = time.monotonic()
        context_hz = self._cfg.perception.context_hz
        context_interval = 1.0 / context_hz if context_hz > 0 else 0.0
        run_context = (
            not self._nets_own_vehicles
            or context_interval == 0.0
            or now - self._last_context_run >= context_interval
        )

        context_dets, lane_result, ego, context_ran = await asyncio.get_event_loop().run_in_executor(
            self._executor, self._analyze, frame, run_context
        )
        if context_ran:
            self._context_dets = context_dets
            self._context_at = now
            self._last_context_run = now

        # AutoSpeed owns the vehicle classes on every frame; the cached context
        # pass fills in the classes it cannot see. Cached boxes expire so a
        # stalled context pass can never pin a stale pedestrian to the road.
        detections: list[Detection] = list(ego.cipo_boxes) if ego is not None else []
        if self._context_dets and now - self._context_at <= _CONTEXT_MAX_AGE_S:
            detections = detect_fusion.merge_detections(detections, self._context_dets)

        # Annotate detections with monocular depth if focal length is known.
        horizon_y: float | None = None
        if frame.focal_length_px is not None:
            image_height = frame.image.shape[0]
            horizon_y = depth_mod.horizon_y_px(
                image_height,
                frame.focal_length_px,
                self._cfg.perception.camera_pitch_deg,
            )
            # dataclasses.replace keeps every other field (mask_xy,
            # light_state, future additions) intact.
            detections = [
                dataclasses.replace(
                    d,
                    distance_m=depth_mod.estimate_distance_fused(
                        d.bbox,
                        d.cls,
                        frame.focal_length_px,
                        self._cfg.perception.camera_height_m,
                        horizon_y,
                    ),
                )
                for d in detections
            ]

        await self._bus.publish(Channel.PERCEPTION_DETECTIONS, detections)

        # Publish lanes before tracks: SceneAssembler builds a Scene on each
        # tracks message, so fresh lanes are already drained by then.
        if lane_result is not None:
            cte = self._smoothed_cte(lane_result, frame, horizon_y)
            await self._bus.publish(
                Channel.PERCEPTION_LANES,
                Lanes(
                    left=lane_result.left,
                    right=lane_result.right,
                    timestamp=frame.timestamp,
                    frame_id=frame.frame_id,
                    left_poly=lane_result.left_poly,
                    right_poly=lane_result.right_poly,
                    cte_m=cte,
                ),
            )

        tracks = self._tracker.update(
            detections,
            timestamp=frame.timestamp,
            frame_id=frame.frame_id,
            image_width=frame.image.shape[1],
            focal_length_px=frame.focal_length_px,
        )
        await self._bus.publish(Channel.PERCEPTION_TRACKS, tracks)

        if ego is not None:
            await self._bus.publish(Channel.PERCEPTION_EGO, ego)

    def _smoothed_cte(
        self, lane_result: lane_mod.LaneResult, frame: Frame, horizon_y: float | None
    ) -> float | None:
        """Kalman-smoothed cross-track error; None when geometry unavailable.
        Missing measurements need no explicit reset — the filter's staleness
        gate reinitializes on the next valid reading."""
        if frame.focal_length_px is None or horizon_y is None:
            return None
        raw = lane_mod.cte_from_lane_positions(
            lane_result.left.x1 if lane_result.left else None,
            lane_result.right.x1 if lane_result.right else None,
            frame.image.shape[1],
            frame.image.shape[0],
            frame.focal_length_px,
            self._cfg.perception.camera_height_m,
            horizon_y,
        )
        if raw is None:
            return None
        smoothed = self._cte_kalman.update(raw, frame.timestamp - self._last_cte_ts)
        self._last_cte_ts = frame.timestamp
        return smoothed

    @property
    def metrics(self) -> dict:
        return {
            "frame_count": self._frame_count,
            "skipped_count": self._skipped_count,
            "drop_count": self._drop_count,
            "avg_latency_ms": (
                self._total_latency_ms / self._frame_count if self._frame_count else 0.0
            ),
        }


def _drain_to_newest(queue: asyncio.Queue, frame: Frame) -> Frame:
    """Return the newest frame currently queued, discarding older ones.

    The bus queue is drop-oldest, so get() hands back the *oldest* buffered
    frame. Whenever perception runs slower than the camera that frame is
    already several periods old, and every downstream consumer inherits the
    lag. Perception only ever wants the most recent view of the road.
    """
    while True:
        try:
            frame = queue.get_nowait()
        except asyncio.QueueEmpty:
            return frame
