"""
Perception service — orchestrates camera frames through the full pipeline:
  Frame → Detector → Lights → Lanes → Depth → Tracker → publish

Latency budget (configurable):
  Total target: cfg.perception.latency_budget_ms
  Split:  detector ~70%, tracker ~10%, depth ~10%, overhead ~10%
  (lights + lanes add ~5 ms on top; classical CV, near-constant)

Publishes:
  Channel.PERCEPTION_DETECTIONS — list[Detection] per frame
  Channel.PERCEPTION_LANES      — Lanes per frame (when enabled)
  Channel.PERCEPTION_TRACKS     — list[Track] per frame
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from typing import TYPE_CHECKING

from cruze.core.bus import Channel, EventBus
from cruze.core.types import Detection, EgoEstimate, Frame, Lanes
from cruze.perception import depth as depth_mod
from cruze.perception import lane as lane_mod
from cruze.perception import lights as lights_mod
from cruze.perception.detector import Detector
from cruze.perception.tracker import Tracker

if TYPE_CHECKING:
    from cruze.core.config import Config

logger = logging.getLogger(__name__)


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
    Subscribes to PERCEPTION_FRAME, runs the detector + tracker, and
    publishes PERCEPTION_DETECTIONS and PERCEPTION_TRACKS.

    Parameters
    ----------
    cfg:
        Full application config.
    bus:
        Shared event bus.
    detector:
        Pre-loaded Detector instance (use perception.detector.load()).
    """

    def __init__(self, cfg: "Config", bus: EventBus, detector: Detector,
                 vision_nets=None) -> None:
        self._cfg = cfg
        self._bus = bus
        self._detector = detector
        self._vision_nets = vision_nets
        self._tracker = Tracker(
            iou_threshold=cfg.perception.iou_threshold,
            max_age=cfg.perception.max_track_age,
        )
        self._running = False
        # Smoothed lane cross-track error (per-stream state, like the tracker).
        self._cte_kalman = _ScalarKalman()
        self._last_cte_ts = 0.0
        # Metrics
        self._frame_count = 0
        self._drop_count = 0
        self._total_latency_ms = 0.0

    async def run(self) -> None:
        self._running = True
        queue = self._bus.subscribe(Channel.PERCEPTION_FRAME, maxsize=4)
        logger.info("PerceptionService started")

        while self._running:
            try:
                frame: Frame = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

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
        if self._frame_count:
            avg = self._total_latency_ms / self._frame_count
            logger.info(
                "PerceptionService stopped: %d frames processed, %.1f ms avg latency, %d over budget",
                self._frame_count, avg, self._drop_count,
            )

    def _analyze(self, frame: Frame) -> tuple[list[Detection], lane_mod.LaneResult | None, EgoEstimate | None]:
        """All CPU-bound classical/ML work for one frame — runs in the executor."""
        detections = self._detector.detect(frame)
        detections = lights_mod.annotate_lights(detections, frame.image)
        lane_result = (
            lane_mod.detect_lanes(frame.image)
            if self._cfg.perception.lane_detection_enabled
            else None
        )
        ego = self._vision_nets.infer(frame) if self._vision_nets is not None else None
        return detections, lane_result, ego

    async def _process_frame(self, frame: Frame) -> None:
        # One executor hop for detector + light state + lanes so no CPU-bound
        # work ever blocks the event loop.
        detections, lane_result, ego = await asyncio.get_event_loop().run_in_executor(
            None, self._analyze, frame
        )

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

        tracks = self._tracker.update(detections)
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
            "drop_count": self._drop_count,
            "avg_latency_ms": (
                self._total_latency_ms / self._frame_count if self._frame_count else 0.0
            ),
        }
