"""
Perception service — orchestrates camera frames through the full pipeline:
  Frame → Detector → Depth → Tracker → publish detections + tracks

Latency budget (configurable):
  Total target: cfg.perception.latency_budget_ms
  Split:  detector ~70%, tracker ~10%, depth ~10%, overhead ~10%

Publishes:
  Channel.PERCEPTION_DETECTIONS — list[Detection] per frame
  Channel.PERCEPTION_TRACKS     — list[Track] per frame
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from cruze.core.bus import Channel, EventBus
from cruze.core.types import Detection, Frame
from cruze.perception import depth as depth_mod
from cruze.perception.detector import Detector
from cruze.perception.tracker import Tracker

if TYPE_CHECKING:
    from cruze.core.config import Config

logger = logging.getLogger(__name__)


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

    def __init__(self, cfg: "Config", bus: EventBus, detector: Detector) -> None:
        self._cfg = cfg
        self._bus = bus
        self._detector = detector
        self._tracker = Tracker(
            iou_threshold=cfg.perception.iou_threshold,
            max_age=cfg.perception.max_track_age,
        )
        self._running = False
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

    async def _process_frame(self, frame: Frame) -> None:
        # Run detector in a thread to avoid blocking the event loop.
        detections: list[Detection] = await asyncio.get_event_loop().run_in_executor(
            None, self._detector.detect, frame
        )

        # Annotate detections with monocular depth if focal length is known.
        if frame.focal_length_px is not None:
            detections = [
                Detection(
                    bbox=d.bbox,
                    confidence=d.confidence,
                    cls=d.cls,
                    distance_m=depth_mod.estimate_distance(
                        d.bbox, d.cls, frame.focal_length_px
                    ),
                )
                for d in detections
            ]

        await self._bus.publish(Channel.PERCEPTION_DETECTIONS, detections)

        tracks = self._tracker.update(detections)
        await self._bus.publish(Channel.PERCEPTION_TRACKS, tracks)

    @property
    def metrics(self) -> dict:
        return {
            "frame_count": self._frame_count,
            "drop_count": self._drop_count,
            "avg_latency_ms": (
                self._total_latency_ms / self._frame_count if self._frame_count else 0.0
            ),
        }
