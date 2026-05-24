"""
Async camera frame publisher.

Publishes Frame objects to Channel.PERCEPTION_FRAME at the configured FPS.

Two modes:
  simulated=False  — opens a real capture device (OpenCV VideoCapture).
  simulated=True   — emits synthetic black frames at the configured rate;
                     no camera hardware or OpenCV needed.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from cruze.core.bus import Channel, EventBus
from cruze.core.types import Frame

if TYPE_CHECKING:
    from cruze.core.config import CameraConfig

logger = logging.getLogger(__name__)


class CameraService:
    """
    Reads frames from a capture device (or simulates them) and publishes to
    Channel.PERCEPTION_FRAME.

    Parameters
    ----------
    cfg:
        Camera configuration block.
    bus:
        Shared event bus.
    focal_length_px:
        Pre-computed focal length in pixels; embedded in every Frame so
        downstream modules don't need to recompute it.
    """

    def __init__(
        self,
        cfg: "CameraConfig",
        bus: EventBus,
        focal_length_px: float | None = None,
    ) -> None:
        self._cfg = cfg
        self._bus = bus
        self._focal_length_px = focal_length_px
        self._frame_id: int = 0
        self._running = False

    async def run(self) -> None:
        self._running = True
        if self._cfg.simulated:
            await self._run_simulated()
        else:
            await self._run_real()

    async def stop(self) -> None:
        self._running = False

    async def _run_simulated(self) -> None:
        import numpy as np  # optional; only needed for real frames

        logger.info("Camera: running in simulated mode (%dx%d @ %d fps)",
                    self._cfg.width, self._cfg.height, self._cfg.fps)
        interval = 1.0 / self._cfg.fps
        while self._running:
            t0 = time.monotonic()
            image = np.zeros((self._cfg.height, self._cfg.width, 3), dtype="uint8")
            frame = Frame(
                image=image,
                timestamp=t0,
                frame_id=self._frame_id,
                focal_length_px=self._focal_length_px,
            )
            await self._bus.publish(Channel.PERCEPTION_FRAME, frame)
            self._frame_id += 1
            elapsed = time.monotonic() - t0
            await asyncio.sleep(max(0.0, interval - elapsed))

    async def _run_real(self) -> None:
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "opencv-python is required for real camera capture. "
                "Install it with: pip install 'cruze[vision]'"
            ) from exc

        cap = cv2.VideoCapture(self._cfg.device_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self._cfg.fps)

        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera device index={self._cfg.device_index}. "
                "Check the device is connected, or use --no-camera for simulated mode."
            )

        logger.info("Camera: opened device %d (%dx%d @ %d fps)",
                    self._cfg.device_index, self._cfg.width, self._cfg.height, self._cfg.fps)

        try:
            interval = 1.0 / self._cfg.fps
            while self._running:
                t0 = time.monotonic()
                ret, image = cap.read()
                if not ret:
                    logger.warning("Camera: frame grab failed — skipping")
                    await asyncio.sleep(0.01)
                    continue
                frame = Frame(
                    image=image,
                    timestamp=time.monotonic(),
                    frame_id=self._frame_id,
                    focal_length_px=self._focal_length_px,
                )
                await self._bus.publish(Channel.PERCEPTION_FRAME, frame)
                self._frame_id += 1
                elapsed = time.monotonic() - t0
                await asyncio.sleep(max(0.0, interval - elapsed))
        finally:
            cap.release()
            logger.info("Camera: released device %d", self._cfg.device_index)
