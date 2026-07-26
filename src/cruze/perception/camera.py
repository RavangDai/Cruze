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
        # Width the configured focal length was derived from. A capture device
        # may ignore a resolution request and a replayed file always does, so
        # the intrinsic is re-derived once the true frame size is known —
        # otherwise every distance downstream is scaled by the size mismatch.
        self._focal_basis_width = cfg.width

    def _focal_for(self, frame_width: int) -> float | None:
        """Focal length in px for the frame actually being captured."""
        if frame_width == self._focal_basis_width:
            return self._focal_length_px
        corrected = (
            None if self._focal_length_px is None
            else self._focal_length_px * frame_width / self._focal_basis_width
        )
        logger.warning(
            "Camera: frames are %d px wide but config says %d — focal length "
            "corrected %s -> %s px so distances stay calibrated",
            frame_width, self._focal_basis_width,
            f"{self._focal_length_px:.1f}" if self._focal_length_px else "none",
            f"{corrected:.1f}" if corrected else "none",
        )
        self._focal_length_px = corrected
        self._focal_basis_width = frame_width
        return corrected

    def _adopt_frame_size(self, image) -> None:
        """Make the shared camera config describe the frames actually arriving.

        Config resolution is a request: a capture device may ignore it and a
        replayed file always does. Consumers that reason in image coordinates
        (lead-lane geometry, edge-truncation tests) read it, so leaving it
        stale silently skews them.
        """
        height, width = image.shape[0], image.shape[1]
        if (width, height) != (self._cfg.width, self._cfg.height):
            logger.info("Camera: frame size %dx%d (config said %dx%d) — config updated",
                        width, height, self._cfg.width, self._cfg.height)
            self._cfg.width = width
            self._cfg.height = height

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

        is_file = bool(self._cfg.source)
        if is_file:
            import os
            if not os.path.isfile(self._cfg.source):
                raise RuntimeError(f"Video file not found: {self._cfg.source}")
            cap = cv2.VideoCapture(self._cfg.source)
        else:
            cap = cv2.VideoCapture(self._cfg.device_index)
            # Resolution/FPS hints only apply to live capture devices; a video
            # file plays at whatever it was encoded with.
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cfg.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.height)
            cap.set(cv2.CAP_PROP_FPS, self._cfg.fps)

        if not cap.isOpened():
            src = self._cfg.source if is_file else f"device index={self._cfg.device_index}"
            raise RuntimeError(
                f"Cannot open camera source: {src}. "
                "Check the device/file, or use --no-camera for simulated mode."
            )

        # Pace file playback at the file's native FPS so timing-dependent rules
        # (closing speed, TTC) behave as they would in real time. Fall back to
        # the configured FPS when the file reports a bogus value.
        if is_file:
            native_fps = cap.get(cv2.CAP_PROP_FPS)
            fps = native_fps if native_fps and native_fps > 0 else self._cfg.fps
            logger.info("Camera: replaying file %s @ %.1f fps (loop=%s)",
                        self._cfg.source, fps, self._cfg.loop)
        else:
            fps = self._cfg.fps
            logger.info("Camera: opened device %d (%dx%d @ %d fps)",
                        self._cfg.device_index, self._cfg.width, self._cfg.height, self._cfg.fps)

        try:
            interval = 1.0 / fps
            while self._running:
                t0 = time.monotonic()
                ret, image = cap.read()
                if not ret:
                    if is_file:
                        if self._cfg.loop:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        logger.info("Camera: end of video file — stopping (%d frames)",
                                    self._frame_id)
                        break
                    logger.warning("Camera: frame grab failed — skipping")
                    await asyncio.sleep(0.01)
                    continue
                focal = self._focal_for(image.shape[1])
                self._adopt_frame_size(image)
                frame = Frame(
                    image=image,
                    timestamp=time.monotonic(),
                    frame_id=self._frame_id,
                    focal_length_px=focal,
                )
                await self._bus.publish(Channel.PERCEPTION_FRAME, frame)
                self._frame_id += 1
                elapsed = time.monotonic() - t0
                await asyncio.sleep(max(0.0, interval - elapsed))
        finally:
            cap.release()
            src = self._cfg.source if is_file else f"device {self._cfg.device_index}"
            logger.info("Camera: released %s", src)
