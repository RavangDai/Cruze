"""
Orchestrator — wires all services to the event bus and manages the lifecycle.

Startup sequence:
  1. Load config (layered YAML + env vars).
  2. Initialise structured logging.
  3. Create EventBus.
  4. Instantiate and start services as asyncio tasks.
  5. Wait for SIGINT / SIGTERM (or KeyboardInterrupt on Windows).
  6. Send stop signals to all services; await graceful shutdown.

Adding a new service:
  1. Instantiate it here with (cfg, bus) args.
  2. Add it to the `tasks` list with `asyncio.create_task(service.run())`.
  3. Add it to the `services` list for stop() calls.
  No other files need to change.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from typing import Any

from cruze.core.bus import EventBus
from cruze.core.config import Config, load_config
from cruze.core.logging import configure_logging
from cruze.core.types import Frame
from cruze.perception.camera import CameraService
from cruze.perception import depth as depth_mod
from cruze.perception.detector import load as load_detector
from cruze.perception.pipeline import PerceptionService
from cruze.telemetry.vehicle_state import VehicleStateService
from cruze.reasoning.scene import SceneAssembler
from cruze.reasoning.events import EventEngine
from cruze.personality.persona import PersonaService
from cruze.voice.wake import WakeWordService
from cruze.voice.stt import STTService
from cruze.voice.tts import TTSService
from cruze.hmi.hud import HUDService
from cruze.hmi.webapp import DashboardService

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Top-level service controller.

    Parameters
    ----------
    cfg:
        Fully loaded Config object.
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._bus = EventBus()
        self._tasks: list[asyncio.Task] = []
        self._services: list[Any] = []

    async def run(self) -> None:
        cfg = self._cfg
        bus = self._bus

        logger.info("Cruze starting — hardware_profile=%s", cfg.hardware_profile)

        # Compute focal length once; embed in every Frame via CameraService.
        focal_px = depth_mod.focal_length_px(cfg.camera.width, cfg.camera.hfov_deg)
        logger.debug("Focal length: %.1f px (width=%d, hfov=%.1f°)",
                     focal_px, cfg.camera.width, cfg.camera.hfov_deg)

        # --- Instantiate services ---
        camera_svc = CameraService(cfg.camera, bus, focal_length_px=focal_px)

        detector = load_detector(
            cfg.perception.backend,
            model_path=cfg.perception.model_path,
            confidence_threshold=cfg.perception.confidence_threshold,
        )
        perception_svc = PerceptionService(cfg, bus, detector)

        telemetry_svc = VehicleStateService(cfg, bus)
        scene_svc = SceneAssembler(cfg, bus, image_width=cfg.camera.width)
        event_svc = EventEngine(cfg, bus)
        persona_svc = PersonaService(cfg, bus)
        wake_svc = WakeWordService(cfg.voice, bus)
        stt_svc = STTService(cfg.voice, bus)
        tts_svc = TTSService(cfg.voice, bus)
        hud_svc = HUDService(cfg, bus)
        dashboard_svc = DashboardService(cfg, bus)

        self._services = [
            camera_svc, perception_svc, telemetry_svc,
            scene_svc, event_svc, persona_svc,
            wake_svc, stt_svc, tts_svc, hud_svc, dashboard_svc,
        ]

        # --- Start tasks ---
        self._tasks = [
            asyncio.create_task(svc.run(), name=type(svc).__name__)
            for svc in self._services
        ]

        # Surface service crashes immediately; without this a failed task is
        # silent until shutdown (gather happens with return_exceptions=True).
        def _log_task_failure(task: asyncio.Task) -> None:
            if task.cancelled():
                return
            exc = task.exception()
            if exc is not None:
                logger.error(
                    "Service task %s died: %s", task.get_name(), exc, exc_info=exc
                )

        for task in self._tasks:
            task.add_done_callback(_log_task_failure)

        logger.info("All services started (%d tasks)", len(self._tasks))

        # --- Set up shutdown triggers ---
        stop_event = asyncio.Event()
        self._stop_event = stop_event

        def _signal_handler() -> None:
            logger.info("Shutdown signal received")
            stop_event.set()

        loop = asyncio.get_running_loop()
        # SIGTERM exists on all POSIX; Windows only has SIGINT reliably.
        for sig in (signal.SIGINT,):
            try:
                loop.add_signal_handler(sig, _signal_handler)
            except (NotImplementedError, OSError):
                # Windows: signal handlers must be set via signal.signal().
                pass

        # On Windows, KeyboardInterrupt is raised in the main thread; wrap
        # the wait so Ctrl+C still triggers clean shutdown.
        try:
            await stop_event.wait()
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt — shutting down")

        await self.shutdown()

    async def shutdown(self) -> None:
        logger.info("Cruze shutting down")
        for svc in reversed(self._services):
            if hasattr(svc, "stop"):
                try:
                    await svc.stop()
                except Exception:
                    logger.exception("Error stopping %s", type(svc).__name__)

        for task in self._tasks:
            task.cancel()

        await asyncio.gather(*self._tasks, return_exceptions=True)
        logger.info("Cruze stopped cleanly")

    def request_stop(self) -> None:
        """Request shutdown from outside the asyncio loop (e.g. a test)."""
        if hasattr(self, "_stop_event"):
            self._stop_event.set()


def build_and_run(
    hardware_profile: str = "desktop",
    no_camera: bool = False,
    log_level: str | None = None,
    video: str | None = None,
    loop: bool = False,
) -> None:
    """
    Load config, configure logging, and run the orchestrator.
    Called by __main__.py.
    """
    cfg = load_config(hardware_profile=hardware_profile)

    if no_camera:
        cfg.camera.simulated = True

    if video:
        # A video file replaces both the live camera and simulated frames.
        cfg.camera.simulated = False
        cfg.camera.source = video
        cfg.camera.loop = loop

    if log_level:
        cfg.logging.level = log_level.upper()

    configure_logging(cfg.logging)

    orchestrator = Orchestrator(cfg)

    try:
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        pass
