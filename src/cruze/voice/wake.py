"""
Wake word detection.

Backends:
  stub           — never fires; for CI / --no-mic mode
  openwakeword   — open-source, runs on CPU
  porcupine      — Picovoice (requires API key in PICOVOICE_ACCESS_KEY env var)

Publishes:
  Channel.VOICE_WAKE — True (just a signal; no payload)
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING

from cruze.core.bus import Channel, EventBus

if TYPE_CHECKING:
    from cruze.core.config import VoiceConfig

logger = logging.getLogger(__name__)


class WakeWordService:
    def __init__(self, cfg: "VoiceConfig", bus: EventBus) -> None:
        self._cfg = cfg
        self._bus = bus
        self._running = False

    async def run(self) -> None:
        self._running = True
        backend = self._cfg.wake_backend.lower()

        if backend == "stub":
            logger.info("WakeWord: stub backend — wake word never fires (set wake_backend to openwakeword for real use)")
            while self._running:
                await asyncio.sleep(1.0)
            return

        if backend == "openwakeword":
            await self._run_openwakeword()
        elif backend == "porcupine":
            await self._run_porcupine()
        else:
            raise ValueError(f"Unknown wake word backend: {backend}")

    async def stop(self) -> None:
        self._running = False

    async def _run_openwakeword(self) -> None:
        try:
            import openwakeword  # type: ignore
            from openwakeword.model import Model  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "openwakeword is required. Install with: pip install openwakeword"
            ) from exc

        model = Model(wakeword_models=["hey_jarvis"])  # swap for custom model
        logger.info("WakeWord: openWakeWord listening for '%s'", self._cfg.wake_word)

        import pyaudio  # type: ignore
        pa = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=16000,
                         input=True, frames_per_buffer=1280)
        try:
            while self._running:
                audio = stream.read(1280, exception_on_overflow=False)
                import numpy as np  # type: ignore
                chunk = np.frombuffer(audio, dtype=np.int16)
                pred = model.predict(chunk)
                if any(v > 0.5 for v in pred.values()):
                    logger.info("Wake word detected")
                    await self._bus.publish(Channel.VOICE_WAKE, True)
                await asyncio.sleep(0)
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

    async def _run_porcupine(self) -> None:
        try:
            import pvporcupine  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "pvporcupine is required. Install with: pip install pvporcupine"
            ) from exc

        access_key = os.environ.get("PICOVOICE_ACCESS_KEY", "")
        if not access_key:
            raise RuntimeError(
                "PICOVOICE_ACCESS_KEY env var required for porcupine backend"
            )
        handle = pvporcupine.create(access_key=access_key, keywords=["ok google"])
        logger.info("WakeWord: Porcupine listening")

        import pyaudio  # type: ignore
        pa = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=handle.sample_rate,
                         input=True, frames_per_buffer=handle.frame_length)
        try:
            while self._running:
                pcm = stream.read(handle.frame_length, exception_on_overflow=False)
                import struct
                pcm_unpacked = struct.unpack_from("h" * handle.frame_length, pcm)
                result = handle.process(pcm_unpacked)
                if result >= 0:
                    await self._bus.publish(Channel.VOICE_WAKE, True)
                await asyncio.sleep(0)
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()
            handle.delete()
