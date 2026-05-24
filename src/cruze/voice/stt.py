"""
Speech-to-text service.

Listens after a wake word event, captures ~5 seconds of audio, transcribes
with faster-whisper, publishes the text to Channel.VOICE_QUERY.

Stub mode: never transcribes (no microphone needed in CI).
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from cruze.core.bus import Channel, EventBus

if TYPE_CHECKING:
    from cruze.core.config import VoiceConfig

logger = logging.getLogger(__name__)

_RECORD_SECONDS = 5
_SAMPLE_RATE = 16000


class STTService:
    def __init__(self, cfg: "VoiceConfig", bus: EventBus) -> None:
        self._cfg = cfg
        self._bus = bus
        self._running = False

    async def run(self) -> None:
        self._running = True
        if self._cfg.stt_backend == "stub":
            logger.info("STT: stub backend — queries never fire")
            while self._running:
                await asyncio.sleep(1.0)
            return

        wake_q = self._bus.subscribe(Channel.VOICE_WAKE, maxsize=2)
        model = self._load_model()

        while self._running:
            try:
                await asyncio.wait_for(wake_q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            logger.debug("STT: wake detected — recording")
            audio = await asyncio.get_event_loop().run_in_executor(
                None, self._record
            )
            if audio is None:
                continue

            text = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._transcribe(model, audio)
            )
            if text:
                logger.info("STT: transcribed '%s'", text)
                await self._bus.publish(Channel.VOICE_QUERY, text)

    async def stop(self) -> None:
        self._running = False

    def _load_model(self):
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "faster-whisper required. Install with: pip install faster-whisper"
            ) from exc
        return WhisperModel(self._cfg.whisper_model, device="cpu", compute_type="int8")

    def _record(self) -> "list | None":
        try:
            import pyaudio  # type: ignore
            import numpy as np  # type: ignore
        except ImportError:
            logger.warning("STT: pyaudio/numpy not installed — cannot record")
            return None

        pa = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=_SAMPLE_RATE, input=True)
        frames = []
        for _ in range(int(_SAMPLE_RATE / 1024 * _RECORD_SECONDS)):
            frames.append(stream.read(1024, exception_on_overflow=False))
        stream.stop_stream()
        stream.close()
        pa.terminate()
        import numpy as np
        audio = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32768.0
        return audio

    def _transcribe(self, model, audio) -> str:
        segments, _ = model.transcribe(audio, beam_size=5, language="en")
        return " ".join(s.text.strip() for s in segments).strip()
