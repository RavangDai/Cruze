"""
Text-to-speech service using Piper (offline neural TTS).

Subscribes to Channel.VOICE_UTTERANCE and synthesises audio.
For low latency, audio is synthesised sentence-by-sentence and streamed to
the speaker while the remainder is still being generated.

Stub mode: logs the utterance instead of speaking.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING

from cruze.core.bus import Channel, EventBus
from cruze.core.types import Utterance

if TYPE_CHECKING:
    from cruze.core.config import VoiceConfig

logger = logging.getLogger(__name__)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class TTSService:
    def __init__(self, cfg: "VoiceConfig", bus: EventBus) -> None:
        self._cfg = cfg
        self._bus = bus
        self._running = False
        self._currently_speaking = False

    async def run(self) -> None:
        self._running = True
        queue = self._bus.subscribe(Channel.VOICE_UTTERANCE, maxsize=4)

        if self._cfg.tts_backend != "stub":
            self._piper = self._load_piper()
        else:
            self._piper = None
            logger.info("TTS: stub mode — utterances logged, not spoken")

        while self._running:
            try:
                utterance: Utterance = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if utterance.interrupt and self._currently_speaking:
                logger.debug("TTS: interrupt requested")
                # In a real impl: kill the current audio playback subprocess.

            await self._speak(utterance)

    async def stop(self) -> None:
        self._running = False

    async def _speak(self, utterance: Utterance) -> None:
        if self._piper is None:
            logger.info("[Cruze says] %s", utterance.text)
            return

        sentences = _SENTENCE_END.split(utterance.text)
        self._currently_speaking = True
        try:
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda s=sentence: self._synthesise_and_play(s)
                )
        finally:
            self._currently_speaking = False

    def _load_piper(self):
        try:
            from piper import PiperVoice  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "piper-tts required. See: https://github.com/rhasspy/piper"
            ) from exc
        return PiperVoice.load(self._cfg.piper_model_path)

    def _synthesise_and_play(self, text: str) -> None:
        try:
            import pyaudio  # type: ignore
            pa = pyaudio.PyAudio()
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self._piper.config.sample_rate,
                output=True,
            )
            for audio_bytes in self._piper.synthesize_stream_raw(text):
                stream.write(audio_bytes)
            stream.stop_stream()
            stream.close()
            pa.terminate()
        except Exception as exc:
            logger.error("TTS synthesis error: %s", exc)
