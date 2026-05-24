"""
Persona layer — decides what Cruze says and when.

Subscribes to:
  Channel.REASONING_EVENT — DrivingEvent
  Channel.VOICE_QUERY     — transcribed user query (string)
  Channel.REASONING_SCENE — Scene (kept for dialog context)

Publishes:
  Channel.VOICE_UTTERANCE — Utterance

Decision logic:
  1. VOICE_QUERY → always respond (user explicitly asked).
  2. REASONING_EVENT → check priority and global speech cooldown before speaking.
     Critical events skip the cooldown.
  3. Response source:
     - offline_mode=True or API failure → canned line from responses.py
     - Otherwise → DialogEngine (with scene context)

Priority mapping (lower = higher priority):
  CRITICAL → 1  (FCW, always interrupt)
  WARNING  → 3
  NOTICE   → 5
  INFO     → 8
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from cruze.core.bus import Channel, EventBus
from cruze.core.types import DrivingEvent, EventLevel, Scene, Utterance
from cruze.personality.responses import pick
from cruze.voice.dialog import DialogEngine

if TYPE_CHECKING:
    from cruze.core.config import Config

logger = logging.getLogger(__name__)

_PRIORITY_MAP = {
    EventLevel.CRITICAL: 1,
    EventLevel.WARNING: 3,
    EventLevel.NOTICE: 5,
    EventLevel.INFO: 8,
}

_SPEECH_COOLDOWN_S = 4.0   # global: don't overlap utterances


class PersonaService:
    """
    Routes events and queries to the right response source and publishes utterances.
    """

    def __init__(self, cfg: "Config", bus: EventBus) -> None:
        self._cfg = cfg
        self._bus = bus
        self._dialog = DialogEngine(cfg.voice)
        self._latest_scene: Scene | None = None
        self._recent_events: list[DrivingEvent] = []
        self._last_speech_ts: float = 0.0
        self._running = False

    async def run(self) -> None:
        self._running = True
        event_q = self._bus.subscribe(Channel.REASONING_EVENT, maxsize=8)
        query_q = self._bus.subscribe(Channel.VOICE_QUERY, maxsize=4)
        scene_q = self._bus.subscribe(Channel.REASONING_SCENE, maxsize=2)

        logger.info("PersonaService started")

        async def drain_scenes() -> None:
            while self._running:
                try:
                    self._latest_scene = await asyncio.wait_for(scene_q.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    pass

        asyncio.ensure_future(drain_scenes())

        while self._running:
            # Wait for either an event or a query.
            done, _ = await asyncio.wait(
                [
                    asyncio.ensure_future(event_q.get()),
                    asyncio.ensure_future(query_q.get()),
                ],
                timeout=1.0,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                item = task.result()
                if isinstance(item, DrivingEvent):
                    await self._handle_event(item)
                elif isinstance(item, str):
                    await self._handle_query(item)

    async def stop(self) -> None:
        self._running = False

    async def _handle_event(self, event: DrivingEvent) -> None:
        self._recent_events.append(event)
        if len(self._recent_events) > 20:
            self._recent_events.pop(0)

        now = time.monotonic()
        elapsed = now - self._last_speech_ts
        is_critical = event.level == EventLevel.CRITICAL

        if not is_critical and elapsed < _SPEECH_COOLDOWN_S:
            return

        priority = _PRIORITY_MAP.get(event.level, 5)
        text = await self._generate_event_response(event)
        if text:
            utterance = Utterance(
                text=text,
                priority=priority,
                interrupt=is_critical,
            )
            await self._bus.publish(Channel.VOICE_UTTERANCE, utterance)
            self._last_speech_ts = time.monotonic()

    async def _handle_query(self, query: str) -> None:
        logger.info("Persona: responding to query '%s'", query)
        text = await self._generate_query_response(query)
        if text:
            await self._bus.publish(Channel.VOICE_UTTERANCE, Utterance(text=text, priority=5))
            self._last_speech_ts = time.monotonic()

    async def _generate_event_response(self, event: DrivingEvent) -> str | None:
        if self._cfg.voice.offline_mode:
            return pick(event.kind, event.context)
        try:
            prompt = f"Driving event: {event.kind}. Context: {event.context}. Say something appropriate."
            return await self._dialog.respond(
                prompt, self._latest_scene, self._recent_events
            )
        except Exception as exc:
            logger.warning("Dialog API failed (%s) — using canned response", exc)
            return pick(event.kind, event.context)

    async def _generate_query_response(self, query: str) -> str | None:
        if self._cfg.voice.offline_mode:
            return pick("generic")
        try:
            return await self._dialog.respond(
                query, self._latest_scene, self._recent_events
            )
        except Exception as exc:
            logger.warning("Dialog API failed (%s) — using canned response", exc)
            return pick("generic")
