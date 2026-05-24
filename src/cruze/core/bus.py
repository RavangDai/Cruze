"""
Async pub/sub event bus.

Design contract:
- Modules subscribe to channels; they never import each other.
- Queues are bounded. When full, the oldest item is dropped so that
  real-time data (frames, detections) is never stale in the buffer.
- Channel names live on the Channel class — use those constants, not
  raw strings, so a typo is a NameError at startup instead of silent silence.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class Channel:
    """All channel name constants."""

    PERCEPTION_FRAME = "perception.frame"
    PERCEPTION_DETECTIONS = "perception.detections"
    PERCEPTION_TRACKS = "perception.tracks"
    TELEMETRY_VEHICLE_STATE = "telemetry.vehicle_state"
    REASONING_SCENE = "reasoning.scene"
    REASONING_EVENT = "reasoning.event"
    VOICE_WAKE = "voice.wake"
    VOICE_QUERY = "voice.query"
    VOICE_UTTERANCE = "voice.utterance"


class _BoundedDropOldestQueue(asyncio.Queue):
    """asyncio.Queue that drops the oldest item instead of blocking when full."""

    def put_nowait(self, item: Any) -> None:
        if self.full():
            try:
                dropped = self.get_nowait()
                logger.debug("Bus overflow — dropped oldest item: %s", type(dropped).__name__)
            except asyncio.QueueEmpty:
                pass
        super().put_nowait(item)

    async def put(self, item: Any) -> None:  # type: ignore[override]
        self.put_nowait(item)


class EventBus:
    """
    Central message broker.

    Usage::

        bus = EventBus()

        # Producer
        await bus.publish(Channel.PERCEPTION_FRAME, frame)

        # Consumer
        queue = bus.subscribe(Channel.PERCEPTION_FRAME, maxsize=4)
        frame = await queue.get()
    """

    def __init__(self) -> None:
        # channel -> list of subscriber queues
        self._subscribers: dict[str, list[_BoundedDropOldestQueue]] = {}

    def subscribe(self, channel: str, maxsize: int = 8) -> _BoundedDropOldestQueue:
        """Return a dedicated queue that will receive all messages on *channel*."""
        q: _BoundedDropOldestQueue = _BoundedDropOldestQueue(maxsize=maxsize)
        self._subscribers.setdefault(channel, []).append(q)
        logger.debug("New subscriber on channel '%s' (maxsize=%d)", channel, maxsize)
        return q

    async def publish(self, channel: str, message: Any) -> None:
        """Fan out *message* to all subscribers on *channel*."""
        queues = self._subscribers.get(channel, [])
        for q in queues:
            q.put_nowait(message)

    def subscriber_count(self, channel: str) -> int:
        return len(self._subscribers.get(channel, []))
