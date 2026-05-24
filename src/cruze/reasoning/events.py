"""
Rule engine — Scenes in, DrivingEvents out.

Each rule is a function that inspects a Scene and returns a DrivingEvent or
None. The engine runs all rules every cycle but enforces per-kind cooldowns
so Cruze doesn't repeat itself.

Publishes:
  Channel.REASONING_EVENT — DrivingEvent (one per fired rule per cooldown window)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from cruze.core.bus import Channel, EventBus
from cruze.core.types import DrivingEvent, EventLevel, Scene
from cruze.reasoning import threat

if TYPE_CHECKING:
    from cruze.core.config import Config

logger = logging.getLogger(__name__)


class EventEngine:
    """
    Subscribes to REASONING_SCENE, evaluates rules, publishes DrivingEvents.

    Parameters
    ----------
    cfg:
        Full application config.
    bus:
        Shared event bus.
    """

    def __init__(self, cfg: "Config", bus: EventBus) -> None:
        self._cfg = cfg
        self._bus = bus
        self._cooldowns: dict[str, float] = {}  # kind → last_fired_timestamp
        self._running = False

    async def run(self) -> None:
        self._running = True
        queue = self._bus.subscribe(Channel.REASONING_SCENE, maxsize=4)
        logger.info("EventEngine started")

        while self._running:
            try:
                scene: Scene = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            events = self._evaluate(scene)
            for event in events:
                await self._bus.publish(Channel.REASONING_EVENT, event)

    async def stop(self) -> None:
        self._running = False

    def _evaluate(self, scene: Scene) -> list[DrivingEvent]:
        """Run all rules; apply cooldowns; return events to publish."""
        now = time.monotonic()
        fired: list[DrivingEvent] = []

        for kind, level, context in self._rules(scene):
            last = self._cooldowns.get(kind, 0.0)
            cooldown = self._cfg.reasoning.event_cooldown_s
            # Critical events use half the normal cooldown.
            if level == EventLevel.CRITICAL:
                cooldown /= 2
            if now - last >= cooldown:
                self._cooldowns[kind] = now
                fired.append(DrivingEvent(kind=kind, level=level, context=context))
                logger.debug("Event fired: kind=%s level=%s", kind, level.value)

        return fired

    def _rules(self, scene: Scene) -> list[tuple[str, EventLevel, dict]]:
        """Return list of (kind, level, context) for every triggered rule."""
        results = []
        cfg = self._cfg.reasoning

        # --- Forward collision warning ---
        if threat.is_forward_collision_warning(scene, cfg):
            lead = scene.lead_track
            from cruze.reasoning.threat import time_to_collision
            ttc = time_to_collision(
                lead.distance_m, lead.closing_speed_mps  # type: ignore[arg-type]
            )
            results.append((
                "fcw",
                EventLevel.CRITICAL,
                {
                    "ttc_s": round(ttc, 1),
                    "distance_m": round(lead.distance_m, 1),  # type: ignore[arg-type]
                    "lead_class": lead.cls.value,
                },
            ))

        # --- Tailgating ---
        if threat.is_tailgating(scene, cfg):
            lead = scene.lead_track
            ego_speed = scene.vehicle_state.speed_mps
            gap = threat.following_gap_seconds(lead.distance_m, ego_speed)  # type: ignore
            results.append((
                "tailgating",
                EventLevel.WARNING,
                {
                    "gap_s": round(gap, 1),
                    "distance_m": round(lead.distance_m, 1),  # type: ignore[arg-type]
                    "speed_mps": round(ego_speed, 1),  # type: ignore[arg-type]
                },
            ))

        # --- Speeding ---
        if threat.is_speeding(scene, cfg):
            ego_speed = scene.vehicle_state.speed_mps
            posted = scene.vehicle_state.posted_speed_limit_mps
            overage_mph = (ego_speed - posted) * 2.237  # type: ignore[operator]
            results.append((
                "speeding",
                EventLevel.NOTICE,
                {
                    "ego_speed_mps": round(ego_speed, 1),  # type: ignore[arg-type]
                    "posted_mps": round(posted, 1),  # type: ignore[arg-type]
                    "overage_mph": round(overage_mph, 1),
                },
            ))

        # --- Slow lead vehicle ---
        if threat.is_slow_lead(scene, cfg):
            results.append((
                "slow_lead",
                EventLevel.INFO,
                {
                    "distance_m": round(scene.lead_track.distance_m, 1),  # type: ignore
                },
            ))

        # --- Stop sign ---
        if threat.stop_sign_present(scene):
            results.append((
                "stop_sign",
                EventLevel.NOTICE,
                {},
            ))

        return results
