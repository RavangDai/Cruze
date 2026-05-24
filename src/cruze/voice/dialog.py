"""
LLM dialog via Anthropic Claude API.

Called by the persona layer when a query arrives or when it wants a
contextual, situationally-aware response rather than a canned line.

System prompt includes:
  - Cruze's personality charter
  - Current scene state (speed, limit, lead vehicle, events)
  - Conversation history (last N turns)

Falls back to canned responses when offline_mode=True or on API failure.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cruze.core.config import VoiceConfig
    from cruze.core.types import DrivingEvent, Scene

logger = logging.getLogger(__name__)

_PERSONALITY_CHARTER = """
You are Cruze, an AI co-pilot built into a car dashboard.

Personality:
- Warm, witty, and situationally aware. You notice what's happening on the road.
- You roast SITUATIONS — slow lead vehicles, mystery construction cones, GPS rerouting
  drama — never the driver personally, never along demographic lines.
- Safety warnings are delivered clearly and immediately, without sarcasm.
- Casual remarks are concise (one sentence). You don't monologue.
- You're knowledgeable about driving physics but you explain things in plain English.
- When asked a question, you answer it. You don't hedge or over-explain.

Hard rules:
- Never comment on the driver's skill, intelligence, age, gender, or any personal trait.
- Never make jokes about accidents, injuries, or death.
- Safety-critical warnings (FCW, stop sign) always take priority over wit.
- If you don't know something, say so briefly.
""".strip()


def _build_scene_context(scene: "Scene | None", recent_events: list["DrivingEvent"]) -> str:
    if scene is None:
        return "Scene: no data yet."

    vs = scene.vehicle_state
    parts = []

    speed_mps = vs.speed_mps
    if speed_mps is not None:
        speed_mph = speed_mps * 2.237
        parts.append(f"Ego speed: {speed_mph:.0f} mph ({speed_mps:.1f} m/s) [{vs.speed_mps_source}]")
    else:
        parts.append("Ego speed: unknown")

    if vs.posted_speed_limit_mps is not None:
        parts.append(f"Posted limit: {vs.posted_speed_limit_mps * 2.237:.0f} mph")
    else:
        parts.append("Posted limit: unknown")

    if scene.lead_track:
        lead = scene.lead_track
        d = f"{lead.distance_m:.0f} m" if lead.distance_m else "unknown distance"
        cs = f", closing {lead.closing_speed_mps:.1f} m/s" if lead.closing_speed_mps else ""
        parts.append(f"Lead vehicle: {lead.cls.value} {d}{cs}")
    else:
        parts.append("Lead vehicle: none detected")

    parts.append(f"Tracks: {len(scene.tracks)} objects")

    if recent_events:
        event_strs = [f"{e.kind} ({e.level.value})" for e in recent_events[-5:]]
        parts.append(f"Recent events: {', '.join(event_strs)}")

    return "\n".join(parts)


class DialogEngine:
    """
    Wraps the Anthropic API to produce contextual responses.

    Parameters
    ----------
    cfg:
        Voice config.
    """

    def __init__(self, cfg: "VoiceConfig") -> None:
        self._cfg = cfg
        self._history: list[dict[str, str]] = []
        self._client: Any = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import anthropic  # type: ignore
            self._client = anthropic.Anthropic()
            return self._client
        except ImportError as exc:
            raise ImportError(
                "anthropic SDK required. Install with: pip install anthropic"
            ) from exc

    async def respond(
        self,
        query: str,
        scene: "Scene | None" = None,
        recent_events: "list[DrivingEvent] | None" = None,
    ) -> str:
        """
        Generate a response to *query* given current scene context.
        Returns the response string.
        Raises on API failure — caller should fall back to canned responses.
        """
        import asyncio

        scene_ctx = _build_scene_context(scene, recent_events or [])
        system_prompt = f"{_PERSONALITY_CHARTER}\n\nCurrent driving context:\n{scene_ctx}"

        self._history.append({"role": "user", "content": query})
        # Keep last 10 turns to stay within context budget.
        history = self._history[-10:]

        client = self._get_client()

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.messages.create(
                model=self._cfg.claude_model,
                max_tokens=256,
                system=system_prompt,
                messages=history,
            ),
        )
        text = response.content[0].text.strip()
        self._history.append({"role": "assistant", "content": text})
        return text

    def clear_history(self) -> None:
        self._history.clear()
