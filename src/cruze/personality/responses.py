"""
Canned response library — offline-safe, no LLM required.

Each entry maps an event kind to a list of lines. The persona layer picks
one at random. Lines use {placeholders} matching DrivingEvent.context keys.

Rules for writing new lines:
- Roast the SITUATION, never the driver.
- Safety lines (fcw) must be clear and unambiguous first, wit second.
- Max 12 words per line (TTS playback time).
- No sarcasm about danger. No jokes about accidents or injuries.
"""

from __future__ import annotations

import random

_CANNED: dict[str, list[str]] = {
    "fcw": [
        "Heads up — something's very close ahead.",
        "Easy there, we've got company up front.",
        "Distance check: too close, backing off now.",
    ],
    "tailgating": [
        "Two-second rule. We're behind on the math.",
        "That gap's a bit cosy. Backing off helps.",
        "Following distance is a suggestion we should take.",
    ],
    "speeding": [
        "Speed limit's {posted_mph} here. We're doing {ego_mph}.",
        "Posted limit says {posted_mph}. Worth a nudge on the brakes.",
        "Cameras love speeders. Limit's {posted_mph} through here.",
    ],
    "slow_lead": [
        "That car up front found a different gear.",
        "Lead vehicle is really enjoying the scenery.",
        "Mystery slowdown ahead — the classics never get old.",
    ],
    "stop_sign": [
        "Stop sign coming up.",
        "Red octagon ahead — full stop territory.",
        "Stop sign. We're stopping.",
    ],
    "brake_hard": [
        "Brake now — closing fast on traffic ahead.",
        "Hard brake — the gap is collapsing.",
        "Brake hard. Too fast for this gap.",
    ],
    "brake_advised": [
        "Ease off — we're gaining on the car ahead.",
        "Coasting would be smart right about now.",
        "Gap's shrinking. Lifting off keeps it civil.",
    ],
    "lane_departure": [
        "Drifting {side} — nudge us back to centre.",
        "We're wandering {side} out of lane.",
        "Lane check: easing back from the {side} edge.",
    ],
    "cut_in": [
        "Someone just merged in front — easing back.",
        "New neighbour up front. Rebuilding the gap.",
        "That merge was ambitious. Giving it room.",
    ],
    "generic": [
        "All good up here.",
        "Nothing dramatic at the moment.",
        "Road looks clear.",
    ],
}


def pick(kind: str, context: dict | None = None) -> str:
    """
    Return a randomly selected canned line for *kind*, with context filled in.
    Falls back to the 'generic' pool if *kind* is unknown.
    """
    pool = _CANNED.get(kind, _CANNED["generic"])
    line = random.choice(pool)
    if context:
        # Pre-format common display values.
        display_ctx = dict(context)
        for key in ("ego_speed_mps", "posted_mps"):
            if key in display_ctx:
                mph_key = key.replace("_mps", "_mph")
                display_ctx[mph_key] = round(float(display_ctx[key]) * 2.237)
        try:
            line = line.format(**display_ctx)
        except KeyError:
            pass
    return line
