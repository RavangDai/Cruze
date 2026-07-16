"""
Dataclass → JSON-safe dict converters for the dashboard WebSocket protocol.

Pure functions, no heavy imports — unit-tested without fastapi installed.
Every payload carries a "type" discriminator the frontend switches on.
"""

from __future__ import annotations

from typing import Any

from cruze.core.types import (
    DrivingEvent,
    LaneLine,
    Lanes,
    Scene,
    Track,
    Utterance,
    VehicleState,
)


def _r(value: float | None, ndigits: int = 2) -> float | None:
    """Round for wire compactness; pass None through."""
    return None if value is None else round(value, ndigits)


def track_to_dict(t: Track) -> dict[str, Any]:
    return {
        "id": t.track_id,
        "cls": t.cls.value,
        "bbox": [round(t.bbox.x1, 1), round(t.bbox.y1, 1),
                 round(t.bbox.x2, 1), round(t.bbox.y2, 1)],
        "distance_m": _r(t.distance_m, 1),
        "closing_mps": _r(t.closing_speed_mps),
        "speed_mps": _r(t.speed_mps),
        "light": t.light_state.value if t.light_state is not None else None,
        # Flat [x1,y1,x2,y2,...] halves the JSON overhead vs nested pairs.
        "mask": (
            [round(coord, 1) for point in t.mask_xy for coord in point]
            if t.mask_xy is not None else None
        ),
    }


def _lane_line_to_list(line: LaneLine | None) -> list[float] | None:
    if line is None:
        return None
    return [round(line.x1, 1), round(line.y1, 1), round(line.x2, 1), round(line.y2, 1)]


# Defensive cap on polyline points per side (producer sends 8): bounds wire
# size even if a future producer over-samples.
_MAX_POLY_POINTS = 16


def _poly_to_list(poly: tuple[tuple[float, float], ...] | None) -> list[float] | None:
    if poly is None:
        return None
    return [
        round(coord, 1) for point in poly[:_MAX_POLY_POINTS] for coord in point
    ]


def lanes_to_dict(lanes: Lanes | None) -> dict[str, Any] | None:
    if lanes is None:
        return None
    return {
        "left": _lane_line_to_list(lanes.left),
        "right": _lane_line_to_list(lanes.right),
        "left_poly": _poly_to_list(lanes.left_poly),
        "right_poly": _poly_to_list(lanes.right_poly),
        "cte_m": _r(lanes.cte_m),
    }


def state_to_dict(v: VehicleState) -> dict[str, Any]:
    return {
        "speed_mps": _r(v.speed_mps),
        "source": v.speed_mps_source,
        "gps_speed_mps": _r(v.gps_speed_mps),
        "heading_deg": _r(v.heading_deg, 1),
        "lat": v.latitude,
        "lon": v.longitude,
        "limit_mps": _r(v.posted_speed_limit_mps),
    }


def scene_to_dict(s: Scene) -> dict[str, Any]:
    return {
        "type": "scene",
        "ts": s.timestamp,
        "tracks": [track_to_dict(t) for t in s.tracks],
        "lead_id": s.lead_track.track_id if s.lead_track is not None else None,
        "state": state_to_dict(s.vehicle_state),
        "lanes": lanes_to_dict(s.lanes),
        # IDM urgency scalar; the frontend colours the corridor with it.
        "accel": _r(s.required_accel_mps2),
    }


def event_to_dict(e: DrivingEvent) -> dict[str, Any]:
    return {
        "type": "event",
        "kind": e.kind,
        "level": e.level.value,
        "ts": e.timestamp,
        "context": e.context,
    }


def utterance_to_dict(u: Utterance) -> dict[str, Any]:
    return {"type": "utterance", "text": u.text, "priority": u.priority}
