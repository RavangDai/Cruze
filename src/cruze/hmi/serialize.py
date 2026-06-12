"""
Dataclass → JSON-safe dict converters for the dashboard WebSocket protocol.

Pure functions, no heavy imports — unit-tested without fastapi installed.
Every payload carries a "type" discriminator the frontend switches on.
"""

from __future__ import annotations

from typing import Any

from cruze.core.types import DrivingEvent, Scene, Track, Utterance, VehicleState


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
