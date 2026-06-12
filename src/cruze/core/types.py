"""
Shared data contract. Every other module imports from here.

Rules:
- Add optional fields freely (backwards-compatible).
- Never rename or remove fields — existing subscribers break silently.
- All dataclasses are frozen so they're safe to share across async tasks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class ObjectClass(Enum):
    CAR = "car"
    TRUCK = "truck"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"
    BICYCLE = "bicycle"
    PERSON = "person"
    STOP_SIGN = "stop_sign"
    TRAFFIC_LIGHT = "traffic_light"
    UNKNOWN = "unknown"


class EventLevel(Enum):
    INFO = "info"
    NOTICE = "notice"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class BBox:
    """Bounding box in pixel coordinates (top-left origin)."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    def iou(self, other: BBox) -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


@dataclass(frozen=True)
class Detection:
    """Single detector output for one object in one frame."""

    bbox: BBox
    confidence: float
    cls: ObjectClass
    # Distance estimate from monocular depth; None if not computed.
    distance_m: float | None = None


@dataclass(frozen=True)
class Track:
    """Tracked object with stable ID across frames."""

    track_id: int
    bbox: BBox
    cls: ObjectClass
    # Smoothed distance estimate; None when depth unavailable.
    distance_m: float | None = None
    # Positive = closing (approaching ego), negative = receding.
    closing_speed_mps: float | None = None
    # Frames since last matched detection (0 = matched this frame).
    age_missed: int = 0
    # Timestamp of the frame this track was last updated.
    timestamp: float = field(default_factory=time.monotonic)
    # Absolute ground speed estimate (ego speed − closing speed); None when
    # either input is unavailable. Valid for same-direction traffic only.
    speed_mps: float | None = None


@dataclass(frozen=True)
class Frame:
    """Raw camera frame off the wire."""

    # HxWxC uint8 numpy array — callers import numpy lazily.
    image: Any
    timestamp: float = field(default_factory=time.monotonic)
    frame_id: int = 0
    # Intrinsic: focal length in pixels, derived from config at startup.
    focal_length_px: float | None = None


@dataclass(frozen=True)
class VehicleState:
    """Fused snapshot from OBD, GPS, IMU."""

    timestamp: float = field(default_factory=time.monotonic)
    speed_mps: float | None = None        # fused: OBD > GPS Doppler > GPS position
    speed_mps_source: str = "unknown"     # "obd" | "gps" | "gps_pos" | "simulated"
    heading_deg: float | None = None      # 0=N, clockwise
    latitude: float | None = None
    longitude: float | None = None
    altitude_m: float | None = None
    acceleration_mps2: float | None = None
    posted_speed_limit_mps: float | None = None  # from maps module; None if unknown
    # Raw GPS Doppler speed (RMC speed-over-ground), kept separate from the
    # fused speed_mps for transparency/debugging. Do not promote to speed_mps
    # in consumers — the telemetry fusion layer decides which source wins.
    gps_speed_mps: float | None = None


@dataclass(frozen=True)
class Scene:
    """Aggregated snapshot handed to the reasoning layer each cycle."""

    timestamp: float = field(default_factory=time.monotonic)
    tracks: tuple[Track, ...] = ()
    vehicle_state: VehicleState = field(default_factory=VehicleState)
    # Nearest lead vehicle (same lane, ahead) if any.
    lead_track: Track | None = None


@dataclass(frozen=True)
class DrivingEvent:
    """Emitted by the rule engine; consumed by the personality layer."""

    kind: str          # "speeding" | "tailgating" | "fcw" | "stop_sign" | ...
    level: EventLevel
    timestamp: float = field(default_factory=time.monotonic)
    # Structured payload for the personality layer to template into utterances.
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Utterance:
    """What Cruze wants to say, ready for TTS."""

    text: str
    priority: int = 5        # 1=highest (FCW), 10=lowest (casual comment)
    timestamp: float = field(default_factory=time.monotonic)
    # If True, interrupt whatever is currently playing.
    interrupt: bool = False
