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


class TrafficLightState(Enum):
    """Lamp state of a detected traffic light. UNKNOWN = unlit/occluded/ambiguous."""

    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    UNKNOWN = "unknown"


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
    # Instance-mask outline polygon in image pixel coords, downsampled at the
    # detector boundary; None for box-only backends/weights.
    mask_xy: tuple[tuple[float, float], ...] | None = None
    # Lamp state; set only for TRAFFIC_LIGHT detections, None otherwise.
    light_state: TrafficLightState | None = None


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
    # Latest instance-mask outline from the matched detection; None if the
    # detector emits no masks or the track went unmatched this frame.
    mask_xy: tuple[tuple[float, float], ...] | None = None
    # Debounced lamp state; set only for TRAFFIC_LIGHT tracks.
    light_state: TrafficLightState | None = None
    # Frame this track was last updated from. Carried end-to-end so the
    # dashboard can pin an overlay to the exact JPEG it belongs to instead of
    # extrapolating between scenes.
    frame_id: int = 0
    # Ground-plane position in metres relative to the camera: lateral offset
    # (positive = right of centre) and forward range. None when the geometry
    # is unavailable. Feeds the dashboard's bird's-eye plan.
    ground_xz_m: tuple[float, float] | None = None


@dataclass(frozen=True)
class LaneLine:
    """One lane boundary as a pixel-space segment (bottom point first)."""

    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class Lanes:
    """Per-frame lane detection result; either side may be None."""

    left: LaneLine | None = None
    right: LaneLine | None = None
    timestamp: float = field(default_factory=time.monotonic)
    frame_id: int = 0
    # Curved boundary polylines in pixel coords, bottom point first; None
    # when the quadratic fit was unavailable or unstable (straight LaneLine
    # above remains the fallback).
    left_poly: tuple[tuple[float, float], ...] | None = None
    right_poly: tuple[tuple[float, float], ...] | None = None
    # Ego lateral offset from the lane centre in metres, measured at the
    # bottom image row via the ground-plane model; positive = ego right of
    # centre. None when either boundary or camera geometry is missing.
    cte_m: float | None = None


@dataclass(frozen=True)
class EgoEstimate:
    """Per-frame bundle of vision_pilot ONNX net outputs (raw, pre-fusion).
    Mirrors vision_pilot's InferenceFrameResult. Any field may be None/empty
    when its net is disabled or produced no output this frame."""

    timestamp: float = field(default_factory=time.monotonic)
    frame_id: int = 0
    # AutoSpeed vehicle boxes — CIPO/lead candidates (lead SELECTION is SP3).
    cipo_boxes: tuple[Detection, ...] = ()
    # AutoSteer ego-path polyline in raw image px, bottom-first; None if masked out.
    ego_path: tuple[tuple[float, float], ...] | None = None
    # AutoDrive scalars (domain-converted). None until the 2-frame buffer fills.
    cipo_distance_m: float | None = None
    road_curvature_1pm: float | None = None
    cipo_flag: bool | None = None


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
    # Camera frame these tracks came from; 0 when unknown. The dashboard
    # matches this against the frame_id tagged onto each streamed JPEG.
    frame_id: int = 0
    tracks: tuple[Track, ...] = ()
    vehicle_state: VehicleState = field(default_factory=VehicleState)
    # Nearest lead vehicle (same lane, ahead) if any.
    lead_track: Track | None = None
    # Latest fresh lane detection; None when unavailable or stale.
    lanes: Lanes | None = None
    # IDM-required longitudinal acceleration in m/s²; negative = braking
    # needed, None when ego speed is unknown. Computed by SceneAssembler so
    # the HUD corridor colour and spoken events derive from the same number.
    required_accel_mps2: float | None = None
    # vision_pilot ONNX net outputs, folded from the PERCEPTION_EGO bundle by
    # SceneAssembler. All None when the nets are disabled (the default).
    ego_path: tuple[tuple[float, float], ...] | None = None
    road_curvature_1pm: float | None = None
    cipo_distance_m: float | None = None
    cipo_flag: bool | None = None


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
