"""SceneAssembler absolute-speed tests — no bus traffic needed."""

import time

import pytest

from cruze.core.bus import EventBus
from cruze.core.config import Config
from cruze.core.types import BBox, EgoEstimate, LaneLine, Lanes, ObjectClass, Track, VehicleState
from cruze.reasoning.scene import SceneAssembler


def _assembler(ego_speed=None):
    sa = SceneAssembler(Config(), EventBus(), image_width=1280)
    sa._latest_vehicle_state = VehicleState(speed_mps=ego_speed)
    return sa


def _track(cls=ObjectClass.CAR, closing=None, dist=30.0, tid=1):
    return Track(
        track_id=tid,
        bbox=BBox(600, 300, 700, 400),
        cls=cls,
        distance_m=dist,
        closing_speed_mps=closing,
    )


def test_absolute_speed_ego_minus_closing():
    sa = _assembler(ego_speed=30.0)
    scene = sa._build_scene([_track(closing=5.0)])
    # Lead closing at 5 m/s while ego does 30 → lead absolute 25 m/s.
    assert scene.tracks[0].speed_mps == pytest.approx(25.0)


def test_absolute_speed_clamped_at_zero():
    sa = _assembler(ego_speed=3.0)
    scene = sa._build_scene([_track(closing=10.0)])
    # Math gives −7 (oncoming/stopped edge case) — clamp to 0 for display sanity.
    assert scene.tracks[0].speed_mps == 0.0


def test_no_ego_speed_means_no_absolute_speed():
    sa = _assembler(ego_speed=None)
    scene = sa._build_scene([_track(closing=5.0)])
    assert scene.tracks[0].speed_mps is None


def test_no_closing_speed_means_no_absolute_speed():
    sa = _assembler(ego_speed=30.0)
    scene = sa._build_scene([_track(closing=None)])
    assert scene.tracks[0].speed_mps is None


def test_person_gets_no_absolute_speed():
    sa = _assembler(ego_speed=30.0)
    scene = sa._build_scene([_track(cls=ObjectClass.PERSON, closing=5.0)])
    assert scene.tracks[0].speed_mps is None


def test_lead_track_carries_absolute_speed():
    sa = _assembler(ego_speed=30.0)
    scene = sa._build_scene([_track(closing=5.0)])
    assert scene.lead_track is not None
    assert scene.lead_track.speed_mps == pytest.approx(25.0)


# --- IDM required-accel attachment ---

def test_required_accel_attached_for_closing_lead():
    sa = _assembler(ego_speed=25.0)
    scene = sa._build_scene([_track(closing=8.0, dist=10.0)])
    assert scene.required_accel_mps2 is not None
    assert scene.required_accel_mps2 < -5.0  # fast-closing 10 m gap = emergency


def test_required_accel_none_without_ego_speed():
    sa = _assembler(ego_speed=None)
    scene = sa._build_scene([_track(closing=5.0)])
    assert scene.required_accel_mps2 is None


def test_required_accel_positive_on_free_road():
    sa = _assembler(ego_speed=10.0)
    scene = sa._build_scene([])
    assert scene.required_accel_mps2 is not None
    assert scene.required_accel_mps2 > 0.0


# --- Lane fold-in ---

def _lanes(age_s=0.0):
    return Lanes(
        left=LaneLine(100.0, 720.0, 550.0, 320.0),
        right=LaneLine(1180.0, 720.0, 730.0, 320.0),
        timestamp=time.monotonic() - age_s,
    )


def test_fresh_lanes_attached_to_scene():
    sa = _assembler()
    sa._latest_lanes = _lanes()
    sa._lanes_seen_at = time.monotonic()
    scene = sa._build_scene([])
    assert scene.lanes is not None
    assert scene.lanes.left.x1 == pytest.approx(100.0)


def test_lanes_attached_despite_old_frame_timestamp():
    """Regression: freshness was anchored to the frame CAPTURE timestamp, so
    on a loaded CPU (perception latency > 0.5 s) every lane result was
    silently dropped. Freshness must mean 'the detector is still producing',
    i.e. message arrival time."""
    sa = _assembler()
    sa._latest_lanes = _lanes(age_s=2.0)  # frame captured 2 s ago
    sa._lanes_seen_at = time.monotonic()  # ...but the message just arrived
    assert sa._build_scene([]).lanes is not None


def test_lanes_dropped_when_detector_stops_producing():
    sa = _assembler()
    sa._latest_lanes = _lanes()
    sa._lanes_seen_at = time.monotonic() - 1.0  # nothing received for 1 s
    assert sa._build_scene([]).lanes is None


def test_no_lanes_means_none():
    sa = _assembler()
    assert sa._build_scene([]).lanes is None


# --- Ego bundle fold-in ---

def test_scene_folds_fresh_ego():
    asm = SceneAssembler(Config(), EventBus(), image_width=1280)
    asm._latest_ego = EgoEstimate(
        cipo_distance_m=42.0, road_curvature_1pm=0.01, cipo_flag=True, ego_path=((1.0, 2.0),))
    asm._ego_seen_at = time.monotonic()
    scene = asm._build_scene([])
    assert scene.cipo_distance_m == 42.0
    assert scene.road_curvature_1pm == 0.01
    assert scene.cipo_flag is True
    assert scene.ego_path == ((1.0, 2.0),)


def test_scene_drops_stale_ego():
    asm = SceneAssembler(Config(), EventBus(), image_width=1280)
    asm._latest_ego = EgoEstimate(cipo_distance_m=42.0)
    asm._ego_seen_at = time.monotonic() - 5.0   # older than the freshness window
    scene = asm._build_scene([])
    assert scene.cipo_distance_m is None
    assert scene.cipo_flag is None
