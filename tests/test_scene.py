"""SceneAssembler absolute-speed tests — no bus traffic needed."""

import pytest

from cruze.core.bus import EventBus
from cruze.core.config import Config
from cruze.core.types import BBox, ObjectClass, Track, VehicleState
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
