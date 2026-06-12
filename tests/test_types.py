"""Data-contract tests for additive fields (backwards compatibility)."""

import dataclasses

from cruze.core.types import BBox, ObjectClass, Track, VehicleState


def test_track_speed_mps_defaults_none():
    t = Track(track_id=1, bbox=BBox(0, 0, 10, 10), cls=ObjectClass.CAR)
    assert t.speed_mps is None


def test_track_speed_mps_settable_via_replace():
    t = Track(track_id=1, bbox=BBox(0, 0, 10, 10), cls=ObjectClass.CAR)
    t2 = dataclasses.replace(t, speed_mps=25.0)
    assert t2.speed_mps == 25.0
    assert t2.track_id == 1
    assert t.speed_mps is None  # original frozen instance unchanged


def test_vehicle_state_gps_speed_defaults_none():
    v = VehicleState()
    assert v.gps_speed_mps is None


def test_vehicle_state_gps_speed_settable_via_replace():
    v = VehicleState()
    v2 = dataclasses.replace(v, gps_speed_mps=12.5)
    assert v2.gps_speed_mps == 12.5
    assert v.gps_speed_mps is None
