"""Speed-fusion pure-function tests."""

import pytest

from cruze.telemetry.fusion import (
    STALENESS_S,
    SpeedSample,
    fuse_speed,
    haversine_m,
    position_speed_mps,
)


def test_haversine_known_distance():
    # 0.001° of latitude ≈ 111.2 m anywhere on Earth.
    d = haversine_m(37.7749, -122.4194, 37.7759, -122.4194)
    assert d == pytest.approx(111.2, rel=0.01)


def test_haversine_zero():
    assert haversine_m(37.0, -122.0, 37.0, -122.0) == 0.0


def test_position_speed_basic():
    # 111.2 m in 10 s ≈ 11.12 m/s.
    v = position_speed_mps(37.7749, -122.4194, 0.0, 37.7759, -122.4194, 10.0)
    assert v == pytest.approx(11.12, rel=0.01)


def test_position_speed_rejects_teleport():
    # 1 full degree (~111 km) in 1 s is a GPS glitch, not motion.
    assert position_speed_mps(37.0, -122.0, 0.0, 38.0, -122.0, 1.0) is None


def test_position_speed_rejects_nonpositive_dt():
    assert position_speed_mps(37.0, -122.0, 5.0, 37.001, -122.0, 5.0) is None


def test_fuse_priority_obd_wins():
    now = 100.0
    samples = [
        (SpeedSample(20.0, 99.5), "obd"),
        (SpeedSample(21.0, 99.9), "gps"),
        (SpeedSample(22.0, 99.9), "gps_pos"),
    ]
    assert fuse_speed(samples, now) == (20.0, "obd")


def test_fuse_falls_through_stale_sources():
    now = 100.0
    samples = [
        (SpeedSample(20.0, now - STALENESS_S - 1.0), "obd"),   # stale
        (SpeedSample(21.0, 99.9), "gps"),
    ]
    assert fuse_speed(samples, now) == (21.0, "gps")


def test_fuse_all_missing_or_stale():
    now = 100.0
    samples = [
        (None, "obd"),
        (SpeedSample(21.0, 0.0), "gps"),
    ]
    assert fuse_speed(samples, now) == (None, "unknown")


def test_fuse_staleness_boundary_inclusive():
    # A sample exactly STALENESS_S old is still accepted ("at most this old").
    now = 100.0
    samples = [(SpeedSample(20.0, now - STALENESS_S), "obd")]
    assert fuse_speed(samples, now) == (20.0, "obd")


def test_position_speed_zero_distance_is_zero():
    # Stationary vehicle: same fix twice with positive dt → 0.0, not None.
    assert position_speed_mps(37.0, -122.0, 0.0, 37.0, -122.0, 1.0) == 0.0
