"""Monocular depth tests — bbox-height and ground-plane methods."""

import pytest

from cruze.core.types import BBox, ObjectClass
from cruze.perception.depth import (
    estimate_distance,
    estimate_distance_fused,
    estimate_distance_ground_plane,
    horizon_y_px,
)


def test_horizon_at_image_center_with_zero_pitch():
    assert horizon_y_px(720, 1000.0, 0.0) == pytest.approx(360.0)


def test_horizon_rises_with_downward_pitch():
    # Camera pitched 5° down → horizon moves up (smaller y).
    assert horizon_y_px(720, 1000.0, 5.0) < 360.0


def test_ground_plane_known_geometry():
    # f=1000 px, camera 1.5 m above road, horizon at y=360.
    # bbox bottom at y=460 → dy=100 px → Z = 1000·1.5/100 = 15 m.
    bbox = BBox(100, 380, 200, 460)
    z = estimate_distance_ground_plane(bbox, 1000.0, 1.5, 360.0)
    assert z == pytest.approx(15.0)


def test_ground_plane_rejects_bbox_above_horizon():
    bbox = BBox(100, 100, 200, 200)  # bottom edge above horizon
    assert estimate_distance_ground_plane(bbox, 1000.0, 1.5, 360.0) is None


def test_ground_plane_clamps_far_distance():
    # f=2000, H=3.0, 9 px below horizon → raw 2000·3/9 ≈ 667 m → clamp to 200.
    bbox = BBox(100, 300, 200, 369.0)
    z = estimate_distance_ground_plane(bbox, 2000.0, 3.0, 360.0)
    assert z == pytest.approx(200.0)


def test_fused_uses_ground_plane_for_vehicles():
    bbox = BBox(100, 380, 200, 460)
    z = estimate_distance_fused(bbox, ObjectClass.CAR, 1000.0, 1.5, 360.0)
    assert z == pytest.approx(15.0)


def test_fused_falls_back_to_bbox_height_above_horizon():
    bbox = BBox(100, 100, 200, 200)
    z = estimate_distance_fused(bbox, ObjectClass.CAR, 1000.0, 1.5, 360.0)
    assert z == pytest.approx(estimate_distance(bbox, ObjectClass.CAR, 1000.0))


def test_fused_uses_bbox_height_for_traffic_lights():
    # Traffic lights don't touch the road — ground-plane geometry is invalid.
    bbox = BBox(100, 380, 200, 460)
    z = estimate_distance_fused(bbox, ObjectClass.TRAFFIC_LIGHT, 1000.0, 1.5, 360.0)
    assert z == pytest.approx(estimate_distance(bbox, ObjectClass.TRAFFIC_LIGHT, 1000.0))
