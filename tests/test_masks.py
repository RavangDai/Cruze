"""Mask polygon downsampling tests — yolo.py imports clean without ultralytics
(the heavy import lives inside YoloDetector.__init__)."""

import numpy as np

from cruze.perception.backends.yolo import _MASK_MAX_POINTS, _downsample_polygon


def _circle(n):
    """n-point polygon around a circle of radius 100 at (200, 200)."""
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([200 + 100 * np.cos(theta), 200 + 100 * np.sin(theta)], axis=1)


def test_long_polygon_capped():
    out = _downsample_polygon(_circle(300))
    assert out is not None
    assert 3 <= len(out) <= _MASK_MAX_POINTS


def test_short_polygon_passed_through():
    out = _downsample_polygon(_circle(10))
    assert out is not None
    assert len(out) == 10


def test_degenerate_polygon_is_none():
    assert _downsample_polygon(_circle(2)) is None
    assert _downsample_polygon(np.zeros((0, 2))) is None
    assert _downsample_polygon(None) is None


def test_points_rounded_to_one_decimal():
    poly = np.array([[10.1234, 20.5678], [30.9999, 40.0001], [50.55, 60.44]])
    out = _downsample_polygon(poly)
    assert out == ((10.1, 20.6), (31.0, 40.0), (50.5, 60.4))


def test_output_is_immutable_tuple_of_tuples():
    out = _downsample_polygon(_circle(5))
    assert isinstance(out, tuple)
    assert all(isinstance(p, tuple) and len(p) == 2 for p in out)
