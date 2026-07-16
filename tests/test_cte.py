"""Cross-track-error geometry + CTE Kalman tests — pure math, no cv2.

Ground-plane model shared with perception/depth.py: for a road pixel at the
bottom image row, forward Z = focal·h_cam/(image_h − horizon_y) and lateral
metres-per-pixel = Z/focal.
"""

import pytest

from cruze.perception.lane import cte_from_lane_positions
from cruze.perception.pipeline import _ScalarKalman

# Reference geometry: 960x540 image, focal 600 px, camera 1.2 m up, level.
_W, _H, _FOCAL, _CAM_H, _HORIZON = 960, 540, 600.0, 1.2, 270.0
# Z at bottom row = 600*1.2/(540-270) = 2.6667 m
_Z = _FOCAL * _CAM_H / (_H - _HORIZON)


def _cte(left_x, right_x):
    return cte_from_lane_positions(left_x, right_x, _W, _H, _FOCAL, _CAM_H, _HORIZON)


def test_symmetric_lanes_zero_cte():
    assert _cte(380.0, 580.0) == pytest.approx(0.0)


def test_lane_centre_left_of_image_centre_positive_cte():
    # Lane centre 100 px left of image centre → ego sits right of lane centre.
    # cte = 100 px * Z/focal = 100 * 2.6667/600 ≈ +0.444 m
    assert _cte(280.0, 480.0) == pytest.approx(100.0 * _Z / _FOCAL)
    assert _cte(280.0, 480.0) == pytest.approx(0.444, abs=0.001)


def test_lane_centre_right_of_image_centre_negative_cte():
    assert _cte(480.0, 680.0) == pytest.approx(-0.444, abs=0.001)


def test_none_when_side_missing():
    assert _cte(None, 580.0) is None
    assert _cte(380.0, None) is None


def test_none_when_horizon_geometry_unusable():
    # Horizon within 8 px of the bottom row → Z blows up; same floor as depth.
    assert cte_from_lane_positions(380.0, 580.0, _W, _H, _FOCAL, _CAM_H, _H - 5.0) is None


# --- _ScalarKalman ---

_DT = 1.0 / 30.0  # 30 fps frame interval


def test_kalman_converges_to_constant():
    k = _ScalarKalman()
    out = 0.0
    for _ in range(5):
        out = k.update(0.5, _DT)
    assert out == pytest.approx(0.5, abs=0.01)


def test_kalman_spike_triggers_reinit_not_average():
    k = _ScalarKalman()
    for _ in range(5):
        k.update(0.0, _DT)
    # 2 m jump = adjacent-lane re-latch (US lane ≈ 3.7 m) → snap, don't blend.
    out = k.update(2.0, _DT)
    assert out == pytest.approx(2.0, abs=0.01)


def test_kalman_tracks_slow_ramp():
    # 0.6 m/s lateral drift at 30 fps = 0.02 m/frame — a real departure ramp.
    k = _ScalarKalman()
    out = 0.0
    for i in range(30):
        out = k.update(0.02 * i, _DT)
    assert out == pytest.approx(0.02 * 29, abs=0.1)


def test_kalman_reinits_after_measurement_gap():
    k = _ScalarKalman()
    for _ in range(5):
        k.update(0.0, _DT)
    # 0.6 s without a measurement → state is stale, snap to the new reading.
    out = k.update(1.5, 0.6)
    assert out == pytest.approx(1.5, abs=0.01)
