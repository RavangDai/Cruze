"""Lane detector tests — skipped when opencv is not installed (repo rule:
the base suite passes with stdlib + numpy; cv2-dependent tests importorskip)."""

import pytest

cv2 = pytest.importorskip("cv2")
import numpy as np

from cruze.core.types import LaneLine
from cruze.perception import lane
from cruze.perception.lane import detect_lanes


def _road_image(w=960, h=540):
    """Black frame with two bright lane boundaries converging toward centre."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Left boundary: bottom-left rising toward the vanishing point.
    cv2.line(img, (150, h), (440, int(h * 0.45)), (255, 255, 255), 6)
    # Right boundary: bottom-right rising toward the vanishing point.
    cv2.line(img, (830, h), (540, int(h * 0.45)), (255, 255, 255), 6)
    return img


def test_left_lane_is_on_the_left():
    """Regression: slope-sign convention was inverted — in image coords the
    LEFT boundary has negative slope (y grows downward), so 'left' came back
    on the right side of the frame and the HUD corridor drew as a bowtie."""
    result = detect_lanes(_road_image())
    assert result.left is not None
    assert result.right is not None
    centre = 960 / 2
    assert result.left.x1 < centre, f"left lane bottom at x={result.left.x1}"
    assert result.right.x1 > centre, f"right lane bottom at x={result.right.x1}"


def test_lane_lines_span_roi_vertically():
    result = detect_lanes(_road_image())
    assert result.left is not None
    # Bottom point first (y1 = frame bottom), top at the ROI ceiling.
    assert result.left.y1 == 540
    assert result.left.y2 == pytest.approx(540 * 0.45, abs=1)


def test_blank_image_detects_nothing():
    img = np.zeros((540, 960, 3), dtype=np.uint8)
    result = detect_lanes(img)
    assert result.left is None
    assert result.right is None


# --- Quadratic polyline fitting ---

def _curved_road_image(w=960, h=540):
    """Boundaries drawn as chained segments approximating a rightward arc."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    left_pts = [(150, 540), (260, 460), (360, 380), (430, 300), (470, 243)]
    right_pts = [(830, 540), (760, 460), (700, 380), (660, 300), (640, 243)]
    for pts in (left_pts, right_pts):
        for p1, p2 in zip(pts, pts[1:]):
            cv2.line(img, p1, p2, (255, 255, 255), 6)
    return img


def test_curved_road_produces_polylines():
    result = detect_lanes(_curved_road_image())
    assert result.left_poly is not None
    assert result.right_poly is not None
    assert len(result.left_poly) >= 2
    # Bottom-first convention: first point sits on the bottom image row.
    assert result.left_poly[0][1] == pytest.approx(540, abs=1)
    # Polyline bottom x agrees with the straight-line fallback bottom x.
    assert result.left_poly[0][0] == pytest.approx(result.left.x1, abs=30)


def test_straight_road_polyline_matches_line():
    result = detect_lanes(_road_image())
    assert result.left_poly is not None
    # On a straight road the quadratic degenerates to the line fit.
    assert result.left_poly[0][0] == pytest.approx(result.left.x1, abs=10)
    assert result.left_poly[-1][0] == pytest.approx(result.left.x2, abs=10)


def test_sparse_side_falls_back_to_line_only():
    """One short segment per side under-determines a quadratic — the straight
    LaneLine must still be produced while the polyline degrades to None."""
    img = np.zeros((540, 960, 3), dtype=np.uint8)
    # Full curve on the right, single short (just over Hough min-length) left.
    right_pts = [(830, 540), (760, 460), (700, 380), (660, 300), (640, 243)]
    for p1, p2 in zip(right_pts, right_pts[1:]):
        cv2.line(img, p1, p2, (255, 255, 255), 6)
    cv2.line(img, (200, 500), (230, 455), (255, 255, 255), 6)
    result = detect_lanes(img)
    assert result.right_poly is not None
    # Sparse left: whatever Hough finds, the poly path must not fabricate a
    # curve from a single segment (its LaneLine fallback may still exist).
    assert result.left_poly is None


# --- Vanishing point / horizon ---

def test_vanishing_point_from_converging_boundaries():
    # Two boundaries converging on (960, 600) in a 1280-tall frame.
    left = LaneLine(x1=460.0, y1=1280.0, x2=960.0, y2=600.0)
    right = LaneLine(x1=1460.0, y1=1280.0, x2=960.0, y2=600.0)
    assert lane.vanishing_point_y(left, right, 1280) == pytest.approx(600.0, abs=0.5)


def test_vanishing_point_needs_both_boundaries():
    line = LaneLine(x1=460.0, y1=1280.0, x2=960.0, y2=600.0)
    assert lane.vanishing_point_y(line, None, 1280) is None
    assert lane.vanishing_point_y(None, line, 1280) is None
    assert lane.vanishing_point_y(None, None, 1280) is None


def test_vanishing_point_rejects_near_parallel_boundaries():
    # Parallel lines meet at infinity, where a pixel of fit noise moves the
    # answer arbitrarily far.
    left = LaneLine(x1=400.0, y1=1280.0, x2=400.0, y2=600.0)
    right = LaneLine(x1=1500.0, y1=1280.0, x2=1500.0, y2=600.0)
    assert lane.vanishing_point_y(left, right, 1280) is None


def test_vanishing_point_rejects_implausible_row():
    # Converging just above the bonnet: the fit latched onto something that is
    # not a pair of lanes, and a horizon there would wreck every distance.
    left = LaneLine(x1=100.0, y1=1280.0, x2=940.0, y2=1270.0)
    right = LaneLine(x1=1800.0, y1=1280.0, x2=980.0, y2=1270.0)
    assert lane.vanishing_point_y(left, right, 1280) is None
