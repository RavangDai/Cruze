"""
Lane detection — classical CV approach (Phase 1).

Pipeline:
  1. Convert to grayscale, apply Gaussian blur.
  2. Canny edge detection.
  3. Region-of-interest mask (trapezoidal, bottom half of frame).
  4. Probabilistic Hough transform to find line segments.
  5. Separate left / right lanes by slope, average into two lane lines.

Returns left_line and right_line as (x1, y1, x2, y2) tuples in pixel
coordinates, or None if that lane isn't detected.

Phase 2: replace with a learned segmentation model (e.g. UFLD) for
curved roads and adverse lighting.
"""

from __future__ import annotations

import logging
from typing import Any, NamedTuple

from cruze.core.types import LaneLine

logger = logging.getLogger(__name__)


class LaneResult(NamedTuple):
    left: LaneLine | None
    right: LaneLine | None
    # Curved-boundary polylines (pixel coords, bottom point first); None when
    # the quadratic fit is under-determined or unstable.
    left_poly: tuple[tuple[float, float], ...] | None = None
    right_poly: tuple[tuple[float, float], ...] | None = None


# Points sampled along each quadratic boundary: 8 keeps the max sagitta error
# under ~2 px across a ~300 px ROI for road-curvature quadratics while costing
# ~130 bytes per side on the wire.
_POLY_SAMPLES = 8

# Minimum pixels between the bottom row and the horizon before the ground
# plane projection diverges (same floor as depth estimation).
_MIN_HORIZON_OFFSET_PX = 8.0


def cte_from_lane_positions(
    left_x_px: float | None,
    right_x_px: float | None,
    image_w: int,
    image_h: int,
    focal_px: float,
    camera_height_m: float,
    horizon_y: float,
) -> float | None:
    """
    Cross-track error: ego's lateral offset from the lane centre in metres,
    measured at the bottom image row. Positive = ego right of centre (the
    lane centre projects left of the image centre when ego sits right of it).

    Ground-plane model shared with perception.depth: at the bottom row,
    forward Z = focal·h_cam/(image_h − horizon_y), and one pixel spans
    Z/focal metres laterally.
    """
    if left_x_px is None or right_x_px is None:
        return None
    if image_h - horizon_y < _MIN_HORIZON_OFFSET_PX:
        return None
    z_m = focal_px * camera_height_m / (image_h - horizon_y)
    lane_centre_x = (left_x_px + right_x_px) / 2.0
    return (image_w / 2.0 - lane_centre_x) * z_m / focal_px


def detect_lanes(image: Any) -> LaneResult:
    """
    Detect left and right lane lines in an BGR image (numpy uint8 HxWxC).

    Returns LaneResult with left/right LaneLines in pixel coordinates,
    or None for a lane that couldn't be detected.
    """
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError:
        return LaneResult(None, None)

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Trapezoidal ROI: bottom 55% of frame.
    roi_mask = np.zeros_like(edges)
    roi_top_y = int(h * 0.45)
    polygon = np.array([[
        (0, h),
        (int(w * 0.45), roi_top_y),
        (int(w * 0.55), roi_top_y),
        (w, h),
    ]], dtype=np.int32)
    cv2.fillPoly(roi_mask, polygon, 255)
    masked = cv2.bitwise_and(edges, roi_mask)

    lines = cv2.HoughLinesP(
        masked,
        rho=1,
        theta=3.14159 / 180,
        threshold=40,
        minLineLength=40,
        maxLineGap=100,
    )
    if lines is None:
        return LaneResult(None, None)

    left_pts: list[tuple[int, int]] = []
    right_pts: list[tuple[int, int]] = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        # Image coords put the origin top-left with y growing downward, so the
        # LEFT boundary (bottom-left rising toward the vanishing point) has
        # NEGATIVE slope and the right boundary positive.
        if -2.5 < slope < -0.4:
            left_pts.extend([(x1, y1), (x2, y2)])
        elif 0.4 < slope < 2.5:
            right_pts.extend([(x1, y1), (x2, y2)])

    def _fit_lane(pts: list[tuple[int, int]]) -> LaneLine | None:
        if len(pts) < 2:
            return None
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        try:
            m, b = np.polyfit(xs, ys, 1)
        except (np.linalg.LinAlgError, ValueError):
            return None
        y_bottom = h
        y_top = roi_top_y
        x_bottom = (y_bottom - b) / m if m != 0 else 0.0
        x_top = (y_top - b) / m if m != 0 else 0.0
        return LaneLine(float(x_bottom), float(y_bottom), float(x_top), float(y_top))

    def _fit_poly(
        pts: list[tuple[int, int]], line: LaneLine | None
    ) -> tuple[tuple[float, float], ...] | None:
        # A quadratic needs ≥3 Hough segments (6 endpoints) to be determined
        # by more than noise; fewer points → straight-line fallback only.
        if line is None or len(pts) < 6:
            return None
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        try:
            # x as a function of y — lanes are near-vertical in image space.
            coeffs = np.polyfit(ys, xs, 2)
        except (np.linalg.LinAlgError, ValueError):
            return None
        poly = np.poly1d(coeffs)
        # Sanity gate: the quadratic's bottom point must agree with the
        # robust straight fit; wilder disagreement means it latched onto
        # outliers (shadows, other markings).
        if abs(float(poly(h)) - line.x1) > 0.25 * w:
            return None
        sample_ys = np.linspace(h, roi_top_y, _POLY_SAMPLES)
        return tuple((float(poly(y)), float(y)) for y in sample_ys)

    left_line = _fit_lane(left_pts)
    right_line = _fit_lane(right_pts)
    return LaneResult(
        left_line,
        right_line,
        _fit_poly(left_pts, left_line),
        _fit_poly(right_pts, right_line),
    )
