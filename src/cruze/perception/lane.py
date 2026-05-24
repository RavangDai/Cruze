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
from typing import NamedTuple

logger = logging.getLogger(__name__)


class LaneLine(NamedTuple):
    x1: int
    y1: int
    x2: int
    y2: int


class LaneResult(NamedTuple):
    left: LaneLine | None
    right: LaneLine | None


def detect_lanes(image: "Any", ) -> LaneResult:
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
        # Positive slope (image coords) = left lane; negative = right.
        if 0.4 < slope < 2.5:
            left_pts.extend([(x1, y1), (x2, y2)])
        elif -2.5 < slope < -0.4:
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
        x_bottom = int((y_bottom - b) / m) if m != 0 else 0
        x_top = int((y_top - b) / m) if m != 0 else 0
        return LaneLine(x_bottom, y_bottom, x_top, y_top)

    return LaneResult(_fit_lane(left_pts), _fit_lane(right_pts))


# Allow importing the type annotation without cv2 installed.
from typing import Any  # noqa: E402
