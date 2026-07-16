"""
Traffic-light state classification — classical colour analysis (no ML).

Approach: hue histogram of the bright, saturated pixels in the detection
crop. Position-based methods (top/middle/bottom thirds) assume a tight,
vertical bbox; YOLO boxes are frequently loose, clipped, or around
horizontally-mounted lights, so hue of the lit lamp is the primary signal —
it is orientation- and framing-agnostic.

Pure numpy (no cv2) so it stays unit-testable with the base test deps and
adds < 1 ms per light for typical crop sizes.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np

from cruze.core.types import Detection, ObjectClass, TrafficLightState

logger = logging.getLogger(__name__)

# Lit-lamp pixel gates: lamp emitters are bright AND saturated. The dark
# housing fails the V gate; sky/backlight is bright but colourless and fails
# the S gate. Values chosen empirically on dashcam lamp crops.
_MIN_VALUE = 0.45
_MIN_SATURATION = 0.35

# Minimum fraction of the crop that must be lit-lamp pixels; below this the
# light is unlit, occluded, or motion-blurred → UNKNOWN.
_MIN_LIT_FRACTION = 0.02

# Winner bin must beat the runner-up by this factor, else the reading is
# ambiguous (e.g. red/green reflections) → UNKNOWN. Tracker debounce smooths
# the resulting flicker.
_MIN_DOMINANCE = 1.5

# Crops smaller than this per side carry too few pixels for a stable histogram.
_MIN_CROP_PX = 4

# Hue bands in degrees (0 = red, 60 = yellow, 120 = green). The green band is
# wide because LED traffic greens skew cyan (~160-180°).
_RED_MAX, _RED_WRAP = 25.0, 330.0
_YELLOW_MAX = 70.0
_GREEN_MAX = 200.0


def _hsv_from_bgr(crop: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized BGR uint8 → (hue°, saturation, value) float arrays."""
    bgr = crop.astype(np.float32) / 255.0
    b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc

    value = maxc
    saturation = np.where(maxc > 0, delta / np.maximum(maxc, 1e-6), 0.0)

    safe_delta = np.where(delta > 1e-6, delta, 1.0)
    r_is_max = (maxc == r) & (delta > 1e-6)
    g_is_max = (maxc == g) & (delta > 1e-6) & ~r_is_max
    b_is_max = (delta > 1e-6) & ~r_is_max & ~g_is_max

    hue = np.zeros_like(maxc)
    hue = np.where(r_is_max, (60.0 * (g - b) / safe_delta) % 360.0, hue)
    hue = np.where(g_is_max, 60.0 * (b - r) / safe_delta + 120.0, hue)
    hue = np.where(b_is_max, 60.0 * (r - g) / safe_delta + 240.0, hue)
    return hue, saturation, value


def classify_light_bgr(crop: np.ndarray) -> TrafficLightState:
    """
    Classify the lamp state of a traffic-light bbox crop (BGR uint8 HxWxC).

    Returns UNKNOWN rather than guessing when the crop is too small, unlit,
    or the colour reading is ambiguous.
    """
    if crop.size == 0 or crop.shape[0] < _MIN_CROP_PX or crop.shape[1] < _MIN_CROP_PX:
        return TrafficLightState.UNKNOWN

    hue, saturation, value = _hsv_from_bgr(crop)
    lit = (value > _MIN_VALUE) & (saturation > _MIN_SATURATION)

    if lit.sum() < _MIN_LIT_FRACTION * lit.size:
        return TrafficLightState.UNKNOWN

    lit_hue = hue[lit]
    red = int((((lit_hue < _RED_MAX) | (lit_hue >= _RED_WRAP))).sum())
    yellow = int(((lit_hue >= _RED_MAX) & (lit_hue < _YELLOW_MAX)).sum())
    green = int(((lit_hue >= _YELLOW_MAX) & (lit_hue < _GREEN_MAX)).sum())

    bins = sorted(
        [
            (red, TrafficLightState.RED),
            (yellow, TrafficLightState.YELLOW),
            (green, TrafficLightState.GREEN),
        ],
        key=lambda item: item[0],
        reverse=True,
    )
    (top_count, top_state), (second_count, _), _ = bins
    if top_count == 0 or top_count < _MIN_DOMINANCE * second_count:
        return TrafficLightState.UNKNOWN
    return top_state


def annotate_lights(detections: list[Detection], image: np.ndarray) -> list[Detection]:
    """
    Return *detections* with light_state filled in for TRAFFIC_LIGHT entries.

    Other classes pass through untouched (light_state stays None). Bboxes are
    clamped to the image; degenerate crops classify as UNKNOWN.
    """
    if not detections:
        return detections

    img_h, img_w = image.shape[:2]
    out: list[Detection] = []
    for det in detections:
        if det.cls is not ObjectClass.TRAFFIC_LIGHT:
            out.append(det)
            continue
        x1 = max(0, int(det.bbox.x1))
        y1 = max(0, int(det.bbox.y1))
        x2 = min(img_w, int(det.bbox.x2))
        y2 = min(img_h, int(det.bbox.y2))
        if x2 <= x1 or y2 <= y1:
            state = TrafficLightState.UNKNOWN
        else:
            state = classify_light_bgr(image[y1:y2, x1:x2])
        out.append(dataclasses.replace(det, light_state=state))
    return out
