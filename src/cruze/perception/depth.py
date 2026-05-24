"""
Monocular distance estimation using the pinhole camera model.

Method: given a detected bounding box of known real-world height H, the
distance Z is estimated from the similar-triangles relationship:

    Z = (H_real * f) / H_bbox_px

where f is the focal length in pixels, derived once at startup from the
configured horizontal field-of-view and image width:

    f = (image_width / 2) / tan(hfov / 2)

Accuracy: ±20-30% under ideal conditions. Degrades with:
  - Non-frontal vehicle orientation (broadside car looks "taller")
  - Partial occlusion (bbox height is clipped)
  - Lens distortion (correct with calibrate_camera.py)

For production-grade accuracy, use stereo vision or LiDAR; treat this as a
useful approximation for warning thresholds, not precise measurement.
"""

from __future__ import annotations

import math

from cruze.core.types import BBox, ObjectClass

# Known real-world heights per class (metres).
# Sources: vehicle design specs and pedestrian anthropometrics.
_REAL_HEIGHT_M: dict[ObjectClass, float] = {
    ObjectClass.CAR: 1.5,
    ObjectClass.TRUCK: 3.0,
    ObjectClass.BUS: 3.2,
    ObjectClass.MOTORCYCLE: 1.2,
    ObjectClass.BICYCLE: 1.1,
    ObjectClass.PERSON: 1.7,
    ObjectClass.STOP_SIGN: 0.75,
    ObjectClass.TRAFFIC_LIGHT: 0.4,
    ObjectClass.UNKNOWN: 1.5,   # fallback: assume car-sized
}


def focal_length_px(image_width: int, hfov_deg: float) -> float:
    """Return focal length in pixels from image width and horizontal FOV."""
    return (image_width / 2.0) / math.tan(math.radians(hfov_deg / 2.0))


def estimate_distance(
    bbox: BBox,
    cls: ObjectClass,
    focal_length: float,
) -> float | None:
    """
    Return distance estimate in metres, or None if bbox is degenerate.

    Parameters
    ----------
    bbox:
        Detected bounding box in pixels.
    cls:
        Object class (used to look up real-world height).
    focal_length:
        Focal length in pixels (from `focal_length_px()`).
    """
    h_px = bbox.height
    if h_px < 1.0:
        return None
    h_real = _REAL_HEIGHT_M.get(cls, 1.5)
    distance = (h_real * focal_length) / h_px
    # Clamp to [0.5 m, 200 m] — outside this range the estimate is noise.
    return max(0.5, min(distance, 200.0))
