"""
Monocular distance estimation using the pinhole camera model.

Two methods, fused by estimate_distance_fused() (the recommended entry point):

1. Ground-plane (primary, road-contact classes): the bbox bottom edge touches
   the road, so flat-road similar triangles give

       Z = (f * H_cam) / (y_bottom - y_horizon)

   Independent of object size → ±5-10 % on flat roads. Invalid for elevated
   objects (signs, lights) or boxes at/above the horizon.

2. Bbox-height (fallback): given a known real-world height H per class,

       Z = (H_real * f) / H_bbox_px

   Accuracy ±20-30 %; degrades with non-frontal orientation, occlusion
   (clipped bbox), and lens distortion (correct with calibrate_camera.py).

f is the focal length in pixels, derived once at startup from the configured
horizontal field-of-view and image width: f = (image_width / 2) / tan(hfov / 2).

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


# Classes whose bbox bottom edge touches the road surface — ground-plane
# geometry applies. Signs and lights are elevated; bbox-height only.
_GROUND_CONTACT_CLASSES = {
    ObjectClass.CAR,
    ObjectClass.TRUCK,
    ObjectClass.BUS,
    ObjectClass.MOTORCYCLE,
    ObjectClass.BICYCLE,
    ObjectClass.PERSON,
}

# Bbox bottoms closer to the horizon than this are unusable: at <8 px a
# single-pixel error changes the estimate by >12 %, swamping the geometry.
_MIN_HORIZON_OFFSET_PX = 8.0


def horizon_y_px(image_height: int, focal_length: float, pitch_deg: float) -> float:
    """
    Image row of the horizon. pitch_deg > 0 = camera tilted down, which moves
    the horizon up (smaller y) by f·tan(pitch) from the principal point.
    """
    return image_height / 2.0 - focal_length * math.tan(math.radians(pitch_deg))


def estimate_distance_ground_plane(
    bbox: BBox,
    focal_length: float,
    camera_height_m: float,
    horizon_y: float,
) -> float | None:
    """
    Flat-road distance from the bbox bottom edge:

        Z = f · H_cam / (y_bottom − y_horizon)

    Pure similar-triangles on the ground plane: independent of object size,
    so it avoids the ±20-30 % class-height error of the bbox-height method.
    Returns None when the bbox bottom is at/above the horizon (crest, clipped
    box, elevated object) — caller should fall back to bbox-height.
    """
    dy = bbox.y2 - horizon_y
    if dy < _MIN_HORIZON_OFFSET_PX:
        return None
    distance = focal_length * camera_height_m / dy
    # Same plausibility clamp as estimate_distance().
    return max(0.5, min(distance, 200.0))


def estimate_distance_fused(
    bbox: BBox,
    cls: ObjectClass,
    focal_length: float,
    camera_height_m: float,
    horizon_y: float,
) -> float | None:
    """Ground-plane estimate for road-contact classes; bbox-height otherwise."""
    if cls in _GROUND_CONTACT_CLASSES:
        gp = estimate_distance_ground_plane(bbox, focal_length, camera_height_m, horizon_y)
        if gp is not None:
            return gp
    return estimate_distance(bbox, cls, focal_length)


def ground_position_xz(
    bbox: BBox,
    distance_m: float | None,
    focal_length: float,
    image_width: int,
) -> tuple[float, float] | None:
    """
    Object position on the ground plane in metres: (lateral X, forward Z),
    X positive to the right of the camera axis.

    This is the (u,v) → (X,Y) step the VisionPilot pipeline applies to a CIPO
    box's bottom centre, expressed with the intrinsics we already have rather
    than a full homography: at range Z one pixel of horizontal offset from the
    principal point spans Z/f metres, so

        X = (u_centre − image_width/2) · Z / f

    Z comes from the caller's existing depth estimate, so this adds no new
    error model — it only resolves the bearing the distance estimate lacks.
    Returns None when distance is unknown or the focal length is degenerate.
    """
    if distance_m is None or focal_length <= 0.0:
        return None
    lateral_m = (bbox.cx - image_width / 2.0) * distance_m / focal_length
    return (lateral_m, distance_m)
