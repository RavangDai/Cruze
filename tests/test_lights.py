"""Traffic-light state classifier tests — pure numpy, no cv2."""

import numpy as np

from cruze.core.types import BBox, Detection, ObjectClass, TrafficLightState
from cruze.perception.lights import annotate_lights, classify_light_bgr


def _crop(h=40, w=20):
    """Dark housing-coloured crop (BGR uint8)."""
    return np.full((h, w, 3), 20, dtype=np.uint8)


def _with_patch(crop, bgr, y0=4, x0=6, size=8):
    crop = crop.copy()
    crop[y0:y0 + size, x0:x0 + size] = bgr
    return crop


# --- classify_light_bgr ---

def test_red_lamp_classified_red():
    crop = _with_patch(_crop(), (0, 0, 255))
    assert classify_light_bgr(crop) is TrafficLightState.RED


def test_green_lamp_classified_green():
    crop = _with_patch(_crop(), (0, 255, 0))
    assert classify_light_bgr(crop) is TrafficLightState.GREEN


def test_cyan_led_green_classified_green():
    # Real traffic-light green LEDs skew cyan (hue ~170°).
    crop = _with_patch(_crop(), (200, 255, 0))
    assert classify_light_bgr(crop) is TrafficLightState.GREEN


def test_amber_lamp_classified_yellow():
    crop = _with_patch(_crop(), (0, 200, 255))
    assert classify_light_bgr(crop) is TrafficLightState.YELLOW


def test_unlit_light_is_unknown():
    assert classify_light_bgr(_crop()) is TrafficLightState.UNKNOWN


def test_tiny_crop_is_unknown():
    assert classify_light_bgr(np.zeros((2, 2, 3), dtype=np.uint8)) is TrafficLightState.UNKNOWN


def test_bright_unsaturated_crop_is_unknown():
    # Sky/backlight: bright but colourless must not read as a lamp.
    crop = np.full((40, 20, 3), 240, dtype=np.uint8)
    assert classify_light_bgr(crop) is TrafficLightState.UNKNOWN


def test_dominant_colour_wins():
    crop = _with_patch(_crop(), (0, 0, 255), y0=2, x0=6, size=4)      # small red
    crop = _with_patch(crop, (0, 255, 0), y0=20, x0=4, size=12)      # larger green
    assert classify_light_bgr(crop) is TrafficLightState.GREEN


def test_ambiguous_mix_is_unknown():
    # Near-equal red and green areas → no confident winner.
    crop = _with_patch(_crop(), (0, 0, 255), y0=4, x0=6, size=8)
    crop = _with_patch(crop, (0, 255, 0), y0=24, x0=6, size=8)
    assert classify_light_bgr(crop) is TrafficLightState.UNKNOWN


# --- annotate_lights ---

def _image_with_light(bgr, x1=100, y1=50, x2=120, y2=90):
    img = np.full((200, 300, 3), 20, dtype=np.uint8)
    img[y1:y2, x1:x2] = bgr
    return img


def test_annotate_sets_state_on_traffic_light():
    img = _image_with_light((0, 0, 255))
    det = Detection(BBox(100, 50, 120, 90), 0.9, ObjectClass.TRAFFIC_LIGHT)
    out = annotate_lights([det], img)
    assert out[0].light_state is TrafficLightState.RED


def test_annotate_leaves_other_classes_alone():
    img = _image_with_light((0, 0, 255))
    det = Detection(BBox(100, 50, 120, 90), 0.9, ObjectClass.CAR)
    out = annotate_lights([det], img)
    assert out[0].light_state is None


def test_annotate_clamps_out_of_frame_bbox():
    img = _image_with_light((0, 255, 0), x1=280, y1=0, x2=300, y2=40)
    det = Detection(BBox(280, -10, 350, 40), 0.9, ObjectClass.TRAFFIC_LIGHT)
    out = annotate_lights([det], img)
    assert out[0].light_state is TrafficLightState.GREEN


def test_annotate_fully_outside_bbox_is_unknown():
    img = _image_with_light((0, 255, 0))
    det = Detection(BBox(500, 500, 600, 600), 0.9, ObjectClass.TRAFFIC_LIGHT)
    out = annotate_lights([det], img)
    assert out[0].light_state is TrafficLightState.UNKNOWN


def test_annotate_preserves_other_fields():
    img = _image_with_light((0, 0, 255))
    det = Detection(
        BBox(100, 50, 120, 90), 0.9, ObjectClass.TRAFFIC_LIGHT,
        distance_m=42.0, mask_xy=((1.0, 2.0), (3.0, 4.0), (5.0, 6.0)),
    )
    out = annotate_lights([det], img)
    assert out[0].distance_m == 42.0
    assert out[0].mask_xy == det.mask_xy
