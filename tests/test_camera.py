"""CameraService intrinsic correction — no capture device needed."""

from cruze.core.bus import EventBus
from cruze.core.config import CameraConfig
from cruze.perception.camera import CameraService


def _service(config_width, focal):
    cfg = CameraConfig(width=config_width, height=720)
    return CameraService(cfg, EventBus(), focal_length_px=focal)


def test_focal_unchanged_when_frames_match_config():
    svc = _service(1280, 914.0)
    assert svc._focal_for(1280) == 914.0


def test_focal_scales_with_actual_frame_width():
    # A replayed 1920-wide file against a config that asked for 1280: focal
    # length is proportional to width for a fixed field of view, so leaving it
    # would under-report every distance by a third.
    svc = _service(1280, 914.0)
    assert svc._focal_for(1920) == 914.0 * 1920 / 1280


def test_focal_correction_applies_once():
    svc = _service(1280, 914.0)
    corrected = svc._focal_for(1920)
    assert svc._focal_for(1920) == corrected


def test_focal_stays_none_when_uncalibrated():
    svc = _service(1280, None)
    assert svc._focal_for(1920) is None
