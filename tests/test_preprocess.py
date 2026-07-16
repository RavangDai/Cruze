"""Preprocessing helpers — pure numpy, no cv2/onnx."""
import numpy as np
import pytest

from cruze.perception import preprocess as pp


def test_top_crop_keeps_bottom_2_1():
    # 1280x720: keep bottom W/2 = 640 rows → crop 80 from the top.
    assert pp.top_crop_2_1(720, 1280) == 80


def test_top_crop_never_negative():
    # Already taller-than-2:1 region → no crop.
    assert pp.top_crop_2_1(400, 1280) == 0


def test_crop_resize_params():
    crop_top, sx, sy = pp.crop_resize_params(720, 1280)
    assert crop_top == 80
    assert sx == pytest.approx(1280 / 1024)
    assert sy == pytest.approx((720 - 80) / 512)


def test_unmap_corners_round_trip():
    crop_top, sx, sy = pp.crop_resize_params(720, 1280)
    assert pp.unmap_point(0, 0, sx, sy, crop_top) == pytest.approx((0.0, 80.0))
    assert pp.unmap_point(1024, 512, sx, sy, crop_top) == pytest.approx((1280.0, 720.0))


def test_chw_shape_and_plain_scaling():
    rgb01 = np.full((512, 1024, 3), 0.5, dtype=np.float32)
    chw = pp.chw_from_rgb01(rgb01, imagenet=False)
    assert chw.shape == (1, 3, 512, 1024)
    assert chw.dtype == np.float32
    assert chw[0, 0, 0, 0] == pytest.approx(0.5)


def test_chw_imagenet_norm():
    rgb01 = np.zeros((1, 1, 3), dtype=np.float32)  # tiny image, values 0
    chw = pp.chw_from_rgb01(rgb01, imagenet=True)
    # (0 - mean)/std for the red channel.
    assert chw[0, 0, 0, 0] == pytest.approx((0.0 - 0.485) / 0.229)


def test_decode_yolo_thresholds_and_boxes():
    # raw [1, C=5, N=2]: 4 box params + 1 class logit.
    raw = np.zeros((1, 5, 2), dtype=np.float32)
    raw[0, :4, 0] = [512, 256, 100, 80]   # anchor 0: cx,cy,w,h
    raw[0, 4, 0] = 10.0                    # logit → sigmoid ≈ 1 (kept)
    raw[0, 4, 1] = -10.0                   # anchor 1 → sigmoid ≈ 0 (dropped)
    boxes, scores, class_ids = pp.decode_yolo(raw, conf_thres=0.5)
    assert boxes.shape == (1, 4)
    assert boxes[0] == pytest.approx([462, 216, 562, 296])
    assert class_ids[0] == 0
    assert scores[0] > 0.99


def test_nms_suppresses_overlap():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [100, 100, 110, 110]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    keep = pp.nms(boxes, scores, iou_thres=0.5)
    assert keep == [0, 2]  # box 1 suppressed by box 0; distant box survives
