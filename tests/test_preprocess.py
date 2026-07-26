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


# --- decode_yolo: pre-sigmoid thresholding ---------------------------------

def _decode_reference(raw, conf_thres):
    """The original post-sigmoid decode, kept as the oracle."""
    data = raw[0]
    cx, cy, w, h = data[0], data[1], data[2], data[3]
    probs = 1.0 / (1.0 + np.exp(-data[4:]))
    class_ids = np.argmax(probs, axis=0)
    scores = probs[class_ids, np.arange(probs.shape[1])]
    keep = scores >= conf_thres
    boxes = np.stack(
        [cx[keep] - w[keep] / 2, cy[keep] - h[keep] / 2,
         cx[keep] + w[keep] / 2, cy[keep] + h[keep] / 2], axis=1,
    ).astype(np.float32)
    return boxes, scores[keep].astype(np.float32), class_ids[keep].astype(np.int64)


def test_decode_yolo_logit_path_matches_reference():
    # For a genuine logit head, thresholding the logit selects the same anchors
    # as thresholding the probability, because sigmoid is monotonic.
    rng = np.random.default_rng(1234)
    raw = rng.normal(0.0, 3.0, size=(1, 4 + 4, 500)).astype(np.float32)
    raw[0, :4] = np.abs(raw[0, :4]) * 100.0  # plausible box geometry
    assert not pp.is_activated(raw[0, 4:])

    for conf in (0.1, 0.5, 0.6, 0.9):
        boxes, scores, ids = pp.decode_yolo(raw, conf)
        e_boxes, e_scores, e_ids = _decode_reference(raw, conf)
        assert boxes.shape == e_boxes.shape
        np.testing.assert_allclose(boxes, e_boxes, rtol=1e-6)
        np.testing.assert_allclose(scores, e_scores, rtol=1e-6)
        np.testing.assert_array_equal(ids, e_ids)


def test_decode_yolo_does_not_re_activate_probabilities():
    # The exported AutoSpeed graph bakes the sigmoid in. Applying it again maps
    # [0,1] onto [0.5, 0.73], which turns a configured 0.6 threshold into an
    # effective 0.405 and reports every score in a band around 0.7.
    raw = np.zeros((1, 4 + 2, 4), dtype=np.float32)
    raw[0, :4] = 50.0
    raw[0, 4] = [0.9, 0.2, 0.65, 0.45]   # class 0 probabilities
    raw[0, 5] = [0.1, 0.8, 0.10, 0.30]   # class 1 probabilities
    assert pp.is_activated(raw[0, 4:])

    boxes, scores, ids = pp.decode_yolo(raw, 0.6)
    # Only the three anchors whose best probability clears 0.6 survive, and the
    # scores come back unchanged rather than squashed.
    np.testing.assert_allclose(sorted(scores), [0.65, 0.8, 0.9], rtol=1e-6)
    np.testing.assert_array_equal(sorted(ids), [0, 0, 1])
    assert len(boxes) == 3

    # Double-sigmoid would have admitted the 0.45 anchor too.
    assert len(pp.decode_yolo(raw, 0.6, activated=False)[0]) == 4


def test_is_activated_discriminates_logits_from_probabilities():
    assert pp.is_activated(np.array([[0.0, 0.5, 1.0]], dtype=np.float32))
    assert not pp.is_activated(np.array([[-4.0, 0.5, 9.0]], dtype=np.float32))
    assert not pp.is_activated(np.array([[0.5, 3.0]], dtype=np.float32))


def test_decode_yolo_empty_when_nothing_clears_threshold():
    raw = np.full((1, 8, 20), -50.0, dtype=np.float32)
    boxes, scores, ids = pp.decode_yolo(raw, 0.6)
    assert len(boxes) == 0 and len(scores) == 0 and len(ids) == 0


def test_logit_saturating_thresholds():
    assert pp.logit(0.0) == float("-inf")
    assert pp.logit(1.0) == float("inf")
    assert pp.logit(0.5) == pytest.approx(0.0)


def test_chw_from_bgr_u8_matches_float_path():
    # The uint8 channel swap + folded scale must be bit-comparable to the old
    # cvtColor -> astype(float32)/255 -> chw path.
    rng = np.random.default_rng(7)
    bgr = rng.integers(0, 256, size=(8, 12, 3), dtype=np.uint8)
    expected = pp.chw_from_rgb01(
        bgr[..., ::-1].astype(np.float32) / 255.0, imagenet=False
    )
    np.testing.assert_allclose(pp.chw_from_bgr_u8(bgr, imagenet=False), expected)
