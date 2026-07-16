"""AutoDrive buffer + domain conversion — cv2 warp monkeypatched away."""
import numpy as np
import pytest

import cruze.perception.autodrive as ad


class _FakeSession:
    def __init__(self, outputs):
        self._outputs = outputs
        self.input_names = ["prev", "curr"]
        self.output_names = ["dist", "curv", "flag"]

    def run(self, feeds):
        return self._outputs


def test_first_frame_returns_none_then_converts(monkeypatch):
    monkeypatch.setattr(ad.preprocess, "preprocess_bev",
                        lambda img, H: np.zeros((1, 3, 512, 1024), np.float32))
    fake = _FakeSession([np.array([[0.4]]), np.array([[0.02]]), np.array([[0.9]])])
    est = ad.AutoDriveEstimator(model_path="", homography=np.eye(3, dtype=np.float32),
                                curv_scale=2.0, flag_threshold=0.5, session=fake)
    img = np.zeros((720, 1280, 3), np.uint8)
    assert est.infer(img) is None                      # first frame buffers
    res = est.infer(img)
    assert res.cipo_distance_m == pytest.approx(150.0 * (1 - 0.4))  # 90.0
    assert res.road_curvature_1pm == pytest.approx(0.02 * 2.0)      # 0.04
    assert res.cipo_flag is True


def test_flag_below_threshold_is_false(monkeypatch):
    monkeypatch.setattr(ad.preprocess, "preprocess_bev",
                        lambda img, H: np.zeros((1, 3, 512, 1024), np.float32))
    fake = _FakeSession([np.array([[0.0]]), np.array([[0.0]]), np.array([[0.3]])])
    est = ad.AutoDriveEstimator(model_path="", homography=np.eye(3, dtype=np.float32), session=fake)
    img = np.zeros((720, 1280, 3), np.uint8)
    est.infer(img)
    assert est.infer(img).cipo_flag is False


def test_load_homography(tmp_path):
    # Non-symmetric matrix so a row/column-major reshape bug would be caught:
    # row-major reshape of [1..9] puts 2 at [0,1] and 4 at [1,0]; a
    # column-major bug would swap them.
    p = tmp_path / "C.yaml"
    p.write_text("C: [1, 2, 3, 4, 5, 6, 7, 8, 9]")
    H = ad.load_homography(str(p))
    assert H.shape == (3, 3)
    assert H[0, 0] == pytest.approx(1.0)
    assert H[0, 1] == pytest.approx(2.0)
    assert H[1, 0] == pytest.approx(4.0)
