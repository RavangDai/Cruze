"""AutoSteer ego-path masking + mapping with a fake session."""
import numpy as np
import pytest

from cruze.perception.autosteer import AutoSteerEstimator


class _FakeSession:
    def __init__(self, xp, h):
        self._out = [xp, h]
        self.input_names = ["input"]
        self.output_names = ["xp", "h"]

    def run(self, feeds):
        return self._out


def test_autosteer_keeps_masked_points_and_maps():
    xp = np.full(64, 0.5, np.float32)          # centre column → u = 512
    h = np.zeros(64, np.float32); h[0] = 1.0   # only first sample passes mask
    est = AutoSteerEstimator(model_path="", session=_FakeSession(xp, h))
    path = est.infer(np.zeros((1, 3, 512, 1024), np.float32), sx=1.25, sy=1.25, crop_top=80)
    assert path is not None and len(path) == 1
    assert path[0][0] == pytest.approx(512 * 1.25)   # u=0.5*1024 → raw x
    assert path[0][1] == pytest.approx(0 * 1.25 + 80)  # row 0 → raw y


def test_autosteer_none_when_all_masked():
    est = AutoSteerEstimator(model_path="", session=_FakeSession(
        np.full(64, 0.5, np.float32), np.zeros(64, np.float32)))
    assert est.infer(np.zeros((1, 3, 512, 1024), np.float32), 1.25, 1.25, 80) is None
