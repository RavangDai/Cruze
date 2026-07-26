"""AutoSpeed decode/unmap with a fake session — no onnxruntime."""
import numpy as np
import pytest

from cruze.core.types import ObjectClass
from cruze.perception.backends.autospeed import AutoSpeedEstimator


class _FakeSession:
    def __init__(self, out):
        self._out = out
        self.input_names = ["images"]
        self.output_names = ["output"]

    def run(self, feeds):
        return [self._out]


def test_autospeed_decodes_and_unmaps():
    raw = np.zeros((1, 8, 1), dtype=np.float32)     # 4 box + 4 class (K=4)
    raw[0, :4, 0] = [512, 256, 100, 80]             # cx,cy,w,h in net px
    raw[0, 7, 0] = 10.0                            # class id 3 → car (argmax)
    est = AutoSpeedEstimator(model_path="", conf_threshold=0.5, session=_FakeSession(raw))
    dets = est.infer(np.zeros((1, 3, 512, 1024), np.float32), sx=1.25, sy=1.25, crop_top=80)
    assert len(dets) == 1
    d = dets[0]
    assert d.bbox.x1 == pytest.approx(462 * 1.25)
    assert d.bbox.y1 == pytest.approx(216 * 1.25 + 80)
    assert d.cls is ObjectClass.CAR


def test_autospeed_unknown_class_maps_to_unknown():
    raw = np.zeros((1, 5, 1), dtype=np.float32)
    raw[0, :4, 0] = [10, 10, 4, 4]
    raw[0, 4, 0] = 10.0
    est = AutoSpeedEstimator(model_path="", conf_threshold=0.5, class_map={}, session=_FakeSession(raw))
    dets = est.infer(np.zeros((1, 3, 512, 1024), np.float32), 1.0, 1.0, 0)
    assert dets[0].cls is ObjectClass.UNKNOWN
