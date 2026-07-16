import numpy as np
import pytest

import cruze.perception.vision_nets as vn
from cruze.core.types import BBox, Detection, Frame, ObjectClass


class _FakeAutoSpeed:
    def infer(self, chw, sx, sy, crop_top):
        return (Detection(BBox(0, 0, 10, 10), 0.9, ObjectClass.CAR),)


def test_infer_none_when_no_nets():
    assert vn.VisionNets().infer(Frame(image=None)) is None
    assert vn.VisionNets().any_enabled is False


def test_infer_bundles_autospeed(monkeypatch):
    monkeypatch.setattr(vn.preprocess, "preprocess_crop2_1",
                        lambda img: (np.zeros((1, 3, 512, 1024), np.float32), 1.25, 1.25, 80))
    nets = vn.VisionNets(autospeed=_FakeAutoSpeed())
    frame = Frame(image=np.zeros((720, 1280, 3), np.uint8), frame_id=7)
    ego = nets.infer(frame)
    assert ego is not None
    assert ego.frame_id == 7
    assert len(ego.cipo_boxes) == 1
    assert ego.cipo_boxes[0].cls is ObjectClass.CAR


@pytest.mark.asyncio
async def test_perception_publishes_ego():
    from cruze.core.bus import Channel, EventBus
    from cruze.core.config import Config
    from cruze.core.types import EgoEstimate
    from cruze.perception import detector as det_mod
    from cruze.perception.pipeline import PerceptionService

    class _FakeNets:
        any_enabled = True
        def infer(self, frame):
            return EgoEstimate(frame_id=frame.frame_id, cipo_flag=True)

    bus = EventBus()
    svc = PerceptionService(Config(), bus, det_mod.load("stub"), vision_nets=_FakeNets())
    ego_q = bus.subscribe(Channel.PERCEPTION_EGO, maxsize=2)
    await svc._process_frame(Frame(image=np.zeros((720, 1280, 3), np.uint8), frame_id=3))
    ego = ego_q.get_nowait()
    assert ego.frame_id == 3 and ego.cipo_flag is True
