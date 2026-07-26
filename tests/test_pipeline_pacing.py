"""Perception pacing: the rate gate, the freshest-frame drain, and the
low-rate context cadence. No ML deps — the detector is the stub backend and the
nets are fakes."""

import asyncio

import numpy as np
import pytest

from cruze.core.bus import Channel, EventBus
from cruze.core.config import Config
from cruze.core.types import BBox, Detection, EgoEstimate, Frame, ObjectClass
from cruze.perception import detector as det_mod
from cruze.perception.pipeline import PerceptionService, _drain_to_newest


def _frame(frame_id: int) -> Frame:
    return Frame(image=np.zeros((720, 1280, 3), np.uint8), frame_id=frame_id)


class _FakeNets:
    """Stands in for VisionNets with AutoSpeed active."""

    any_enabled = True
    has_vehicle_source = True

    def __init__(self) -> None:
        self.calls = 0

    def infer(self, frame):
        self.calls += 1
        return EgoEstimate(
            frame_id=frame.frame_id,
            cipo_boxes=(
                Detection(BBox(10.0, 10.0, 50.0, 50.0), 0.9, ObjectClass.CAR),
            ),
        )


class _CountingDetector:
    """Context detector that records how often it is asked to run."""

    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return []


def _service(bus, nets=None, detector=None, **perception):
    cfg = Config()
    for key, value in perception.items():
        setattr(cfg.perception, key, value)
    cfg.perception.lane_detection_enabled = False
    return PerceptionService(
        cfg, bus, detector or det_mod.load("stub"), vision_nets=nets
    )


def test_drain_to_newest_returns_last_queued_frame():
    # The bus queue is drop-oldest, so get() hands back the oldest buffered
    # frame. Perception only ever wants the most recent view of the road.
    queue: asyncio.Queue = asyncio.Queue(maxsize=8)
    for i in range(1, 5):
        queue.put_nowait(_frame(i))
    first = queue.get_nowait()
    assert first.frame_id == 1
    assert _drain_to_newest(queue, first).frame_id == 4
    assert queue.empty()


def test_drain_to_newest_passes_through_when_queue_empty():
    queue: asyncio.Queue = asyncio.Queue(maxsize=8)
    frame = _frame(9)
    assert _drain_to_newest(queue, frame) is frame


@pytest.mark.asyncio
async def test_rate_gate_skips_frames_faster_than_target_hz():
    bus = EventBus()
    # 2 Hz → a 500 ms period. Every frame below arrives well inside it, so
    # exactly one is processed and the rest are dropped here rather than
    # queued behind it.
    svc = _service(bus, nets=_FakeNets(), target_hz=2.0)
    task = asyncio.create_task(svc.run())
    await asyncio.sleep(0.05)  # let run() reach bus.subscribe()

    for i in range(5):
        await bus.publish(Channel.PERCEPTION_FRAME, _frame(i))
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.05)

    await svc.stop()
    task.cancel()
    assert svc.metrics["frame_count"] == 1
    assert svc.metrics["skipped_count"] >= 1


@pytest.mark.asyncio
async def test_every_frame_processed_when_rate_gate_disabled():
    bus = EventBus()
    svc = _service(bus, nets=_FakeNets(), target_hz=0.0)
    task = asyncio.create_task(svc.run())
    await asyncio.sleep(0.05)

    for i in range(3):
        await bus.publish(Channel.PERCEPTION_FRAME, _frame(i))
        await asyncio.sleep(0.02)
    await asyncio.sleep(0.05)

    await svc.stop()
    task.cancel()
    assert svc.metrics["frame_count"] == 3
    assert svc.metrics["skipped_count"] == 0


@pytest.mark.asyncio
async def test_context_pass_runs_at_its_own_cadence():
    # AutoSpeed covers vehicles every frame; the general detector is demoted,
    # so it must not run on every processed frame.
    bus = EventBus()
    detector = _CountingDetector()
    svc = _service(bus, nets=_FakeNets(), detector=detector,
                   target_hz=0.0, context_hz=1.0)

    for i in range(4):
        await svc._process_frame(_frame(i))

    assert detector.calls == 1, "context detector should be gated by context_hz"


@pytest.mark.asyncio
async def test_context_detections_are_cached_between_passes():
    class _PersonDetector:
        def detect(self, frame):
            return [Detection(BBox(0.0, 0.0, 20.0, 60.0), 0.8, ObjectClass.PERSON)]

    bus = EventBus()
    svc = _service(bus, nets=_FakeNets(), detector=_PersonDetector(),
                   target_hz=0.0, context_hz=1.0)
    tracks_q = bus.subscribe(Channel.PERCEPTION_TRACKS, maxsize=8)

    await svc._process_frame(_frame(0))   # context runs
    await svc._process_frame(_frame(1))   # context skipped — cache must persist

    tracks_q.get_nowait()
    second = tracks_q.get_nowait()
    classes = {t.cls for t in second}
    assert ObjectClass.PERSON in classes, "cached context detection was dropped"
    assert ObjectClass.CAR in classes, "AutoSpeed box missing from hot path"


@pytest.mark.asyncio
async def test_context_detector_runs_every_frame_without_a_vehicle_source():
    # No AutoSpeed: the general detector is the only thing that sees a car, so
    # the context cadence must be bypassed entirely.
    bus = EventBus()
    detector = _CountingDetector()
    svc = _service(bus, nets=None, detector=detector, target_hz=0.0, context_hz=1.0)

    for i in range(3):
        await svc._process_frame(_frame(i))

    assert detector.calls == 3


@pytest.mark.asyncio
async def test_tracks_carry_the_frame_they_came_from():
    bus = EventBus()
    svc = _service(bus, nets=_FakeNets(), target_hz=0.0)
    tracks_q = bus.subscribe(Channel.PERCEPTION_TRACKS, maxsize=4)

    await svc._process_frame(_frame(77))

    tracks = tracks_q.get_nowait()
    assert tracks and all(t.frame_id == 77 for t in tracks)
