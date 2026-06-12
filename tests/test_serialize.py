"""hmi.serialize tests — everything must be json.dumps-able with no heavy deps."""

import json

from cruze.core.types import (
    BBox,
    DrivingEvent,
    EventLevel,
    ObjectClass,
    Scene,
    Track,
    Utterance,
    VehicleState,
)
from cruze.hmi.serialize import event_to_dict, scene_to_dict, utterance_to_dict


def _track(tid=1, speed=25.0):
    return Track(
        track_id=tid,
        bbox=BBox(10.5, 20.5, 110.5, 120.5),
        cls=ObjectClass.CAR,
        distance_m=34.27,
        closing_speed_mps=4.789,
        speed_mps=speed,
    )


def test_scene_to_dict_roundtrips_json():
    scene = Scene(
        tracks=(_track(),),
        vehicle_state=VehicleState(speed_mps=27.7, speed_mps_source="gps",
                                   latitude=37.7, longitude=-122.4,
                                   posted_speed_limit_mps=29.0),
        lead_track=_track(),
    )
    d = scene_to_dict(scene)
    payload = json.dumps(d)  # must not raise
    back = json.loads(payload)
    assert back["type"] == "scene"
    assert back["lead_id"] == 1
    assert back["tracks"][0]["cls"] == "car"
    assert back["tracks"][0]["speed_mps"] == 25.0
    assert back["state"]["source"] == "gps"
    assert back["state"]["lat"] == 37.7


def test_scene_to_dict_handles_all_none():
    d = scene_to_dict(Scene())
    json.dumps(d)
    assert d["lead_id"] is None
    assert d["tracks"] == []
    assert d["state"]["speed_mps"] is None


def test_event_to_dict():
    ev = DrivingEvent(kind="fcw", level=EventLevel.CRITICAL, context={"ttc_s": 1.2})
    d = event_to_dict(ev)
    json.dumps(d)
    assert d == {"type": "event", "kind": "fcw", "level": "critical",
                 "ts": ev.timestamp, "context": {"ttc_s": 1.2}}


def test_utterance_to_dict():
    u = Utterance(text="Watch the gap.", priority=2)
    d = utterance_to_dict(u)
    json.dumps(d)
    assert d["type"] == "utterance"
    assert d["text"] == "Watch the gap."
