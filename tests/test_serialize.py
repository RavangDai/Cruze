"""hmi.serialize tests — everything must be json.dumps-able with no heavy deps."""

import json

from cruze.core.types import (
    BBox,
    DrivingEvent,
    EventLevel,
    LaneLine,
    Lanes,
    ObjectClass,
    Scene,
    Track,
    TrafficLightState,
    Utterance,
    VehicleState,
)
from cruze.hmi.serialize import event_to_dict, scene_to_dict, utterance_to_dict


def _track(tid=1, speed=25.0, mask=None, light=None, cls=ObjectClass.CAR):
    return Track(
        track_id=tid,
        bbox=BBox(10.5, 20.5, 110.5, 120.5),
        cls=cls,
        distance_m=34.27,
        closing_speed_mps=4.789,
        speed_mps=speed,
        mask_xy=mask,
        light_state=light,
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
    assert d["lanes"] is None
    assert d["accel"] is None


def test_scene_accel_serialized():
    d = scene_to_dict(Scene(required_accel_mps2=-4.567))
    json.dumps(d)
    assert d["accel"] == -4.57


def test_lane_polylines_and_cte_serialized():
    lanes = Lanes(
        left=LaneLine(100.0, 720.0, 550.0, 320.0),
        right=None,
        left_poly=((100.04, 720.0), (300.0, 500.0), (550.0, 320.0)),
        right_poly=None,
        cte_m=0.444,
    )
    d = scene_to_dict(Scene(lanes=lanes))
    json.dumps(d)
    assert d["lanes"]["left_poly"] == [100.0, 720.0, 300.0, 500.0, 550.0, 320.0]
    assert d["lanes"]["right_poly"] is None
    assert d["lanes"]["cte_m"] == 0.44


def test_lanes_without_new_fields_serialize_null():
    d = scene_to_dict(Scene(lanes=Lanes(left=LaneLine(1.0, 2.0, 3.0, 4.0))))
    json.dumps(d)
    assert d["lanes"]["left_poly"] is None
    assert d["lanes"]["cte_m"] is None


def test_track_mask_serialized_flat():
    # Flat [x1,y1,x2,y2,...] halves JSON overhead vs nested pairs.
    t = _track(mask=((10.04, 20.06), (30.0, 40.0), (50.58, 20.0)))
    d = scene_to_dict(Scene(tracks=(t,)))
    json.dumps(d)
    assert d["tracks"][0]["mask"] == [10.0, 20.1, 30.0, 40.0, 50.6, 20.0]


def test_track_without_mask_serializes_null():
    d = scene_to_dict(Scene(tracks=(_track(),)))
    json.dumps(d)
    assert d["tracks"][0]["mask"] is None
    assert d["tracks"][0]["light"] is None


def test_track_light_state_serialized():
    t = _track(light=TrafficLightState.RED, cls=ObjectClass.TRAFFIC_LIGHT)
    d = scene_to_dict(Scene(tracks=(t,)))
    json.dumps(d)
    assert d["tracks"][0]["light"] == "red"


def test_lanes_serialized():
    lanes = Lanes(
        left=LaneLine(100.04, 720.0, 550.0, 320.0),
        right=None,
    )
    d = scene_to_dict(Scene(lanes=lanes))
    json.dumps(d)
    assert d["lanes"]["left"] == [100.0, 720.0, 550.0, 320.0]
    assert d["lanes"]["right"] is None


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
