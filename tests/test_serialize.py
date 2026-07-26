"""hmi.serialize tests — everything must be json.dumps-able with no heavy deps."""

import dataclasses
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
    # vision_pilot ego fields absent (nets disabled) → all null on the wire.
    assert d["ego_path"] is None
    assert d["curvature"] is None
    assert d["cipo_dist"] is None
    assert d["cipo"] is None


def test_scene_accel_serialized():
    d = scene_to_dict(Scene(required_accel_mps2=-4.567))
    json.dumps(d)
    assert d["accel"] == -4.57


def test_ego_fields_serialized():
    # vision_pilot outputs: ego_path flattened + rounded, curvature to 4dp
    # (values are ~0.002), cipo distance to 1dp, cipo flag passed through.
    scene = Scene(
        ego_path=((10.04, 20.06), (30.0, 40.0), (50.58, 60.0)),
        road_curvature_1pm=0.00234,
        cipo_distance_m=42.37,
        cipo_flag=True,
    )
    d = scene_to_dict(scene)
    json.dumps(d)  # must not raise
    assert d["ego_path"] == [10.0, 20.1, 30.0, 40.0, 50.6, 60.0]
    assert d["curvature"] == 0.0023
    assert d["cipo_dist"] == 42.4
    assert d["cipo"] is True


def test_ego_path_point_cap():
    # Defensive cap bounds wire size even if a producer over-samples.
    from cruze.hmi.serialize import _MAX_EGO_POINTS

    pts = tuple((float(i), float(i)) for i in range(_MAX_EGO_POINTS + 10))
    d = scene_to_dict(Scene(ego_path=pts))
    json.dumps(d)
    assert len(d["ego_path"]) == _MAX_EGO_POINTS * 2


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


def test_track_mask_not_on_the_wire():
    # Masks left the protocol: the general detector now runs at a few Hz, so an
    # outline would be stale by the time it was drawn, and the overlay is
    # stroke-based with nothing to fill. The upstream field still exists.
    t = _track(mask=((10.04, 20.06), (30.0, 40.0), (50.58, 20.0)))
    assert t.mask_xy is not None
    d = scene_to_dict(Scene(tracks=(t,)))
    json.dumps(d)
    assert "mask" not in d["tracks"][0]


def test_track_without_optional_fields_serializes_null():
    d = scene_to_dict(Scene(tracks=(_track(),)))
    json.dumps(d)
    assert d["tracks"][0]["light"] is None
    assert d["tracks"][0]["xz"] is None


def test_track_ground_position_serialized():
    # [lateral, forward] metres — drives the plan view.
    t = dataclasses.replace(_track(), ground_xz_m=(-1.234, 24.56))
    d = scene_to_dict(Scene(tracks=(t,)))
    json.dumps(d)
    assert d["tracks"][0]["xz"] == [-1.2, 24.6]


def test_scene_carries_frame_id():
    # The dashboard pairs each JPEG with the scene sharing its frame_id.
    d = scene_to_dict(Scene(frame_id=4242))
    json.dumps(d)
    assert d["frame_id"] == 4242


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
