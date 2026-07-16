"""Data-contract tests for additive fields (backwards compatibility)."""

import dataclasses

from cruze.core.types import (
    BBox,
    Detection,
    LaneLine,
    Lanes,
    ObjectClass,
    Scene,
    Track,
    TrafficLightState,
    VehicleState,
)


def test_track_speed_mps_defaults_none():
    t = Track(track_id=1, bbox=BBox(0, 0, 10, 10), cls=ObjectClass.CAR)
    assert t.speed_mps is None


def test_track_speed_mps_settable_via_replace():
    t = Track(track_id=1, bbox=BBox(0, 0, 10, 10), cls=ObjectClass.CAR)
    t2 = dataclasses.replace(t, speed_mps=25.0)
    assert t2.speed_mps == 25.0
    assert t2.track_id == 1
    assert t.speed_mps is None  # original frozen instance unchanged


def test_vehicle_state_gps_speed_defaults_none():
    v = VehicleState()
    assert v.gps_speed_mps is None


def test_vehicle_state_gps_speed_settable_via_replace():
    v = VehicleState()
    v2 = dataclasses.replace(v, gps_speed_mps=12.5)
    assert v2.gps_speed_mps == 12.5
    assert v.gps_speed_mps is None


# --- Segmentation / light-state / lane additions ---

def test_detection_mask_and_light_default_none():
    d = Detection(bbox=BBox(0, 0, 10, 10), confidence=0.9, cls=ObjectClass.CAR)
    assert d.mask_xy is None
    assert d.light_state is None


def test_detection_new_fields_settable_via_replace():
    d = Detection(bbox=BBox(0, 0, 10, 10), confidence=0.9, cls=ObjectClass.TRAFFIC_LIGHT)
    poly = ((0.0, 0.0), (10.0, 0.0), (5.0, 10.0))
    d2 = dataclasses.replace(d, mask_xy=poly, light_state=TrafficLightState.RED)
    assert d2.mask_xy == poly
    assert d2.light_state is TrafficLightState.RED
    assert d.mask_xy is None  # original frozen instance unchanged


def test_track_mask_and_light_default_none():
    t = Track(track_id=1, bbox=BBox(0, 0, 10, 10), cls=ObjectClass.CAR)
    assert t.mask_xy is None
    assert t.light_state is None


def test_traffic_light_state_values():
    assert TrafficLightState.RED.value == "red"
    assert TrafficLightState.YELLOW.value == "yellow"
    assert TrafficLightState.GREEN.value == "green"
    assert TrafficLightState.UNKNOWN.value == "unknown"


def test_scene_lanes_defaults_none():
    assert Scene().lanes is None


def test_lanes_holds_lane_lines():
    left = LaneLine(x1=100.0, y1=720.0, x2=550.0, y2=320.0)
    lanes = Lanes(left=left, right=None, frame_id=7)
    scene = Scene(lanes=lanes)
    assert scene.lanes.left == left
    assert scene.lanes.right is None
    assert scene.lanes.frame_id == 7


# --- EgoEstimate and ego fields additions (Task 3) ---

from cruze.core.types import EgoEstimate
from cruze.core.bus import Channel


def test_ego_estimate_defaults():
    ego = EgoEstimate()
    assert ego.cipo_boxes == ()
    assert ego.ego_path is None
    assert ego.cipo_distance_m is None
    assert ego.road_curvature_1pm is None
    assert ego.cipo_flag is None


def test_ego_estimate_carries_boxes():
    d = Detection(BBox(0, 0, 1, 1), 0.9, ObjectClass.CAR)
    ego = EgoEstimate(cipo_boxes=(d,), cipo_distance_m=42.0, cipo_flag=True)
    assert ego.cipo_boxes[0].cls is ObjectClass.CAR
    assert ego.cipo_distance_m == 42.0


def test_scene_ego_fields_default_none():
    s = Scene()
    assert s.ego_path is None
    assert s.road_curvature_1pm is None
    assert s.cipo_distance_m is None
    assert s.cipo_flag is None


def test_channel_ego_constant():
    assert Channel.PERCEPTION_EGO == "perception.ego"
