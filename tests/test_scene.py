"""SceneAssembler absolute-speed tests — no bus traffic needed."""

import time

import pytest

from cruze.core.bus import EventBus
from cruze.core.config import Config
from cruze.core.types import BBox, EgoEstimate, LaneLine, Lanes, ObjectClass, Track, VehicleState
from cruze.reasoning.scene import SceneAssembler


def _assembler(ego_speed=None):
    sa = SceneAssembler(Config(), EventBus(), image_width=1280)
    sa._latest_vehicle_state = VehicleState(speed_mps=ego_speed)
    return sa


def _track(cls=ObjectClass.CAR, closing=None, dist=30.0, tid=1):
    return Track(
        track_id=tid,
        bbox=BBox(600, 300, 700, 400),
        cls=cls,
        distance_m=dist,
        closing_speed_mps=closing,
    )


def test_absolute_speed_ego_minus_closing():
    sa = _assembler(ego_speed=30.0)
    scene = sa._build_scene([_track(closing=5.0)])
    # Lead closing at 5 m/s while ego does 30 → lead absolute 25 m/s.
    assert scene.tracks[0].speed_mps == pytest.approx(25.0)


def test_absolute_speed_clamped_at_zero():
    sa = _assembler(ego_speed=3.0)
    scene = sa._build_scene([_track(closing=10.0)])
    # Math gives −7 (oncoming/stopped edge case) — clamp to 0 for display sanity.
    assert scene.tracks[0].speed_mps == 0.0


def test_no_ego_speed_means_no_absolute_speed():
    sa = _assembler(ego_speed=None)
    scene = sa._build_scene([_track(closing=5.0)])
    assert scene.tracks[0].speed_mps is None


def test_no_closing_speed_means_no_absolute_speed():
    sa = _assembler(ego_speed=30.0)
    scene = sa._build_scene([_track(closing=None)])
    assert scene.tracks[0].speed_mps is None


def test_person_gets_no_absolute_speed():
    sa = _assembler(ego_speed=30.0)
    scene = sa._build_scene([_track(cls=ObjectClass.PERSON, closing=5.0)])
    assert scene.tracks[0].speed_mps is None


def test_lead_track_carries_absolute_speed():
    sa = _assembler(ego_speed=30.0)
    scene = sa._build_scene([_track(closing=5.0)])
    assert scene.lead_track is not None
    assert scene.lead_track.speed_mps == pytest.approx(25.0)


# --- IDM required-accel attachment ---

def test_required_accel_attached_for_closing_lead():
    sa = _assembler(ego_speed=25.0)
    scene = sa._build_scene([_track(closing=8.0, dist=10.0)])
    assert scene.required_accel_mps2 is not None
    assert scene.required_accel_mps2 < -5.0  # fast-closing 10 m gap = emergency


def test_required_accel_none_without_ego_speed():
    sa = _assembler(ego_speed=None)
    scene = sa._build_scene([_track(closing=5.0)])
    assert scene.required_accel_mps2 is None


def test_required_accel_positive_on_free_road():
    sa = _assembler(ego_speed=10.0)
    scene = sa._build_scene([])
    assert scene.required_accel_mps2 is not None
    assert scene.required_accel_mps2 > 0.0


# --- Lane fold-in ---

def _lanes(age_s=0.0):
    return Lanes(
        left=LaneLine(100.0, 720.0, 550.0, 320.0),
        right=LaneLine(1180.0, 720.0, 730.0, 320.0),
        timestamp=time.monotonic() - age_s,
    )


def test_fresh_lanes_attached_to_scene():
    sa = _assembler()
    sa._latest_lanes = _lanes()
    sa._lanes_seen_at = time.monotonic()
    scene = sa._build_scene([])
    assert scene.lanes is not None
    assert scene.lanes.left.x1 == pytest.approx(100.0)


def test_lanes_attached_despite_old_frame_timestamp():
    """Regression: freshness was anchored to the frame CAPTURE timestamp, so
    on a loaded CPU (perception latency > 0.5 s) every lane result was
    silently dropped. Freshness must mean 'the detector is still producing',
    i.e. message arrival time."""
    sa = _assembler()
    sa._latest_lanes = _lanes(age_s=2.0)  # frame captured 2 s ago
    sa._lanes_seen_at = time.monotonic()  # ...but the message just arrived
    assert sa._build_scene([]).lanes is not None


def test_lanes_dropped_when_detector_stops_producing():
    sa = _assembler()
    sa._latest_lanes = _lanes()
    sa._lanes_seen_at = time.monotonic() - 1.0  # nothing received for 1 s
    assert sa._build_scene([]).lanes is None


def test_no_lanes_means_none():
    sa = _assembler()
    assert sa._build_scene([]).lanes is None


# --- Ego bundle fold-in ---

def test_scene_folds_fresh_ego():
    asm = SceneAssembler(Config(), EventBus(), image_width=1280)
    asm._latest_ego = EgoEstimate(
        cipo_distance_m=42.0, road_curvature_1pm=0.01, cipo_flag=True, ego_path=((1.0, 2.0),))
    asm._ego_seen_at = time.monotonic()
    scene = asm._build_scene([])
    assert scene.cipo_distance_m == 42.0
    assert scene.road_curvature_1pm == 0.01
    assert scene.cipo_flag is True
    assert scene.ego_path == ((1.0, 2.0),)


def test_scene_drops_stale_ego():
    asm = SceneAssembler(Config(), EventBus(), image_width=1280)
    asm._latest_ego = EgoEstimate(cipo_distance_m=42.0)
    asm._ego_seen_at = time.monotonic() - 5.0   # older than the freshness window
    scene = asm._build_scene([])
    assert scene.cipo_distance_m is None
    assert scene.cipo_flag is None


# --- lead selection by ground position -------------------------------------

def _vehicle(tid, bbox, distance_m, ground_xz_m=None):
    return Track(track_id=tid, bbox=bbox, cls=ObjectClass.CAR,
                 distance_m=distance_m, ground_xz_m=ground_xz_m)


def test_lead_uses_metres_when_ground_position_known():
    # The pixel proxy depends on a configured image width that a replayed file
    # ignores. With a ground position the test becomes the physical one.
    cfg = Config()
    cfg.camera.width = 1920
    asm = SceneAssembler(cfg, EventBus(), image_width=1920)

    # Far right of a 1920-wide frame — the pixel proxy would reject it — but
    # only 0.5 m laterally, so it is genuinely in the ego lane.
    in_lane = _vehicle(1, BBox(1500, 600, 1700, 780), 30.0, ground_xz_m=(0.5, 30.0))
    # Near frame centre but 5 m across: an adjacent lane, and nearer, so it
    # would win on distance alone.
    beside = _vehicle(2, BBox(900, 600, 1000, 700), 12.0, ground_xz_m=(5.0, 12.0))

    lead = asm._find_lead([in_lane, beside])
    assert lead is not None and lead.track_id == 1


def test_lead_falls_back_to_pixel_proxy_without_ground_position():
    asm = SceneAssembler(Config(), EventBus(), image_width=1280)
    centred = _vehicle(1, BBox(600, 400, 700, 500), 25.0)
    edge = _vehicle(2, BBox(20, 400, 90, 500), 10.0)

    lead = asm._find_lead([centred, edge])
    assert lead is not None and lead.track_id == 1


def test_no_lead_when_every_vehicle_is_out_of_lane():
    asm = SceneAssembler(Config(), EventBus(), image_width=1280)
    left = _vehicle(1, BBox(0, 400, 60, 500), 20.0, ground_xz_m=(-6.0, 20.0))
    right = _vehicle(2, BBox(1800, 400, 1900, 500), 18.0, ground_xz_m=(6.0, 18.0))

    assert asm._find_lead([left, right]) is None


def test_edge_truncated_vehicle_is_not_the_lead():
    # A box running off the side of the frame has no meaningful centre and its
    # ground-plane range is measured against the boundary, not the object.
    # Treating one as the lead produces a stream of phantom collision warnings.
    cfg = Config()
    cfg.camera.width = 1920
    asm = SceneAssembler(cfg, EventBus(), image_width=1920)

    clipped = _vehicle(1, BBox(1500, 700, 1920, 1280), 2.5, ground_xz_m=(1.3, 2.5))
    ahead = _vehicle(2, BBox(900, 600, 1050, 720), 40.0, ground_xz_m=(0.2, 40.0))

    lead = asm._find_lead([clipped, ahead])
    assert lead is not None and lead.track_id == 2


def test_bottom_truncated_lead_is_kept():
    # Close following legitimately clips the bottom edge; dropping those would
    # lose the real target exactly when a warning matters most.
    cfg = Config()
    cfg.camera.width = 1920
    asm = SceneAssembler(cfg, EventBus(), image_width=1920)

    close = _vehicle(1, BBox(700, 500, 1200, 1280), 6.0, ground_xz_m=(0.1, 6.0))
    assert asm._find_lead([close]) is not None


def test_frame_width_follows_corrected_camera_config():
    # CameraService rewrites camera config once it sees a real frame; the
    # lane geometry must pick that up rather than the construction-time value.
    cfg = Config()
    cfg.camera.width = 1280
    asm = SceneAssembler(cfg, EventBus(), image_width=1280)
    assert asm._frame_width == 1280
    cfg.camera.width = 1920
    assert asm._frame_width == 1920
