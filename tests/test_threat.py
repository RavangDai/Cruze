"""
Threat assessment unit tests — pure functions, no async, no deps.
"""

import math
import pytest

from cruze.core.types import BBox, EventLevel, ObjectClass, Scene, Track, VehicleState
from cruze.reasoning import threat
from cruze.core.config import ReasoningConfig


def _cfg(**overrides) -> ReasoningConfig:
    cfg = ReasoningConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _track(
    distance_m=20.0,
    closing_speed_mps=5.0,
    cls=ObjectClass.CAR,
    track_id=1,
    cx=640,
) -> Track:
    bbox = BBox(cx - 30, 300, cx + 30, 400)
    return Track(
        track_id=track_id,
        bbox=bbox,
        cls=cls,
        distance_m=distance_m,
        closing_speed_mps=closing_speed_mps,
    )


def _scene(
    lead_distance_m=20.0,
    lead_closing_mps=5.0,
    ego_speed_mps=15.0,
    posted_limit_mps=13.4,
    lead_cls=ObjectClass.CAR,
    extra_tracks=(),
    no_lead=False,
) -> Scene:
    lead = None if no_lead else _track(lead_distance_m, lead_closing_mps, cls=lead_cls)
    tracks = (lead,) + tuple(extra_tracks) if lead else tuple(extra_tracks)
    state = VehicleState(
        speed_mps=ego_speed_mps,
        speed_mps_source="simulated",
        posted_speed_limit_mps=posted_limit_mps,
    )
    return Scene(tracks=tuple(t for t in tracks if t), vehicle_state=state, lead_track=lead)


# --- time_to_collision ---

def test_ttc_normal():
    assert threat.time_to_collision(30.0, 10.0) == pytest.approx(3.0)


def test_ttc_not_closing():
    assert threat.time_to_collision(30.0, 0.0) == math.inf


def test_ttc_receding():
    assert threat.time_to_collision(30.0, -5.0) == math.inf


def test_ttc_zero_distance():
    assert threat.time_to_collision(0.0, 10.0) == math.inf


def test_ttc_very_close():
    assert threat.time_to_collision(1.0, 10.0) == pytest.approx(0.1)


# --- following_gap_seconds ---

def test_gap_normal():
    assert threat.following_gap_seconds(20.0, 10.0) == pytest.approx(2.0)


def test_gap_stationary_ego():
    assert threat.following_gap_seconds(20.0, 0.0) == math.inf


def test_gap_zero_distance():
    assert threat.following_gap_seconds(0.0, 10.0) == math.inf


# --- is_forward_collision_warning ---

def test_fcw_triggers_below_threshold():
    scene = _scene(lead_distance_m=10.0, lead_closing_mps=10.0, ego_speed_mps=15.0)
    # TTC = 1.0 s < 3.0 threshold
    assert threat.is_forward_collision_warning(scene, _cfg())


def test_fcw_does_not_trigger_above_threshold():
    scene = _scene(lead_distance_m=50.0, lead_closing_mps=5.0, ego_speed_mps=15.0)
    # TTC = 10.0 s > 3.0 threshold
    assert not threat.is_forward_collision_warning(scene, _cfg())


def test_fcw_no_lead():
    scene = _scene(no_lead=True, ego_speed_mps=20.0)
    assert not threat.is_forward_collision_warning(scene, _cfg())


def test_fcw_stationary_ego():
    # Ego not moving → no FCW even if something is close.
    scene = _scene(lead_distance_m=5.0, lead_closing_mps=10.0, ego_speed_mps=0.0)
    assert not threat.is_forward_collision_warning(scene, _cfg())


def test_fcw_not_closing():
    scene = _scene(lead_distance_m=10.0, lead_closing_mps=-2.0, ego_speed_mps=15.0)
    assert not threat.is_forward_collision_warning(scene, _cfg())


def test_fcw_custom_threshold():
    scene = _scene(lead_distance_m=10.0, lead_closing_mps=10.0, ego_speed_mps=15.0)
    # TTC = 1.0 s; threshold = 0.5 s → no FCW
    assert not threat.is_forward_collision_warning(scene, _cfg(fcw_ttc_threshold_s=0.5))


# --- is_tailgating ---

def test_tailgating_close_gap():
    # gap = 5 m / 15 m/s ≈ 0.33 s < 2.0 threshold
    scene = _scene(lead_distance_m=5.0, ego_speed_mps=15.0)
    assert threat.is_tailgating(scene, _cfg())


def test_tailgating_safe_gap():
    # gap = 40 m / 15 m/s ≈ 2.67 s > 2.0 threshold
    scene = _scene(lead_distance_m=40.0, ego_speed_mps=15.0)
    assert not threat.is_tailgating(scene, _cfg())


def test_tailgating_stopped_ego():
    scene = _scene(lead_distance_m=2.0, ego_speed_mps=0.0)
    assert not threat.is_tailgating(scene, _cfg())


def test_tailgating_no_lead():
    scene = _scene(no_lead=True)
    assert not threat.is_tailgating(scene, _cfg())


# --- is_speeding ---

def test_speeding_over_limit():
    # 20 m/s ego, 13.4 limit + 1.4 grace = 14.8 → 20 > 14.8
    scene = _scene(ego_speed_mps=20.0, posted_limit_mps=13.4)
    assert threat.is_speeding(scene, _cfg())


def test_speeding_within_grace():
    # 14.0 m/s < 13.4 + 1.4 = 14.8
    scene = _scene(ego_speed_mps=14.0, posted_limit_mps=13.4)
    assert not threat.is_speeding(scene, _cfg())


def test_speeding_no_posted_limit():
    vs = VehicleState(speed_mps=30.0, posted_speed_limit_mps=None)
    scene = Scene(vehicle_state=vs)
    assert not threat.is_speeding(scene, _cfg())


def test_speeding_no_ego_speed():
    vs = VehicleState(speed_mps=None, posted_speed_limit_mps=13.4)
    scene = Scene(vehicle_state=vs)
    assert not threat.is_speeding(scene, _cfg())


# --- stop_sign_present ---

def test_stop_sign_detected():
    stop = _track(cls=ObjectClass.STOP_SIGN, distance_m=15.0, closing_speed_mps=0.0, track_id=5)
    scene = Scene(tracks=(stop,), vehicle_state=VehicleState())
    assert threat.stop_sign_present(scene)


def test_stop_sign_absent():
    scene = _scene()
    assert not threat.stop_sign_present(scene)


# --- is_slow_lead ---

def test_slow_lead_detected():
    # ego=15 m/s, posted=13.4, closing=12 → lead_speed=3 m/s < 0.6*13.4=8.04
    scene = _scene(ego_speed_mps=15.0, posted_limit_mps=13.4, lead_closing_mps=12.0)
    assert threat.is_slow_lead(scene, _cfg())


def test_slow_lead_normal_traffic():
    # ego=15, closing=2 → lead_speed=13 > 0.6*13.4=8.04
    scene = _scene(ego_speed_mps=15.0, posted_limit_mps=13.4, lead_closing_mps=2.0)
    assert not threat.is_slow_lead(scene, _cfg())


# --- idm_required_accel ---

def test_idm_free_road_below_desired_speed_accelerates():
    scene = _scene(no_lead=True, ego_speed_mps=10.0, posted_limit_mps=27.0)
    accel = threat.idm_required_accel(scene, _cfg())
    assert accel is not None
    assert 0.0 < accel <= 1.5  # bounded by idm_max_accel_mps2


def test_idm_at_desired_speed_near_zero():
    scene = _scene(no_lead=True, ego_speed_mps=27.0, posted_limit_mps=27.0)
    accel = threat.idm_required_accel(scene, _cfg())
    assert accel == pytest.approx(0.0, abs=0.01)


def test_idm_over_desired_speed_decelerates():
    scene = _scene(no_lead=True, ego_speed_mps=35.0, posted_limit_mps=27.0)
    accel = threat.idm_required_accel(scene, _cfg())
    assert accel < 0.0


def test_idm_close_fast_lead_demands_hard_braking():
    # v=25, gap 10 m, closing 8 m/s → interaction term dominates.
    scene = _scene(lead_distance_m=10.0, lead_closing_mps=8.0,
                   ego_speed_mps=25.0, posted_limit_mps=27.0)
    accel = threat.idm_required_accel(scene, _cfg())
    assert accel < -5.0


def test_idm_moderate_lead_lands_in_advise_band():
    # v=20, gap 30 m, closing 5 m/s → uncomfortable but not emergency.
    scene = _scene(lead_distance_m=30.0, lead_closing_mps=5.0,
                   ego_speed_mps=20.0, posted_limit_mps=27.0)
    accel = threat.idm_required_accel(scene, _cfg())
    assert -5.0 <= accel <= -3.0


def test_idm_tiny_gap_no_division_blowup():
    scene = _scene(lead_distance_m=0.1, lead_closing_mps=5.0,
                   ego_speed_mps=20.0, posted_limit_mps=27.0)
    accel = threat.idm_required_accel(scene, _cfg())
    assert accel == -10.0  # clamped at ~1 g physical braking limit


def test_idm_none_when_ego_speed_unknown():
    vs = VehicleState(speed_mps=None)
    scene = Scene(vehicle_state=vs)
    assert threat.idm_required_accel(scene, _cfg()) is None


def test_idm_v0_falls_back_to_desired_speed():
    # No posted limit → v0 = cfg.desired_speed_mps (27); at 27 → ~0.
    scene = _scene(no_lead=True, ego_speed_mps=27.0, posted_limit_mps=None)
    accel = threat.idm_required_accel(scene, _cfg())
    assert accel == pytest.approx(0.0, abs=0.01)


def test_idm_lead_without_distance_uses_free_term_only():
    scene = _scene(lead_distance_m=None, lead_closing_mps=None,
                   ego_speed_mps=10.0, posted_limit_mps=27.0)
    accel = threat.idm_required_accel(scene, _cfg())
    assert accel is not None
    assert accel > 0.0


# --- brake bands ---

def test_brake_advised_in_band():
    scene = Scene(required_accel_mps2=-4.0)
    assert threat.is_brake_advised(scene, _cfg())
    assert not threat.is_brake_hard(scene, _cfg())


def test_brake_hard_below_band():
    scene = Scene(required_accel_mps2=-6.0)
    assert threat.is_brake_hard(scene, _cfg())
    assert not threat.is_brake_advised(scene, _cfg())  # hard excludes advised


def test_no_brake_event_in_comfort_zone():
    scene = Scene(required_accel_mps2=-2.0)
    assert not threat.is_brake_advised(scene, _cfg())
    assert not threat.is_brake_hard(scene, _cfg())


def test_brake_bands_recompute_when_accel_missing():
    # Scene without precomputed accel → predicates fall back to IDM.
    scene = _scene(lead_distance_m=10.0, lead_closing_mps=8.0,
                   ego_speed_mps=25.0, posted_limit_mps=27.0)
    assert scene.required_accel_mps2 is None
    assert threat.is_brake_hard(scene, _cfg())
