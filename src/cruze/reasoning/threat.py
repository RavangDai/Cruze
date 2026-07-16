"""
Threat assessment — pure functions, no side effects.

All functions accept explicit values rather than the full Scene so they
remain individually unit-testable without constructing scene graphs.

Physical reference:
  - Time-to-collision (TTC): classic kinematic model, t = d / Δv
  - Following gap (headway): t = d / v_ego; 2 s is the standard safety margin
  - Forward Collision Warning threshold: TTC < 3 s is the NHTSA benchmark
    for low-speed FCW systems; we use a configurable threshold.
  - IDM (Intelligent Driver Model, Treiber 2000): desired acceleration from
    ego speed, desired speed, and the gap/closing-rate to the lead — the
    approach VisionPilot's longitudinal planner uses. Braking bands on its
    output give physically meaningful warning levels.

Units: SI throughout (metres, m/s, seconds).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from cruze.core.types import ObjectClass, Scene, Track

if TYPE_CHECKING:
    from cruze.core.config import ReasoningConfig

# Braking on dry asphalt tops out around 1 g; demands beyond that carry no
# extra information, so IDM output is clamped here.
_IDM_ACCEL_FLOOR_MPS2 = -10.0
# Gap floor for the IDM interaction term — below half a metre the (s*/s)²
# ratio diverges without adding meaning (VisionPilot uses the same floor).
_IDM_MIN_GAP_FLOOR_M = 0.5


def time_to_collision(lead_distance_m: float, closing_speed_mps: float) -> float:
    """
    Compute time-to-collision in seconds.

    Parameters
    ----------
    lead_distance_m:
        Distance to the lead vehicle in metres (must be > 0).
    closing_speed_mps:
        Rate at which the gap is closing in m/s.
        Positive = closing (danger), negative = opening (safe).

    Returns
    -------
    float
        TTC in seconds, or math.inf if not closing or invalid inputs.
    """
    if lead_distance_m <= 0 or closing_speed_mps <= 0:
        return math.inf
    return lead_distance_m / closing_speed_mps


def following_gap_seconds(lead_distance_m: float, ego_speed_mps: float) -> float:
    """
    Compute following gap in seconds (headway = distance / ego speed).

    Returns math.inf when ego is stationary (no collision risk from following).
    """
    if ego_speed_mps <= 0 or lead_distance_m <= 0:
        return math.inf
    return lead_distance_m / ego_speed_mps


def is_forward_collision_warning(scene: Scene, cfg: "ReasoningConfig") -> bool:
    """
    True when TTC to the lead vehicle is below the configured threshold AND
    the ego vehicle is actually moving (stationary ego never triggers FCW).
    """
    if scene.lead_track is None:
        return False
    if scene.vehicle_state.speed_mps is None or scene.vehicle_state.speed_mps < 0.5:
        return False
    if scene.lead_track.distance_m is None or scene.lead_track.closing_speed_mps is None:
        return False
    ttc = time_to_collision(scene.lead_track.distance_m, scene.lead_track.closing_speed_mps)
    return ttc < cfg.fcw_ttc_threshold_s


def is_tailgating(scene: Scene, cfg: "ReasoningConfig") -> bool:
    """
    True when the following gap to the lead vehicle is below 2 seconds.
    Only fires when ego is moving above walking speed (1 m/s).
    """
    if scene.lead_track is None or scene.lead_track.distance_m is None:
        return False
    ego_speed = scene.vehicle_state.speed_mps
    if ego_speed is None or ego_speed < 1.0:
        return False
    gap = following_gap_seconds(scene.lead_track.distance_m, ego_speed)
    return gap < cfg.tailgating_gap_threshold_s


def is_speeding(scene: Scene, cfg: "ReasoningConfig") -> bool:
    """
    True when ego speed exceeds the posted limit by more than the grace margin.
    Returns False when posted limit is unknown (no GPS or no map data).
    """
    ego_speed = scene.vehicle_state.speed_mps
    posted = scene.vehicle_state.posted_speed_limit_mps
    if ego_speed is None or posted is None:
        return False
    return ego_speed > posted + cfg.speeding_grace_mps


def is_slow_lead(scene: Scene, cfg: "ReasoningConfig") -> bool:
    """
    True when the lead vehicle is travelling well below the posted speed limit.
    Useful for suggesting a lane change or alerting to obstruction ahead.
    Threshold: lead speed < 60% of posted limit (not yet implemented in tracks
    but derived from lead closing speed + ego speed).
    """
    if scene.lead_track is None or scene.vehicle_state.speed_mps is None:
        return False
    posted = scene.vehicle_state.posted_speed_limit_mps
    if posted is None or posted < 1.0:
        return False
    # Estimate lead speed = ego_speed - closing_speed (positive closing = lead slower).
    closing = scene.lead_track.closing_speed_mps
    if closing is None:
        return False
    lead_speed = scene.vehicle_state.speed_mps - closing
    return lead_speed < 0.6 * posted and lead_speed >= 0


def stop_sign_present(scene: Scene) -> bool:
    """True when a stop sign is among the tracked objects."""
    return any(t.cls == ObjectClass.STOP_SIGN for t in scene.tracks)


def idm_required_accel(scene: Scene, cfg: "ReasoningConfig") -> float | None:
    """
    Intelligent Driver Model: the longitudinal acceleration a rational driver
    would apply right now.  Negative = braking needed.

        s*   = s0 + max(0, v·T + v·Δv / (2·√(a·b)))
        accel = a · (1 − (v/v0)^δ − (s*/s)²)

    where v = ego speed, Δv = closing speed (positive = approaching lead),
    s = gap to lead, v0 = desired speed (posted limit, else config fallback).
    Returns None when ego speed is unknown. Output clamped to
    [_IDM_ACCEL_FLOOR_MPS2, a].
    """
    v = scene.vehicle_state.speed_mps
    if v is None:
        return None

    a = cfg.idm_max_accel_mps2
    b = cfg.idm_comfort_decel_mps2
    v0 = scene.vehicle_state.posted_speed_limit_mps or cfg.desired_speed_mps
    v0 = max(v0, 0.1)  # guard: a zero desired speed would divide by zero

    accel = a * (1.0 - (v / v0) ** cfg.idm_delta)

    lead = scene.lead_track
    if (
        lead is not None
        and lead.distance_m is not None
        and lead.closing_speed_mps is not None
    ):
        delta_v = lead.closing_speed_mps  # already ego − lead by contract
        s_star = cfg.idm_min_gap_m + max(
            0.0, v * cfg.idm_headway_s + v * delta_v / (2.0 * math.sqrt(a * b))
        )
        s = max(lead.distance_m, _IDM_MIN_GAP_FLOOR_M)
        accel -= a * (s_star / s) ** 2

    return max(_IDM_ACCEL_FLOOR_MPS2, min(accel, a))


def _scene_accel(scene: Scene, cfg: "ReasoningConfig") -> float | None:
    """Prefer the SceneAssembler-precomputed value; recompute otherwise."""
    if scene.required_accel_mps2 is not None:
        return scene.required_accel_mps2
    return idm_required_accel(scene, cfg)


def is_brake_advised(scene: Scene, cfg: "ReasoningConfig") -> bool:
    """True when IDM demands more than comfortable braking but short of the
    emergency band — the driver should start slowing down."""
    accel = _scene_accel(scene, cfg)
    if accel is None:
        return False
    return -cfg.idm_hard_decel_mps2 <= accel <= -cfg.idm_advise_decel_mps2


def is_brake_hard(scene: Scene, cfg: "ReasoningConfig") -> bool:
    """True when IDM demands emergency-level deceleration (≈0.5 g+)."""
    accel = _scene_accel(scene, cfg)
    if accel is None:
        return False
    return accel < -cfg.idm_hard_decel_mps2


def is_cut_in(
    prev_lead: Track | None, lead: Track | None, cfg: "ReasoningConfig"
) -> bool:
    """
    True when the lead vehicle changed to a meaningfully closer track — a
    vehicle from the adjacent lane merged into the gap. The margin clears
    monocular depth noise (~7% of range) so lead-swap jitter doesn't fire;
    tracker ID churn on the same physical car doesn't either, because the
    distance barely changes.
    """
    if prev_lead is None or lead is None:
        return False
    if lead.track_id == prev_lead.track_id:
        return False
    if lead.distance_m is None or prev_lead.distance_m is None:
        return False
    return lead.distance_m < prev_lead.distance_m - cfg.cut_in_margin_m


def is_lane_departure(scene: Scene, cfg: "ReasoningConfig") -> bool:
    """
    True when ego has drifted more than the CTE threshold from lane centre
    while moving at road speed. Lane freshness is already enforced by the
    SceneAssembler (stale lanes never reach the Scene), so presence implies
    a current measurement.
    """
    if scene.lanes is None or scene.lanes.cte_m is None:
        return False
    speed = scene.vehicle_state.speed_mps
    if speed is None or speed <= cfg.ldw_min_speed_mps:
        return False  # below ~18 mph, large offsets are parking manoeuvres
    return abs(scene.lanes.cte_m) > cfg.ldw_cte_threshold_m
