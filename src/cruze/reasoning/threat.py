"""
Threat assessment — pure functions, no side effects.

All functions accept explicit values rather than the full Scene so they
remain individually unit-testable without constructing scene graphs.

Physical reference:
  - Time-to-collision (TTC): classic kinematic model, t = d / Δv
  - Following gap (headway): t = d / v_ego; 2 s is the standard safety margin
  - Forward Collision Warning threshold: TTC < 3 s is the NHTSA benchmark
    for low-speed FCW systems; we use a configurable threshold.

Units: SI throughout (metres, m/s, seconds).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from cruze.core.types import ObjectClass, Scene

if TYPE_CHECKING:
    from cruze.core.config import ReasoningConfig


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
