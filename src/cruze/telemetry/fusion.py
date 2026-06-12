"""
Speed-source fusion — pure functions, no I/O, fully unit-tested.

Priority: OBD wheel speed > GPS Doppler (RMC speed-over-ground) >
GPS position-derived (haversine Δposition/Δtime). A source is used only if
its sample is fresher than STALENESS_S.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

EARTH_RADIUS_M = 6_371_000.0

# A source older than this is considered dead and skipped (GPS modules emit
# at ≥1 Hz, OBD at ~10 Hz — 2 s of silence means the source dropped out).
STALENESS_S = 2.0

# Fixes implying faster than ~200 mph are receiver glitches, not driving.
MAX_PLAUSIBLE_SPEED_MPS = 90.0


@dataclass(frozen=True)
class SpeedSample:
    speed_mps: float
    timestamp: float  # time.monotonic() at receipt


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two WGS-84 points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def position_speed_mps(
    lat1: float, lon1: float, t1: float,
    lat2: float, lon2: float, t2: float,
) -> float | None:
    """
    Speed implied by two consecutive position fixes, or None when the pair is
    unusable (non-positive Δt, or implausibly fast → GPS glitch).
    """
    dt = t2 - t1
    if dt <= 0.0:
        return None
    speed = haversine_m(lat1, lon1, lat2, lon2) / dt
    if speed > MAX_PLAUSIBLE_SPEED_MPS:
        return None
    return speed


def fuse_speed(
    samples: Sequence[tuple[SpeedSample | None, str]],
    now: float,
    staleness_s: float = STALENESS_S,
) -> tuple[float | None, str]:
    """
    Pick the highest-priority fresh sample.

    samples: (sample, source_label) pairs in priority order.
    Returns (speed_mps, source) — (None, "unknown") when nothing is fresh.
    """
    for sample, source in samples:
        if sample is not None and now - sample.timestamp <= staleness_s:
            return sample.speed_mps, source
    return None, "unknown"
