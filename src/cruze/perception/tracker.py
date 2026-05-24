"""
SORT-style multi-object tracker.

Algorithm:
  1. For each new frame, compute IoU between every existing track and every
     new detection (same class only — a car bbox shouldn't match a person).
  2. Solve the assignment problem to maximise total IoU:
     - Use scipy.optimize.linear_sum_assignment (Hungarian) when available.
     - Fall back to greedy highest-IoU-first matching otherwise.
  3. Matched tracks → update bbox, reset missed counter, update distance.
  4. Unmatched detections → create new track with fresh ID.
  5. Unmatched tracks → increment missed counter; drop when > max_age.
  6. Estimate closing speed: Δdistance / Δtime using an exponential moving
     average to smooth per-frame noise.

Track IDs are monotonically increasing integers. They wrap at 2^31 (won't
happen in practice).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from cruze.core.types import BBox, Detection, ObjectClass, Track

logger = logging.getLogger(__name__)

# Exponential moving average alpha for closing speed smoothing.
# Smaller = more smoothing, more lag. 0.3 is a reasonable default.
_EMA_ALPHA = 0.3


@dataclass
class _TrackState:
    """Mutable internal state for a single tracked object."""

    track_id: int
    bbox: BBox
    cls: ObjectClass
    distance_m: float | None = None
    closing_speed_mps: float | None = None
    age_missed: int = 0
    last_updated: float = field(default_factory=time.monotonic)

    def to_track(self) -> Track:
        return Track(
            track_id=self.track_id,
            bbox=self.bbox,
            cls=self.cls,
            distance_m=self.distance_m,
            closing_speed_mps=self.closing_speed_mps,
            age_missed=self.age_missed,
            timestamp=self.last_updated,
        )


def _hungarian_assign(cost_matrix: list[list[float]]) -> list[tuple[int, int]]:
    """
    Solve assignment with Hungarian algorithm (scipy) or greedy fallback.
    Returns list of (row, col) matched pairs.
    cost_matrix[i][j] = cost of assigning track i to detection j.
    """
    if not cost_matrix or not cost_matrix[0]:
        return []

    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment  # type: ignore

        mat = np.array(cost_matrix)
        row_ind, col_ind = linear_sum_assignment(mat)
        return list(zip(row_ind.tolist(), col_ind.tolist()))

    except ImportError:
        # Greedy: repeatedly pick the minimum-cost (row, col) pair.
        pairs: list[tuple[int, int]] = []
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        # Flatten and sort by cost ascending.
        candidates = sorted(
            ((cost_matrix[r][c], r, c)
             for r in range(len(cost_matrix))
             for c in range(len(cost_matrix[0]))),
            key=lambda x: x[0],
        )
        for cost, r, c in candidates:
            if r not in used_rows and c not in used_cols:
                pairs.append((r, c))
                used_rows.add(r)
                used_cols.add(c)
        return pairs


class Tracker:
    """
    Multi-object tracker.

    Parameters
    ----------
    iou_threshold:
        Minimum IoU for a detection to be associated with a track.
    max_age:
        Frames a track can go unmatched before being deleted.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 5) -> None:
        self._iou_threshold = iou_threshold
        self._max_age = max_age
        self._tracks: list[_TrackState] = []
        self._next_id: int = 1

    def update(
        self,
        detections: list[Detection],
        timestamp: float | None = None,
    ) -> list[Track]:
        """
        Ingest a new set of detections and return the current live tracks.
        Call once per frame in frame-timestamp order.

        Parameters
        ----------
        timestamp:
            Frame timestamp in seconds (monotonic). Defaults to time.monotonic().
            Pass an explicit value in tests to make closing-speed math deterministic.
        """
        now = timestamp if timestamp is not None else time.monotonic()

        if not self._tracks:
            # Bootstrap: every detection becomes a new track.
            for det in detections:
                self._tracks.append(self._new_track(det, now))
            return [t.to_track() for t in self._tracks]

        # --- Build IoU cost matrix (tracks × detections, same class only) ---
        n_tracks = len(self._tracks)
        n_dets = len(detections)
        INF = 1e9  # high cost = no match

        cost = [[INF] * n_dets for _ in range(n_tracks)]
        for i, trk in enumerate(self._tracks):
            for j, det in enumerate(detections):
                if det.cls != trk.cls:
                    continue
                iou = trk.bbox.iou(det.bbox)
                if iou >= self._iou_threshold:
                    # Cost = 1 - IoU so that linear_sum_assignment minimises.
                    cost[i][j] = 1.0 - iou

        # --- Solve assignment ---
        assignments = _hungarian_assign(cost)
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()

        for trk_idx, det_idx in assignments:
            if cost[trk_idx][det_idx] >= INF:
                continue
            trk = self._tracks[trk_idx]
            det = detections[det_idx]
            self._update_track(trk, det, now)
            matched_tracks.add(trk_idx)
            matched_dets.add(det_idx)

        # --- Unmatched detections → new tracks ---
        for j, det in enumerate(detections):
            if j not in matched_dets:
                self._tracks.append(self._new_track(det, now))

        # --- Unmatched tracks → age out ---
        survivors: list[_TrackState] = []
        for i, trk in enumerate(self._tracks):
            if i not in matched_tracks and i < n_tracks:
                trk.age_missed += 1
            if trk.age_missed <= self._max_age:
                survivors.append(trk)
            else:
                logger.debug("Track %d aged out after %d missed frames", trk.track_id, trk.age_missed)
        self._tracks = survivors

        return [t.to_track() for t in self._tracks]

    def _new_track(self, det: Detection, now: float) -> _TrackState:
        tid = self._next_id
        self._next_id += 1
        return _TrackState(
            track_id=tid,
            bbox=det.bbox,
            cls=det.cls,
            distance_m=det.distance_m,
            last_updated=now,
        )

    def _update_track(self, trk: _TrackState, det: Detection, now: float) -> None:
        dt = now - trk.last_updated
        trk.bbox = det.bbox
        trk.age_missed = 0
        trk.last_updated = now

        # Update closing speed via EMA.
        if det.distance_m is not None and trk.distance_m is not None and dt > 0:
            raw_closing = (trk.distance_m - det.distance_m) / dt
            if trk.closing_speed_mps is None:
                trk.closing_speed_mps = raw_closing
            else:
                trk.closing_speed_mps = (
                    _EMA_ALPHA * raw_closing + (1 - _EMA_ALPHA) * trk.closing_speed_mps
                )

        trk.distance_m = det.distance_m

    @property
    def active_track_count(self) -> int:
        return len(self._tracks)

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
