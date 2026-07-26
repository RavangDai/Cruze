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
  6. Estimate distance + closing speed with a per-track constant-velocity
     Kalman filter — rejects single-frame depth spikes, converges in a few
     frames (replaces the earlier EMA smoothing).

Track IDs are monotonically increasing integers. They wrap at 2^31 (won't
happen in practice).
"""

from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass, field

from cruze.core.types import BBox, Detection, ObjectClass, Track, TrafficLightState
from cruze.perception import depth

logger = logging.getLogger(__name__)

# Consecutive UNKNOWN light readings tolerated before a track's held state
# reverts to UNKNOWN: ~0.5 s at 30 fps — one lamp blink/occlusion cycle plus
# margin, so a briefly blocked light doesn't flicker in the HUD.
_LIGHT_UNKNOWN_HOLD_FRAMES = 15

# Box-filter noise, in pixels, tuned against detector jitter measured at this
# reference width; scaled to whatever the frame actually is.
_NOISE_REFERENCE_WIDTH_PX = 1920.0
# Measured AutoSpeed frame-to-frame centre jitter on a matched track was
# ~14.7 px combined across both axes, so ~7 px per axis; 6 is a slightly
# optimistic σ that keeps the filter responsive.
_BOX_MEAS_STD_PX = 6.0
# Process noise driver for the box centre. Perspective makes image-space
# acceleration large for near objects, but tracking that fully would pass the
# detector's jitter straight through. Tuned on real footage against trajectory
# curvature (second difference of the centre, which isolates jitter from the
# genuine motion that dominates raw step size): the unfiltered detector shows a
# median |d2x| of 31.8 px, this brings it to 4.4 px. Below ~40 the gain is
# marginal and the estimate starts lagging real manoeuvres.
_BOX_ACCEL_STD_PX = 150.0
# Box size changes far more slowly than position (it only grows as the object
# nears), so size gets a tighter process noise and is smoothed harder.
_SIZE_ACCEL_STD_PX = 120.0


@dataclass(frozen=True)
class CameraGeometry:
    """Intrinsics the tracker needs to turn a filtered box into metres."""

    image_width: int = 0
    focal_px: float | None = None
    camera_height_m: float = 1.2
    horizon_y: float | None = None

    @property
    def usable(self) -> bool:
        return bool(self.image_width > 0 and self.focal_px and self.horizon_y is not None)


class _ScalarCV:
    """Constant-velocity scalar Kalman filter over (value, d value/dt).

    Same structure as _DistanceKalman below, but with noise supplied by the
    caller so one implementation serves both box geometry and range.
    """

    def __init__(self, x: float, meas_std: float, accel_std: float) -> None:
        self.x = x
        self.v = 0.0
        self._r = meas_std ** 2
        self._q = accel_std ** 2
        # Position trusted to the measurement; velocity unknown until a second
        # observation arrives, so seed it wide.
        self.p11 = self._r
        self.p12 = 0.0
        self.p22 = (4.0 * meas_std) ** 2

    def predict(self, dt: float) -> None:
        self.x += self.v * dt
        q = self._q
        p11 = self.p11 + 2.0 * dt * self.p12 + dt * dt * self.p22 + q * dt ** 4 / 4.0
        p12 = self.p12 + dt * self.p22 + q * dt ** 3 / 2.0
        p22 = self.p22 + q * dt ** 2
        self.p11, self.p12, self.p22 = p11, p12, p22

    def correct(self, z: float) -> None:
        s = self.p11 + self._r
        k1 = self.p11 / s
        k2 = self.p12 / s
        innovation = z - self.x
        self.x += k1 * innovation
        self.v += k2 * innovation
        p11, p12, p22 = self.p11, self.p12, self.p22
        self.p11 = (1.0 - k1) * p11
        self.p12 = (1.0 - k1) * p12
        self.p22 = p22 - k2 * p12


class _BoxKalman:
    """Constant-velocity filter over a box as (cx, cy, w, h).

    The detector re-localises every box from scratch each frame, so its output
    carries several pixels of jitter that used to reach the screen unfiltered —
    and, worse, reached the depth estimator, where a jittering box bottom near
    the horizon swings range by tens of metres. Filtering here fixes both at the
    source.

    Tracking the centre and size rather than the corners keeps the two kinds of
    motion independent: an approaching vehicle grows without its centre moving,
    and a crossing one moves without changing size.
    """

    def __init__(self, bbox: BBox, scale: float = 1.0) -> None:
        meas = _BOX_MEAS_STD_PX * scale
        self._cx = _ScalarCV(bbox.cx, meas, _BOX_ACCEL_STD_PX * scale)
        self._cy = _ScalarCV(bbox.cy, meas, _BOX_ACCEL_STD_PX * scale)
        self._w = _ScalarCV(bbox.width, meas, _SIZE_ACCEL_STD_PX * scale)
        self._h = _ScalarCV(bbox.height, meas, _SIZE_ACCEL_STD_PX * scale)

    def predict(self, dt: float) -> None:
        if dt <= 0.0:
            return
        for f in (self._cx, self._cy, self._w, self._h):
            f.predict(dt)

    def correct(self, bbox: BBox) -> None:
        self._cx.correct(bbox.cx)
        self._cy.correct(bbox.cy)
        self._w.correct(bbox.width)
        self._h.correct(bbox.height)

    @property
    def bbox(self) -> BBox:
        # Degenerate boxes break IoU and depth; a box narrower than a pixel is
        # a filter artefact, not an object.
        w = max(1.0, self._w.x)
        h = max(1.0, self._h.x)
        return BBox(self._cx.x - w / 2, self._cy.x - h / 2,
                    self._cx.x + w / 2, self._cy.x + h / 2)


class _DistanceKalman:
    """
    Per-track 1-D constant-velocity Kalman filter over (distance, d_dist/dt).

    Replaces EMA smoothing: rejects single-frame monocular-depth spikes while
    converging to a constant closing speed within a few observations.
    """

    # Process noise driver: relative acceleration between ego and target can
    # reach ~3 m/s² under hard braking.
    _ACCEL_NOISE = 3.0
    # Measurement σ as a fraction of distance — monocular depth error grows
    # roughly linearly with range (~7 % with ground-plane estimation).
    _MEAS_FRAC = 0.07

    def __init__(self, distance_m: float) -> None:
        self.d = distance_m
        self.v = 0.0  # d(distance)/dt; negative = approaching
        # Initial covariance: distance fairly trusted (σ≈2 m), velocity unknown (σ≈5 m/s).
        self.p11, self.p12, self.p22 = 4.0, 0.0, 25.0
        self.updates = 0

    @property
    def ready(self) -> bool:
        """True after at least one step() call — i.e. two distance
        measurements total, since the first one seeds __init__."""
        return self.updates >= 1

    def step(self, measured_m: float, dt: float) -> None:
        # --- Predict (constant velocity model) ---
        d = self.d + self.v * dt
        v = self.v
        q = self._ACCEL_NOISE ** 2
        q11 = q * dt ** 4 / 4.0
        q12 = q * dt ** 3 / 2.0
        q22 = q * dt ** 2
        p11 = self.p11 + 2.0 * dt * self.p12 + dt * dt * self.p22 + q11
        p12 = self.p12 + dt * self.p22 + q12
        p22 = self.p22 + q22
        # --- Update with the measured distance (H = [1, 0]) ---
        # Floor R at (0.1 m)²: a measurement of ~0 m would otherwise give
        # r=0 → gain 1 → covariance collapses to 0 → divide-by-zero next step.
        r = max((self._MEAS_FRAC * measured_m) ** 2, 0.01)
        s = p11 + r
        k1 = p11 / s
        k2 = p12 / s
        innovation = measured_m - d
        self.d = d + k1 * innovation
        self.v = v + k2 * innovation
        self.p11 = (1.0 - k1) * p11
        self.p12 = (1.0 - k1) * p12
        # `p12` here is the predicted value (local above), not self.p12.
        self.p22 = p22 - k2 * p12
        self.updates += 1


@dataclass
class _TrackState:
    """Mutable internal state for a single tracked object."""

    track_id: int
    box: _BoxKalman
    cls: ObjectClass
    kalman: _DistanceKalman | None = None
    age_missed: int = 0
    # Last time a detection matched — drives age-out and the distance filter.
    last_updated: float = field(default_factory=time.monotonic)
    # Last time the box filter was advanced. Distinct from last_updated because
    # predict() mutates state: deriving the step from last_updated would make a
    # coasting track re-predict from the same origin with a growing dt, and it
    # would compound its own extrapolation until the box flew off the object.
    last_predicted: float = field(default_factory=time.monotonic)
    mask_xy: tuple[tuple[float, float], ...] | None = None
    light_state: TrafficLightState | None = None
    # Consecutive UNKNOWN readings while holding a confident light state.
    light_unknown_count: int = 0

    @property
    def bbox(self) -> BBox:
        """Filtered box. Consumers never see the raw detection — an unmatched
        track has been predicted forward, so it keeps following its object
        instead of freezing on the road behind it."""
        return self.box.bbox

    def to_track(self) -> Track:
        return Track(
            track_id=self.track_id,
            bbox=self.bbox,
            cls=self.cls,
            distance_m=self.kalman.d if self.kalman is not None else None,
            closing_speed_mps=(
                -self.kalman.v if self.kalman is not None and self.kalman.ready else None
            ),
            age_missed=self.age_missed,
            timestamp=self.last_updated,
            mask_xy=self.mask_xy,
            light_state=self.light_state,
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

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 5,
                 max_age_s: float = 0.4) -> None:
        self._iou_threshold = iou_threshold
        self._max_age = max_age
        self._max_age_s = max_age_s
        self._tracks: list[_TrackState] = []
        self._next_id: int = 1
        # Per-frame context set by update(); used when emitting Tracks.
        self._frame_id: int = 0
        self._geometry = CameraGeometry()

    def _emit(self) -> list[Track]:
        """Current tracks as immutable Tracks, stamped with the frame being
        processed and their ground-plane position.

        The whole list belongs to this frame even where an individual track
        went unmatched, so every track carries the same frame_id — that is what
        lets the dashboard pair the overlay with the right JPEG. Position is
        derived from the Kalman-filtered distance rather than the raw
        measurement, so it stays consistent with Track.distance_m.
        """
        geo = self._geometry
        out: list[Track] = []
        for state in self._tracks:
            track = dataclasses.replace(state.to_track(), frame_id=self._frame_id)
            if geo.usable:
                track = dataclasses.replace(
                    track,
                    ground_xz_m=depth.ground_position_xz(
                        track.bbox, track.distance_m, geo.focal_px, geo.image_width
                    ),
                )
            out.append(track)
        return out

    def _measure_distance(self, bbox: BBox, cls: ObjectClass, fallback: float | None) -> float | None:
        """Range from the *filtered* box.

        The caller's per-detection estimate comes off the raw box, whose bottom
        edge jitters by several pixels. Near the horizon dZ/dy is enormous, so
        that jitter reached range as swings of tens of metres. Re-measuring from
        the filtered box removes the swing at its source; without geometry we
        fall back to whatever the caller computed.
        """
        geo = self._geometry
        if not geo.usable:
            return fallback
        return depth.estimate_distance_fused(
            bbox, cls, geo.focal_px, geo.camera_height_m, geo.horizon_y
        )

    def update(
        self,
        detections: list[Detection],
        timestamp: float | None = None,
        frame_id: int = 0,
        geometry: CameraGeometry | None = None,
    ) -> list[Track]:
        """
        Ingest a new set of detections and return the current live tracks.
        Call once per frame in frame-timestamp order.

        Parameters
        ----------
        timestamp:
            Frame timestamp in seconds (monotonic). Defaults to time.monotonic().
            Pass an explicit value in tests to make closing-speed math deterministic.
        frame_id:
            Camera frame these detections came from. Stamped onto every emitted
            track so the dashboard can pin its overlay to the matching JPEG.
        geometry:
            Camera intrinsics. Without them boxes are still filtered, but range
            falls back to the caller's per-detection estimate and
            `Track.ground_xz_m` stays None.
        """
        now = timestamp if timestamp is not None else time.monotonic()
        self._frame_id = frame_id
        self._geometry = geometry if geometry is not None else CameraGeometry()

        # Predict every track forward before matching, so association compares
        # detections against where each object should be now rather than where
        # it was last seen. This is what lets a track survive fast motion.
        for trk in self._tracks:
            trk.box.predict(max(0.0, now - trk.last_predicted))
            trk.last_predicted = now

        if not self._tracks:
            # Bootstrap: every detection becomes a new track.
            for det in detections:
                self._tracks.append(self._new_track(det, now))
            return self._emit()

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
        # Two caps, whichever bites first. The frame count bounds how many
        # consecutive misses are tolerated; the wall-clock cap bounds how long a
        # box may coast regardless of rate. Without the latter, `max_age` frames
        # means 170 ms at 30 fps but a full second at 5 Hz — the same setting
        # silently leaves stale boxes on the road for six times longer when the
        # pipeline slows down.
        survivors: list[_TrackState] = []
        for i, trk in enumerate(self._tracks):
            if i not in matched_tracks and i < n_tracks:
                trk.age_missed += 1
            coasted_s = now - trk.last_updated
            if trk.age_missed <= self._max_age and coasted_s <= self._max_age_s:
                survivors.append(trk)
            else:
                logger.debug("Track %d aged out after %d missed frames (%.2f s)",
                             trk.track_id, trk.age_missed, coasted_s)
        self._tracks = survivors

        return self._emit()

    def _box_scale(self) -> float:
        """Filter noise is tuned in pixels at a reference width; scale it to the
        frame in hand so behaviour is the same at any capture resolution."""
        width = self._geometry.image_width
        return width / _NOISE_REFERENCE_WIDTH_PX if width > 0 else 1.0

    def _new_track(self, det: Detection, now: float) -> _TrackState:
        tid = self._next_id
        self._next_id += 1
        box = _BoxKalman(det.bbox, self._box_scale())
        distance = self._measure_distance(box.bbox, det.cls, det.distance_m)
        return _TrackState(
            track_id=tid,
            box=box,
            cls=det.cls,
            kalman=_DistanceKalman(distance) if distance is not None else None,
            last_updated=now,
            last_predicted=now,
            mask_xy=det.mask_xy,
            light_state=det.light_state,
        )

    def _update_track(self, trk: _TrackState, det: Detection, now: float) -> None:
        dt = now - trk.last_updated
        # The track was already predicted to `now` at the top of update(), so
        # this only folds in the measurement.
        trk.box.correct(det.bbox)
        trk.age_missed = 0
        trk.last_updated = now
        # Masks are per-frame geometry: always take the fresh outline, even if
        # None — a stale outline on a moved object is worse than no outline.
        trk.mask_xy = det.mask_xy
        self._update_light_state(trk, det)

        distance = self._measure_distance(trk.bbox, trk.cls, det.distance_m)
        if distance is None:
            return  # keep last filtered state; no measurement this frame
        if trk.kalman is None:
            trk.kalman = _DistanceKalman(distance)
        elif dt > 0:
            trk.kalman.step(distance, dt)

    @staticmethod
    def _update_light_state(trk: _TrackState, det: Detection) -> None:
        """Debounce lamp readings: hold the last confident state through brief
        UNKNOWN gaps (occlusion, blur, LED blink) instead of flickering."""
        if det.light_state is None:
            return  # not a traffic light — leave None
        if det.light_state is not TrafficLightState.UNKNOWN:
            trk.light_state = det.light_state
            trk.light_unknown_count = 0
            return
        trk.light_unknown_count += 1
        if trk.light_unknown_count > _LIGHT_UNKNOWN_HOLD_FRAMES or trk.light_state is None:
            trk.light_state = TrafficLightState.UNKNOWN

    @property
    def active_track_count(self) -> int:
        return len(self._tracks)

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
