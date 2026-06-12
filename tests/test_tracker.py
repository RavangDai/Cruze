"""
Tracker unit tests — no ML deps; pure geometry.
"""

import pytest

from cruze.core.types import BBox, Detection, ObjectClass
from cruze.perception.tracker import Tracker


def _det(x1, y1, x2, y2, cls=ObjectClass.CAR, conf=0.9, dist=None):
    return Detection(BBox(x1, y1, x2, y2), conf, cls, distance_m=dist)


# --- ID stability ---

def test_stable_id_across_frames():
    tracker = Tracker(iou_threshold=0.3, max_age=3)
    dets = [_det(10, 10, 50, 50)]
    tracks1 = tracker.update(dets)
    tracks2 = tracker.update(dets)
    assert tracks1[0].track_id == tracks2[0].track_id


def test_new_id_for_new_object():
    tracker = Tracker(iou_threshold=0.3, max_age=3)
    tracks1 = tracker.update([_det(10, 10, 50, 50)])
    tracks2 = tracker.update([_det(200, 200, 250, 250)])  # non-overlapping
    # tracks2 should contain a new track with a different ID.
    old_ids = {t.track_id for t in tracks1}
    new_ids = {t.track_id for t in tracks2}
    # At least one new ID should have appeared (the second object).
    assert new_ids - old_ids, "Expected a new track ID for a non-overlapping detection"


def test_id_survives_small_movement():
    tracker = Tracker(iou_threshold=0.3, max_age=3)
    tracks1 = tracker.update([_det(10, 10, 50, 50)])
    # Shift bbox by 5 px — still high IoU.
    tracks2 = tracker.update([_det(13, 13, 53, 53)])
    assert tracks1[0].track_id == tracks2[0].track_id


# --- Class separation ---

def test_no_cross_class_match():
    tracker = Tracker(iou_threshold=0.3, max_age=3)
    # Establish a car track.
    car_tracks = tracker.update([_det(10, 10, 100, 100, cls=ObjectClass.CAR)])
    car_id = car_tracks[0].track_id
    # Submit overlapping person detection — must NOT match the car track.
    mixed = tracker.update([_det(10, 10, 100, 100, cls=ObjectClass.PERSON)])
    ids = {t.track_id for t in mixed}
    assert car_id not in ids or any(
        t.track_id == car_id and t.cls == ObjectClass.CAR for t in mixed
    ), "Cross-class match occurred"


# --- Ageing out ---

def test_track_ages_out_after_max_missed():
    tracker = Tracker(iou_threshold=0.3, max_age=2)
    tracker.update([_det(10, 10, 50, 50)])
    # Three frames with no matching detection → track should disappear.
    for _ in range(3):
        tracks = tracker.update([])
    assert len(tracks) == 0


def test_track_survives_within_max_missed():
    tracker = Tracker(iou_threshold=0.3, max_age=3)
    tracker.update([_det(10, 10, 50, 50)])
    # Two frames with no detection (< max_age=3).
    for _ in range(2):
        tracks = tracker.update([])
    assert len(tracks) == 1


# --- Multiple objects ---

def test_two_objects_tracked_simultaneously():
    tracker = Tracker(iou_threshold=0.3, max_age=3)
    d1 = _det(0, 0, 40, 40)
    d2 = _det(200, 200, 240, 240)
    tracks = tracker.update([d1, d2])
    assert len(tracks) == 2
    ids = {t.track_id for t in tracks}
    assert len(ids) == 2


def test_track_count_correct_after_object_leaves():
    tracker = Tracker(iou_threshold=0.3, max_age=1)
    tracker.update([_det(0, 0, 40, 40), _det(200, 200, 240, 240)])
    # Only one object remains.
    tracks = tracker.update([_det(0, 0, 40, 40)])
    # After max_age=1 miss, second track disappears.
    tracks2 = tracker.update([_det(0, 0, 40, 40)])
    assert len(tracks2) == 1


# --- Closing speed ---

def test_closing_speed_positive_when_approaching():
    """Object moving toward ego (distance decreasing) → closing_speed_mps > 0."""
    tracker = Tracker(iou_threshold=0.3, max_age=5)
    # Frame 1 at t=0.0, distance=30 m.
    tracker.update([_det(10, 10, 50, 50, dist=30.0)], timestamp=0.0)
    # Frame 2 at t=1.0, distance=25 m → closing at 5 m/s.
    tracks = tracker.update([_det(10, 10, 50, 50, dist=25.0)], timestamp=1.0)
    track = tracks[0]
    assert track.closing_speed_mps is not None
    assert track.closing_speed_mps > 0


def test_closing_speed_negative_when_receding():
    tracker = Tracker(iou_threshold=0.3, max_age=5)
    tracker.update([_det(10, 10, 50, 50, dist=15.0)], timestamp=0.0)
    tracks = tracker.update([_det(10, 10, 50, 50, dist=20.0)], timestamp=1.0)
    track = tracks[0]
    assert track.closing_speed_mps is not None
    assert track.closing_speed_mps < 0


# --- IoU helper (on BBox) ---

def test_iou_identical_boxes():
    b = BBox(0, 0, 10, 10)
    assert b.iou(b) == pytest.approx(1.0)


def test_iou_no_overlap():
    a = BBox(0, 0, 10, 10)
    b = BBox(20, 20, 30, 30)
    assert a.iou(b) == pytest.approx(0.0)


def test_iou_partial_overlap():
    a = BBox(0, 0, 10, 10)
    b = BBox(5, 5, 15, 15)
    # intersection 5x5=25, union 100+100-25=175
    assert a.iou(b) == pytest.approx(25 / 175)


# --- Reset ---

def test_reset_clears_all_tracks():
    tracker = Tracker()
    tracker.update([_det(0, 0, 50, 50)])
    tracker.reset()
    assert tracker.active_track_count == 0
    tracks = tracker.update([_det(0, 0, 50, 50)])
    assert tracks[0].track_id == 1  # IDs restart from 1


# --- Kalman filtering ---

def test_closing_speed_none_until_second_distance():
    tracker = Tracker(iou_threshold=0.3, max_age=5)
    tracks = tracker.update([_det(10, 10, 50, 50, dist=30.0)], timestamp=0.0)
    assert tracks[0].closing_speed_mps is None
    assert tracks[0].distance_m == pytest.approx(30.0)


def test_kalman_rejects_distance_spike():
    """A single wild distance measurement must not yank the filtered distance."""
    tracker = Tracker(iou_threshold=0.3, max_age=5)
    tracker.update([_det(10, 10, 50, 50, dist=30.0)], timestamp=0.0)
    tracker.update([_det(10, 10, 50, 50, dist=29.5)], timestamp=0.1)
    tracker.update([_det(10, 10, 50, 50, dist=29.0)], timestamp=0.2)
    # Spike: monocular depth glitches to 60 m for one frame.
    tracks = tracker.update([_det(10, 10, 50, 50, dist=60.0)], timestamp=0.3)
    assert tracks[0].distance_m < 40.0, "filter swallowed the 60 m spike"
    # Next normal frame pulls it back.
    tracks = tracker.update([_det(10, 10, 50, 50, dist=28.5)], timestamp=0.4)
    assert tracks[0].distance_m < 33.0


def test_kalman_converges_to_constant_closing_speed():
    """Constant 5 m/s approach at 1 Hz → closing speed near 5 within a few frames."""
    tracker = Tracker(iou_threshold=0.3, max_age=5)
    for i, dist in enumerate([50.0, 45.0, 40.0, 35.0, 30.0]):
        tracks = tracker.update([_det(10, 10, 50, 50, dist=dist)], timestamp=float(i))
    closing = tracks[0].closing_speed_mps
    assert closing is not None
    assert 4.0 < closing < 6.0


def test_kalman_survives_zero_distance_measurements():
    """distance_m of exactly 0.0 must not divide-by-zero the filter."""
    tracker = Tracker(iou_threshold=0.3, max_age=5)
    tracker.update([_det(10, 10, 50, 50, dist=0.0)], timestamp=0.0)
    tracker.update([_det(10, 10, 50, 50, dist=0.0)], timestamp=0.1)
    tracks = tracker.update([_det(10, 10, 50, 50, dist=0.0)], timestamp=0.2)
    assert tracks[0].distance_m == pytest.approx(0.0, abs=0.5)


def test_track_without_distance_has_none_fields():
    tracker = Tracker(iou_threshold=0.3, max_age=5)
    tracker.update([_det(10, 10, 50, 50)], timestamp=0.0)
    tracks = tracker.update([_det(10, 10, 50, 50)], timestamp=0.1)
    assert tracks[0].distance_m is None
    assert tracks[0].closing_speed_mps is None
