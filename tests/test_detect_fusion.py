"""detect_fusion.merge_detections — the per-frame vehicle detector (AutoSpeed)
is primary; the low-rate context pass gap-fills the classes it cannot see."""

from cruze.core.types import BBox, Detection, ObjectClass
from cruze.perception.detect_fusion import merge_detections


def _det(x1, y1, x2, y2, cls=ObjectClass.CAR, conf=0.8):
    return Detection(BBox(x1, y1, x2, y2), conf, cls)


def test_non_overlapping_candidate_appended():
    primary = [_det(0, 0, 10, 10)]
    cand = (_det(100, 100, 110, 110, ObjectClass.TRUCK),)
    out = merge_detections(primary, cand)
    assert len(out) == 2
    assert out[1].cls is ObjectClass.TRUCK


def test_overlapping_candidate_dropped():
    primary = [_det(0, 0, 100, 100)]                 # area 10000
    cand = (_det(5, 5, 98, 98, ObjectClass.BUS),)    # IoU ~0.86 with primary
    out = merge_detections(primary, cand)
    assert len(out) == 1
    assert out[0].cls is ObjectClass.CAR             # primary wins on overlap


def test_context_classes_survive_alongside_vehicles():
    # The realistic case: AutoSpeed supplies vehicles, the context pass supplies
    # the classes it has no head for. Nothing overlaps, so nothing is lost.
    vehicles = [_det(0, 0, 100, 100), _det(200, 0, 300, 100, ObjectClass.TRUCK)]
    context = (
        _det(400, 0, 430, 120, ObjectClass.PERSON),
        _det(600, 0, 620, 40, ObjectClass.TRAFFIC_LIGHT),
    )
    out = merge_detections(vehicles, context)
    assert {d.cls for d in out} == {
        ObjectClass.CAR, ObjectClass.TRUCK,
        ObjectClass.PERSON, ObjectClass.TRAFFIC_LIGHT,
    }


def test_stale_cached_vehicle_never_doubles_a_live_one():
    # A cached context box lingering on a vehicle AutoSpeed already reported
    # must not produce a second bracket on the same car.
    live = [_det(10, 10, 110, 110)]
    cached = (_det(12, 12, 108, 108, ObjectClass.CAR),)
    assert len(merge_detections(live, cached)) == 1


def test_empty_candidates_returns_primary_copy():
    primary = [_det(0, 0, 10, 10)]
    out = merge_detections(primary, ())
    assert out == primary
    assert out is not primary                        # a fresh list, not aliased


def test_empty_primary_keeps_all_candidates():
    cand = (_det(0, 0, 10, 10), _det(50, 50, 60, 60))
    out = merge_detections([], cand)
    assert len(out) == 2


def test_threshold_boundary_partial_overlap_kept():
    # Small overlap (IoU well below 0.45) -> candidate is a distinct object.
    primary = [_det(0, 0, 100, 100)]
    cand = (_det(90, 90, 190, 190),)                 # IoU ~0.05
    out = merge_detections(primary, cand)
    assert len(out) == 2
