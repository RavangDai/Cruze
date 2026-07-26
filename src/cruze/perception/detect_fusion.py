"""Detection-level fusion of the per-frame vehicle detector with the low-rate
context detector. Pure function — no ML deps, unit-tested standalone
(CLAUDE.md rule)."""

from __future__ import annotations

from cruze.core.types import Detection

# IoU above which a supplementary detection is treated as the same object as an
# existing primary box and dropped as a duplicate. 0.45 mirrors the detector NMS
# threshold — two detectors agree on a vehicle well above this overlap.
_DUP_IOU = 0.45


def merge_detections(
    primary: list[Detection],
    candidates: tuple[Detection, ...],
    iou_thres: float = _DUP_IOU,
) -> list[Detection]:
    """Union of two detection sources, de-duplicated by IoU.

    ``primary`` is kept in full and wins on overlap; a ``candidate`` is appended
    only where it overlaps no primary box.

    In the VisionPilot-aligned pipeline the primary source is AutoSpeed, which
    runs on every processed frame and owns the vehicle classes. Candidates come
    from the low-rate context pass (person / traffic light / stop sign), whose
    classes AutoSpeed cannot produce at all. The IoU filter is therefore mostly
    inert for the classes that matter and exists to stop a stale cached box from
    doubling up on a vehicle when the context detector does see one.
    """
    merged = list(primary)
    for c in candidates:
        if all(c.bbox.iou(p.bbox) < iou_thres for p in primary):
            merged.append(c)
    return merged
