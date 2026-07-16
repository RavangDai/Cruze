"""
YOLOv8 detector backend via ultralytics.

Works with both box-only (yolov8n.pt) and segmentation (yolov8n-seg.pt)
weights: seg models additionally fill Detection.mask_xy with a downsampled
instance-mask outline polygon.

ultralytics is an optional dependency; importing this module without it
installed raises ImportError with a clear install hint. The module itself
imports clean so the pure helpers stay unit-testable.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from cruze.core.types import BBox, Detection, Frame, ObjectClass

logger = logging.getLogger(__name__)

# Ultralytics COCO class IDs → ObjectClass.
_COCO_MAP: dict[int, ObjectClass] = {
    0: ObjectClass.PERSON,
    1: ObjectClass.BICYCLE,
    2: ObjectClass.CAR,
    3: ObjectClass.MOTORCYCLE,
    5: ObjectClass.BUS,
    7: ObjectClass.TRUCK,
    11: ObjectClass.STOP_SIGN,
    9: ObjectClass.TRAFFIC_LIGHT,
}

# 32 points keeps the outline within ~2 px of the full mask for typical
# vehicles at 720p while capping wire cost (~450 B/track as JSON).
_MASK_MAX_POINTS = 32


def _downsample_polygon(points: Any) -> tuple[tuple[float, float], ...] | None:
    """
    Reduce an (N, 2) mask-outline array (ultralytics results.masks.xy entry,
    image pixel coords) to at most _MASK_MAX_POINTS, rounded to 0.1 px.
    Returns None for degenerate polygons (< 3 points) — callers draw the
    bbox only in that case.
    """
    if points is None or len(points) < 3:
        return None
    step = max(1, math.ceil(len(points) / _MASK_MAX_POINTS))
    return tuple(
        (round(float(x), 1), round(float(y), 1)) for x, y in points[::step]
    )


class YoloDetector:
    """
    Wraps ultralytics YOLOv8 model.

    Parameters
    ----------
    model_path:
        Path to .pt weights. On first run ultralytics will download if missing.
    confidence_threshold:
        Detections below this score are discarded.
    device:
        PyTorch device string — "cpu", "cuda:0", "mps", etc.
    """

    def __init__(
        self,
        model_path: str = "models/yolov8n-seg.pt",
        confidence_threshold: float = 0.4,
        device: str = "cpu",
    ) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for the yolo backend. "
                "Install it with: pip install 'cruze[vision]'"
            ) from exc

        logger.info("Loading YOLOv8 model from %s on device=%s", model_path, device)
        self._model: Any = YOLO(model_path)
        self._model.to(device)
        self._conf = confidence_threshold

    def detect(self, frame: Frame) -> list[Detection]:
        results = self._model(frame.image, conf=self._conf, verbose=False)
        detections: list[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            # Seg weights fill r.masks with outline polygons index-aligned to
            # r.boxes; box-only weights leave it None (masks simply absent).
            mask_polys = r.masks.xy if r.masks is not None else None
            for i, box in enumerate(r.boxes):
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                obj_cls = _COCO_MAP.get(cls_id, ObjectClass.UNKNOWN)
                mask_xy = None
                if mask_polys is not None and i < len(mask_polys):
                    mask_xy = _downsample_polygon(mask_polys[i])
                detections.append(
                    Detection(
                        bbox=BBox(x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3]),
                        confidence=conf,
                        cls=obj_cls,
                        mask_xy=mask_xy,
                    )
                )
        return detections
