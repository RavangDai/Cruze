"""
YOLOv8 detector backend via ultralytics.

ultralytics is an optional dependency; importing this module without it
installed raises ImportError with a clear install hint.
"""

from __future__ import annotations

import logging
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
        model_path: str = "models/yolov8n.pt",
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
            for box in r.boxes:
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                obj_cls = _COCO_MAP.get(cls_id, ObjectClass.UNKNOWN)
                detections.append(
                    Detection(
                        bbox=BBox(x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3]),
                        confidence=conf,
                        cls=obj_cls,
                    )
                )
        return detections
