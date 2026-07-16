"""AutoSpeed (vision_pilot) — YOLO-style vehicle detector, run alongside YOLO
as a CIPO/lead candidate source. Emits Detections in raw-frame pixels."""

from __future__ import annotations

import numpy as np

from cruze.core.types import BBox, Detection, ObjectClass
from cruze.perception import preprocess
from cruze.perception.onnx_runtime import OnnxSession

# AutoSpeed class-id → ObjectClass. Taxonomy pinned at implementation from the
# ONNX metadata (C-4 = num_classes) + the upstream auto_speed repo; unknown ids
# fall through to UNKNOWN. Placeholder map (vehicle-centric) until confirmed.
_AUTOSPEED_CLASSES: dict[int, ObjectClass] = {
    0: ObjectClass.CAR,
    1: ObjectClass.CAR,
    2: ObjectClass.TRUCK,
    3: ObjectClass.BUS,
}


class AutoSpeedEstimator:
    def __init__(self, model_path: str, provider: str = "cpu",
                 conf_threshold: float = 0.6, iou_threshold: float = 0.45,
                 class_map: dict[int, ObjectClass] | None = None, session=None) -> None:
        self._conf = conf_threshold
        self._iou = iou_threshold
        self._class_map = _AUTOSPEED_CLASSES if class_map is None else class_map
        self._session = session if session is not None else OnnxSession(model_path, provider)

    def infer(self, chw: np.ndarray, sx: float, sy: float, crop_top: int) -> tuple[Detection, ...]:
        raw = self._session.run({self._session.input_names[0]: chw})[0]
        boxes, scores, class_ids = preprocess.decode_yolo(raw, self._conf)
        out: list[Detection] = []
        for i in preprocess.nms(boxes, scores, self._iou):
            x1, y1 = preprocess.unmap_point(boxes[i, 0], boxes[i, 1], sx, sy, crop_top)
            x2, y2 = preprocess.unmap_point(boxes[i, 2], boxes[i, 3], sx, sy, crop_top)
            cls = self._class_map.get(int(class_ids[i]), ObjectClass.UNKNOWN)
            out.append(Detection(BBox(float(x1), float(y1), float(x2), float(y2)), float(scores[i]), cls))
        return tuple(out)
