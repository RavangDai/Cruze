"""
TFLite detector backend — for edge devices (Pi 5 + Hailo-8, quantized models).

TODO: implement model conversion from .pt → .tflite:
    yolo export model=yolov8n.pt format=tflite int8=True imgsz=640
Then load with TFLite interpreter and post-process NMS output.
"""

from __future__ import annotations

from cruze.core.types import Detection, Frame


class TFLiteDetector:
    def __init__(self, model_path: str = "models/yolov8n.tflite", **kwargs) -> None:
        try:
            import tflite_runtime.interpreter as tflite  # type: ignore  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "tflite_runtime is required for the tflite backend. "
                "See deploy/jetson/setup.sh for installation instructions."
            ) from exc
        raise NotImplementedError(
            "TFLite backend: load model, run inference, decode NMS output. "
            "See TODO in this file."
        )

    def detect(self, frame: Frame) -> list[Detection]:  # pragma: no cover
        raise NotImplementedError
