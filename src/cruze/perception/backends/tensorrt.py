"""
TensorRT detector backend — for NVIDIA Jetson Orin Nano.

TODO: implement model conversion from .pt → .engine:
    yolo export model=yolov8n.pt format=engine device=0 half=True imgsz=640
Then load with tensorrt.Runtime, allocate CUDA buffers, run async inference.
"""

from __future__ import annotations

from cruze.core.types import Detection, Frame


class TensorRTDetector:
    def __init__(self, model_path: str = "models/yolov8n.engine", **kwargs) -> None:
        try:
            import tensorrt  # type: ignore  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "tensorrt is required for the tensorrt backend. "
                "See deploy/jetson/setup.sh — TensorRT ships with JetPack."
            ) from exc
        raise NotImplementedError(
            "TensorRT backend: load .engine, allocate CUDA buffers, run inference. "
            "See TODO in this file."
        )

    def detect(self, frame: Frame) -> list[Detection]:  # pragma: no cover
        raise NotImplementedError
