"""
Detector protocol + factory.

Any object with a `detect(frame) -> list[Detection]` method satisfies the
Detector protocol. The factory maps backend names to implementations and
raises ImportError with a helpful message if optional deps are missing.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from cruze.core.types import Detection, Frame


@runtime_checkable
class Detector(Protocol):
    def detect(self, frame: Frame) -> list[Detection]: ...


def load(backend: str, **kwargs) -> Detector:
    """
    Instantiate and return the requested detector backend.

    Parameters
    ----------
    backend:
        "stub" | "yolo" | "tflite" | "tensorrt"
    **kwargs:
        Passed to the backend constructor (model_path, confidence_threshold, …).
    """
    backend = backend.lower()

    if backend == "stub":
        from cruze.perception.backends.stub import StubDetector
        return StubDetector()

    if backend == "yolo":
        from cruze.perception.backends.yolo import YoloDetector
        return YoloDetector(**kwargs)

    if backend == "tflite":
        from cruze.perception.backends.tflite import TFLiteDetector
        return TFLiteDetector(**kwargs)

    if backend == "tensorrt":
        from cruze.perception.backends.tensorrt import TensorRTDetector
        return TensorRTDetector(**kwargs)

    raise ValueError(
        f"Unknown detector backend '{backend}'. "
        "Valid choices: stub, yolo, tflite, tensorrt."
    )
