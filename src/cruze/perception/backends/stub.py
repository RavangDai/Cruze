"""
Stub detector — returns zero detections.
Used in CI and for headless --no-camera runs where no ML runtime is available.
"""

from __future__ import annotations

from cruze.core.types import Detection, Frame


class StubDetector:
    """Always returns an empty detection list. Zero dependencies."""

    def detect(self, frame: Frame) -> list[Detection]:
        return []
