"""AutoSteer (vision_pilot) — single-frame ego-path net. 64 lateral samples at
fixed image rows; keeps samples whose confidence h_vector >= 0.5, mapped to raw px."""

from __future__ import annotations

import numpy as np

from cruze.perception import preprocess
from cruze.perception.onnx_runtime import OnnxSession

_N_PTS = 64
# Waypoint confidence cutoff (vision_pilot debug_draw: h_vector >= 0.5).
_MASK_THRESHOLD = 0.5


class AutoSteerEstimator:
    def __init__(self, model_path: str, provider: str = "cpu", session=None,
                 intra_op_threads: int = 0) -> None:
        self._session = (
            session if session is not None
            else OnnxSession(model_path, provider, intra_op_threads)
        )
        # Fixed sample rows in net px: np.linspace(0, 511, 64).
        self._rows = np.linspace(0, preprocess.NET_H - 1, _N_PTS)

    def infer(self, chw: np.ndarray, sx: float, sy: float,
              crop_top: int) -> tuple[tuple[float, float], ...] | None:
        xp, h_vector = self._read(self._session.run({self._session.input_names[0]: chw}))
        pts: list[tuple[float, float]] = []
        for i in range(_N_PTS):
            if h_vector[i] < _MASK_THRESHOLD:
                continue
            u = float(xp[i]) * preprocess.NET_W
            pts.append(preprocess.unmap_point(u, float(self._rows[i]), sx, sy, crop_top))
        return tuple(pts) if pts else None

    @staticmethod
    def _read(outs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Two (1,64) tensors → (xp, h_vector), flattened."""
        return np.asarray(outs[0]).reshape(-1), np.asarray(outs[1]).reshape(-1)
