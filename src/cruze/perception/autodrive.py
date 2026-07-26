"""AutoDrive (vision_pilot) — two-frame BEV net producing lead distance, road
curvature, and a CIPO in-path flag. Requires a camera-matched homography; see
the spec §11 — defaults OFF, the vendored C only fits vision_pilot's camera."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cruze.perception import preprocess
from cruze.perception.onnx_runtime import OnnxSession

# AutoDrive normalised-distance full scale (vision_pilot D_MAX_M, empirical).
_D_MAX_M = 150.0


@dataclass(frozen=True)
class AutoDriveResult:
    cipo_distance_m: float
    road_curvature_1pm: float
    cipo_flag: bool


def load_homography(path: str) -> np.ndarray:
    """Load the raw-px→BEV homography 'C' (3x3) from a YAML file with key 'C'
    holding 9 row-major values."""
    import yaml
    with open(path) as fh:
        data = yaml.safe_load(fh)
    return np.asarray(data["C"], dtype=np.float32).reshape(3, 3)


class AutoDriveEstimator:
    def __init__(self, model_path: str, homography: np.ndarray, provider: str = "cpu",
                 curv_scale: float = 1.0, flag_threshold: float = 0.5, session=None,
                 intra_op_threads: int = 0) -> None:
        self._homography = np.asarray(homography, dtype=np.float32)
        self._curv_scale = curv_scale
        self._flag_threshold = flag_threshold
        self._session = (
            session if session is not None
            else OnnxSession(model_path, provider, intra_op_threads)
        )
        self._prev_chw: np.ndarray | None = None

    def infer(self, image_bgr: np.ndarray) -> AutoDriveResult | None:
        curr = preprocess.preprocess_bev(image_bgr, self._homography)
        prev, self._prev_chw = self._prev_chw, curr
        if prev is None:
            return None  # two-frame model — first frame only primes the buffer
        names = self._session.input_names
        dist_norm, curv_raw, flag_prob = self._read(self._session.run({names[0]: prev, names[1]: curr}))
        return AutoDriveResult(
            cipo_distance_m=_D_MAX_M * (1.0 - float(dist_norm)),
            road_curvature_1pm=float(curv_raw) * self._curv_scale,
            cipo_flag=float(flag_prob) >= self._flag_threshold,
        )

    @staticmethod
    def _read(outs: list[np.ndarray]) -> tuple[float, float, float]:
        """Accept three scalar tensors (or one [1,3]) → (dist_norm, curv_raw, flag_prob)."""
        flat = np.concatenate([np.asarray(o).reshape(-1) for o in outs])
        return float(flat[0]), float(flat[1]), float(flat[2])
