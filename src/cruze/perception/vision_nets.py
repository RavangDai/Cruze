"""Holder for the enabled vision_pilot ONNX estimators. Runs them on one frame,
sharing the crop-2:1 preprocessing between AutoSpeed and AutoSteer, and returns
one EgoEstimate bundle for the PERCEPTION_EGO channel."""

from __future__ import annotations

import logging

from cruze.core.types import EgoEstimate, Frame
from cruze.perception import preprocess

logger = logging.getLogger(__name__)


class VisionNets:
    def __init__(self, autospeed=None, autosteer=None, autodrive=None) -> None:
        self._autospeed = autospeed
        self._autosteer = autosteer
        self._autodrive = autodrive
        # Failure keys already logged once — avoids per-frame log spam when a
        # net/preprocessing fault persists (e.g. cv2 missing, weight mismatch).
        self._logged_failures: set[str] = set()

    @property
    def any_enabled(self) -> bool:
        return any((self._autospeed, self._autosteer, self._autodrive))

    def _log_failure_once(self, key: str, msg: str) -> None:
        if key not in self._logged_failures:
            self._logged_failures.add(key)
            logger.exception(msg)

    def infer(self, frame: Frame) -> EgoEstimate | None:
        if not self.any_enabled:
            return None
        cipo_boxes: tuple = ()
        ego_path = None
        dist = curv = flag = None

        if self._autospeed is not None or self._autosteer is not None:
            # Graceful degradation: a failure here (missing cv2, bad frame,
            # etc.) must not drop the whole perception frame — skip both nets
            # this cycle and leave their outputs at the defaults above.
            shared = None
            try:
                shared = preprocess.preprocess_crop2_1(frame.image)
            except Exception:
                self._log_failure_once(
                    "preprocess",
                    "VisionNets: crop-2:1 preprocessing failed — AutoSpeed/AutoSteer skipped this frame",
                )

            if shared is not None:
                chw, sx, sy, crop_top = shared
                if self._autospeed is not None:
                    try:
                        cipo_boxes = self._autospeed.infer(chw, sx, sy, crop_top)
                    except Exception:
                        self._log_failure_once(
                            "autospeed",
                            "VisionNets: AutoSpeed inference failed — degraded to empty this frame",
                        )
                if self._autosteer is not None:
                    try:
                        ego_path = self._autosteer.infer(chw, sx, sy, crop_top)
                    except Exception:
                        self._log_failure_once(
                            "autosteer",
                            "VisionNets: AutoSteer inference failed — degraded to empty this frame",
                        )

        if self._autodrive is not None:
            try:
                res = self._autodrive.infer(frame.image)
                if res is not None:
                    dist, curv, flag = res.cipo_distance_m, res.road_curvature_1pm, res.cipo_flag
            except Exception:
                self._log_failure_once(
                    "autodrive",
                    "VisionNets: AutoDrive inference failed — degraded to empty this frame",
                )

        return EgoEstimate(
            timestamp=frame.timestamp, frame_id=frame.frame_id,
            cipo_boxes=cipo_boxes, ego_path=ego_path,
            cipo_distance_m=dist, road_curvature_1pm=curv, cipo_flag=flag,
        )


def build_vision_nets(cfg) -> VisionNets:
    """Construct estimators from config. Each is independent and degrades
    gracefully: an enabled net whose deps/weights are missing logs a warning
    and is left out (never crashes startup)."""
    pc = cfg.perception
    autospeed = autosteer = autodrive = None

    if pc.autospeed_enabled:
        try:
            from cruze.perception.backends.autospeed import AutoSpeedEstimator
            autospeed = AutoSpeedEstimator(
                pc.autospeed_model_path, pc.onnx_provider,
                pc.autospeed_conf_threshold, pc.autospeed_iou_threshold)
        except Exception:
            logger.exception("AutoSpeed disabled — failed to load")

    if pc.autosteer_enabled:
        try:
            from cruze.perception.autosteer import AutoSteerEstimator
            autosteer = AutoSteerEstimator(pc.autosteer_model_path, pc.onnx_provider)
        except Exception:
            logger.exception("AutoSteer disabled — failed to load")

    if pc.autodrive_enabled:
        try:
            from cruze.perception.autodrive import AutoDriveEstimator, load_homography
            homography = load_homography(pc.autodrive_homography_path)
            autodrive = AutoDriveEstimator(
                pc.autodrive_model_path, homography, pc.onnx_provider,
                pc.autodrive_curv_scale, pc.autodrive_flag_threshold)
        except Exception:
            logger.exception("AutoDrive disabled — failed to load")

    return VisionNets(autospeed, autosteer, autodrive)
