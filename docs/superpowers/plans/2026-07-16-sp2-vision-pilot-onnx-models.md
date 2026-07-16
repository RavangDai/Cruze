# SP2 — vision_pilot ONNX Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run vision_pilot's AutoSpeed, AutoSteer, and AutoDrive ONNX nets as native, config-selectable Cruze estimators that publish their raw outputs onto a new `PERCEPTION_EGO` channel.

**Architecture:** Three optional estimators layer on top of the existing pipeline — YOLO stays the primary detector, untouched. AutoSpeed (vehicle boxes) and AutoSteer (ego-path) share one crop-2:1 preprocessing; AutoDrive (distance/curvature/CIPO-flag) uses a two-frame BEV-warped buffer. A `VisionNets` holder runs the enabled nets per frame and emits an `EgoEstimate` bundle, which `SceneAssembler` folds onto new optional `Scene` fields. Fusion (particle filter, RANSAC) is out of scope — SP3.

**Tech Stack:** Python 3.11+, asyncio event bus, `onnxruntime` (optional `cruze[onnx]` extra), numpy (core dep), OpenCV (optional `cruze[vision]`, for warp/resize only), pytest.

**Spec:** [`docs/superpowers/specs/2026-07-16-sp2-vision-pilot-onnx-models-design.md`](../specs/2026-07-16-sp2-vision-pilot-onnx-models-design.md)

## Global Constraints

Every task's requirements implicitly include these (verbatim from the spec / CLAUDE.md):

- **Python ≥ 3.11.**
- **No ML deps in tests.** `python -m pytest tests/` must pass with only pytest, pytest-asyncio, stdlib, and **numpy**. `onnxruntime` and `cv2` must never be imported at test time — use fake sessions and monkeypatch cv2-backed functions.
- **Bus only.** Modules import only from `cruze.core.*`; cross-module communication goes through the bus. (`VisionNets` and estimators live in `cruze.perception.*` and import only `cruze.core.*` + sibling perception helpers, which is the existing pipeline pattern.)
- **Additive data contract only.** Add optional fields with defaults; never rename or remove. New channel is additive.
- **No blocking calls in the asyncio loop.** All net inference runs inside the existing `run_in_executor` hop in `PerceptionService._analyze`.
- **Magic numbers carry a comment** stating their physical/empirical basis.
- **Net constants:** `NET_W=1024`, `NET_H=512`. ImageNet `mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`. Crop: `crop_top = max(0, round(H − W/2))`. Un-map: `x_raw = x·(W/1024)`, `y_raw = y·((H−crop_top)/512) + crop_top`. AutoSpeed defaults `conf=0.6`, `iou=0.45`. AutoDrive: `dist_m = 150·(1−dist_norm)`, `curv_1pm = raw·curv_scale`.
- **All three nets default OFF.** They are enabled only via a hardware profile.
- **Commits** are checkpoints; the repo owner performs them (plain messages, no Co-Authored-By / Claude attribution).

---

### Task 1: Pure preprocessing helpers

**Files:**
- Create: `src/cruze/perception/preprocess.py`
- Test: `tests/test_preprocess.py`

**Interfaces:**
- Produces:
  - `NET_W=1024`, `NET_H=512` (module constants)
  - `top_crop_2_1(h:int, w:int) -> int`
  - `crop_resize_params(h:int, w:int) -> tuple[int,float,float]` → `(crop_top, sx, sy)`
  - `unmap_point(x:float, y:float, sx:float, sy:float, crop_top:int) -> tuple[float,float]`
  - `chw_from_rgb01(rgb01:np.ndarray, imagenet:bool) -> np.ndarray` → `[1,3,H,W]` float32
  - `decode_yolo(raw:np.ndarray, conf_thres:float) -> tuple[np.ndarray,np.ndarray,np.ndarray]` → `(boxes[M,4] xyxy, scores[M], class_ids[M])` in net px
  - `nms(boxes:np.ndarray, scores:np.ndarray, iou_thres:float) -> list[int]`
  - `preprocess_crop2_1(image_bgr:np.ndarray) -> tuple[np.ndarray,float,float,int]` (cv2-backed)
  - `preprocess_bev(image_bgr:np.ndarray, homography:np.ndarray) -> np.ndarray` (cv2-backed)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_preprocess.py
"""Preprocessing helpers — pure numpy, no cv2/onnx."""
import numpy as np
import pytest

from cruze.perception import preprocess as pp


def test_top_crop_keeps_bottom_2_1():
    # 1280x720: keep bottom W/2 = 640 rows → crop 80 from the top.
    assert pp.top_crop_2_1(720, 1280) == 80


def test_top_crop_never_negative():
    # Already taller-than-2:1 region → no crop.
    assert pp.top_crop_2_1(400, 1280) == 0


def test_crop_resize_params():
    crop_top, sx, sy = pp.crop_resize_params(720, 1280)
    assert crop_top == 80
    assert sx == pytest.approx(1280 / 1024)
    assert sy == pytest.approx((720 - 80) / 512)


def test_unmap_corners_round_trip():
    crop_top, sx, sy = pp.crop_resize_params(720, 1280)
    assert pp.unmap_point(0, 0, sx, sy, crop_top) == pytest.approx((0.0, 80.0))
    assert pp.unmap_point(1024, 512, sx, sy, crop_top) == pytest.approx((1280.0, 720.0))


def test_chw_shape_and_plain_scaling():
    rgb01 = np.full((512, 1024, 3), 0.5, dtype=np.float32)
    chw = pp.chw_from_rgb01(rgb01, imagenet=False)
    assert chw.shape == (1, 3, 512, 1024)
    assert chw.dtype == np.float32
    assert chw[0, 0, 0, 0] == pytest.approx(0.5)


def test_chw_imagenet_norm():
    rgb01 = np.zeros((1, 1, 3), dtype=np.float32)  # tiny image, values 0
    chw = pp.chw_from_rgb01(rgb01, imagenet=True)
    # (0 - mean)/std for the red channel.
    assert chw[0, 0, 0, 0] == pytest.approx((0.0 - 0.485) / 0.229)


def test_decode_yolo_thresholds_and_boxes():
    # raw [1, C=5, N=2]: 4 box params + 1 class logit.
    raw = np.zeros((1, 5, 2), dtype=np.float32)
    raw[0, :4, 0] = [512, 256, 100, 80]   # anchor 0: cx,cy,w,h
    raw[0, 4, 0] = 10.0                    # logit → sigmoid ≈ 1 (kept)
    raw[0, 4, 1] = -10.0                   # anchor 1 → sigmoid ≈ 0 (dropped)
    boxes, scores, class_ids = pp.decode_yolo(raw, conf_thres=0.5)
    assert boxes.shape == (1, 4)
    assert boxes[0] == pytest.approx([462, 216, 562, 296])
    assert class_ids[0] == 0
    assert scores[0] > 0.99


def test_nms_suppresses_overlap():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [100, 100, 110, 110]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    keep = pp.nms(boxes, scores, iou_thres=0.5)
    assert keep == [0, 2]  # box 1 suppressed by box 0; distant box survives
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_preprocess.py -v`
Expected: FAIL — `ModuleNotFoundError: cruze.perception.preprocess`

- [ ] **Step 3: Implement `preprocess.py`**

```python
# src/cruze/perception/preprocess.py
"""Pure-numpy preprocessing for the vision_pilot ONNX nets, plus two thin
cv2-backed spatial transforms. All coordinate/decoding math is numpy-only so
it is unit-tested without cv2 or onnxruntime (CLAUDE.md testing rule)."""

from __future__ import annotations

import numpy as np

# vision_pilot net input size (see AutoSpeed/AutoSteer/AutoDrive headers).
NET_W = 1024
NET_H = 512

# ImageNet normalisation — AutoDrive only (inference.cpp chw_imagenet).
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def top_crop_2_1(h: int, w: int) -> int:
    """Rows to drop from the top so the kept region is 2:1 (w:h) — drops sky,
    keeps the bottom w/2 rows. Matches vision_pilot compute_top_crop_2_1()."""
    return max(0, round(h - w / 2.0))


def crop_resize_params(h: int, w: int) -> tuple[int, float, float]:
    """(crop_top, sx, sy) — the scale factors that map resized 1024x512 px
    back to raw px after crop-2:1 + resize."""
    crop_top = top_crop_2_1(h, w)
    sx = w / NET_W
    sy = (h - crop_top) / NET_H
    return crop_top, sx, sy


def unmap_point(x: float, y: float, sx: float, sy: float, crop_top: int) -> tuple[float, float]:
    """Resized-space (1024x512) pixel → raw-frame pixel."""
    return x * sx, y * sy + crop_top


def chw_from_rgb01(rgb01: np.ndarray, imagenet: bool) -> np.ndarray:
    """HWC float RGB in [0,1] → contiguous [1,3,H,W] float32, optional ImageNet norm."""
    arr = rgb01.astype(np.float32, copy=False)
    if imagenet:
        arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    return np.ascontiguousarray(np.transpose(arr, (2, 0, 1))[None])


def decode_yolo(raw: np.ndarray, conf_thres: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """YOLO decode of [1, 4+K, N] → (boxes[M,4] xyxy, scores[M], class_ids[M])
    in net px. Mirrors auto_speed.cpp post_process (sigmoid + argmax + threshold)."""
    data = raw[0]                       # [C, N]
    cx, cy, w, h = data[0], data[1], data[2], data[3]
    logits = data[4:]                   # [K, N]
    probs = 1.0 / (1.0 + np.exp(-logits))
    class_ids = np.argmax(probs, axis=0)
    scores = probs[class_ids, np.arange(probs.shape[1])]
    keep = scores >= conf_thres
    boxes = np.stack(
        [cx[keep] - w[keep] / 2, cy[keep] - h[keep] / 2,
         cx[keep] + w[keep] / 2, cy[keep] + h[keep] / 2],
        axis=1,
    ).astype(np.float32)
    return boxes, scores[keep].astype(np.float32), class_ids[keep].astype(np.int64)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> list[int]:
    """Greedy NMS → kept indices, highest score first."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thres]
    return keep


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without cv2
        raise ImportError(
            "opencv-python required for image preprocessing. "
            "Install with: pip install 'cruze[vision]'"
        ) from exc
    return cv2


def preprocess_crop2_1(image_bgr: np.ndarray) -> tuple[np.ndarray, float, float, int]:
    """Raw BGR frame → ([1,3,512,1024] RGB[0,1] CHW, sx, sy, crop_top).
    Shared input for AutoSpeed + AutoSteer."""
    cv2 = _require_cv2()
    h, w = image_bgr.shape[:2]
    crop_top, sx, sy = crop_resize_params(h, w)
    cropped = image_bgr[crop_top:h, 0:w]
    resized = cv2.resize(cropped, (NET_W, NET_H), interpolation=cv2.INTER_LINEAR)
    rgb01 = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return chw_from_rgb01(rgb01, imagenet=False), sx, sy, crop_top


def preprocess_bev(image_bgr: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Raw BGR frame → BEV-warped [1,3,512,1024] CHW, ImageNet-normed. AutoDrive input."""
    cv2 = _require_cv2()
    warped = cv2.warpPerspective(
        image_bgr, homography, (NET_W, NET_H),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
    )
    rgb01 = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return chw_from_rgb01(rgb01, imagenet=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_preprocess.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add src/cruze/perception/preprocess.py tests/test_preprocess.py
git commit -m "feat(perception): add vision_pilot preprocessing helpers"
```

---

### Task 2: Shared ONNX Runtime engine

**Files:**
- Create: `src/cruze/perception/onnx_runtime.py`
- Modify: `pyproject.toml` (add `onnx` extra)
- Test: `tests/test_onnx_runtime.py`

**Interfaces:**
- Produces:
  - `select_providers(provider:str, available:list[str]) -> list[str]`
  - `class OnnxSession(model_path:str, provider:str="cpu")` with `.input_names: list[str]`, `.output_names: list[str]`, `.run(feeds:dict[str,np.ndarray]) -> list[np.ndarray]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_onnx_runtime.py
"""ONNX provider selection — pure logic, no onnxruntime import."""
from cruze.perception.onnx_runtime import select_providers


def test_cpu_only():
    assert select_providers("cpu", ["CPUExecutionProvider"]) == ["CPUExecutionProvider"]


def test_cuda_available():
    got = select_providers("cuda", ["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert got == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_cuda_unavailable_falls_back_to_cpu():
    assert select_providers("cuda", ["CPUExecutionProvider"]) == ["CPUExecutionProvider"]


def test_tensorrt_prefers_trt_then_cuda():
    avail = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    assert select_providers("tensorrt", avail)[0] == "TensorrtExecutionProvider"


def test_unknown_provider_is_cpu():
    assert select_providers("bogus", ["CPUExecutionProvider"]) == ["CPUExecutionProvider"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_onnx_runtime.py -v`
Expected: FAIL — `ModuleNotFoundError: cruze.perception.onnx_runtime`

- [ ] **Step 3: Implement `onnx_runtime.py`**

```python
# src/cruze/perception/onnx_runtime.py
"""Thin wrapper over onnxruntime.InferenceSession with config-driven execution
providers and graceful CPU fallback. onnxruntime is an optional dependency —
imported lazily so this module (and select_providers) stay import-clean for tests."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_PROVIDER_MAP: dict[str, list[str]] = {
    "cpu": ["CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "tensorrt": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
}


def select_providers(provider: str, available: list[str]) -> list[str]:
    """Ordered EP list for the requested provider, filtered to what's installed,
    always ending in CPU. Unknown provider → CPU. Logs a warning on fallback."""
    wanted = _PROVIDER_MAP.get(provider.lower(), ["CPUExecutionProvider"])
    chosen = [p for p in wanted if p in available]
    if "CPUExecutionProvider" not in chosen:
        chosen.append("CPUExecutionProvider")
    if chosen[0] != wanted[0]:
        logger.warning("ONNX provider '%s' unavailable; falling back to %s", provider, chosen[0])
    return chosen


class OnnxSession:
    def __init__(self, model_path: str, provider: str = "cpu") -> None:
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "onnxruntime required for ONNX model backends. "
                "Install with: pip install 'cruze[onnx]'"
            ) from exc
        providers = select_providers(provider, ort.get_available_providers())
        self._session = ort.InferenceSession(model_path, providers=providers)
        self._input_names = [i.name for i in self._session.get_inputs()]
        self._output_names = [o.name for o in self._session.get_outputs()]
        logger.info("OnnxSession %s providers=%s in=%s out=%s",
                    model_path, providers, self._input_names, self._output_names)

    @property
    def input_names(self) -> list[str]:
        return self._input_names

    @property
    def output_names(self) -> list[str]:
        return self._output_names

    def run(self, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        return self._session.run(self._output_names, feeds)
```

- [ ] **Step 4: Add the `onnx` extra to `pyproject.toml`**

In `[project.optional-dependencies]`, add after the `dashboard` block:

```toml
onnx = [
    "onnxruntime>=1.17",   # use onnxruntime-gpu instead for cuda/tensorrt providers
]
```

And add `"cruze[onnx]",` to the `all` list.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_onnx_runtime.py -v`
Expected: PASS (5 passed)

- [ ] **Step 6: Commit**

```bash
git add src/cruze/perception/onnx_runtime.py tests/test_onnx_runtime.py pyproject.toml
git commit -m "feat(perception): add OnnxSession engine with provider fallback"
```

---

### Task 3: Data contract — `EgoEstimate`, Scene fields, channel

**Files:**
- Modify: `src/cruze/core/types.py`
- Modify: `src/cruze/core/bus.py:27` (after `PERCEPTION_LANES`)
- Test: `tests/test_types.py` (extend)

**Interfaces:**
- Produces:
  - `EgoEstimate(timestamp, frame_id, cipo_boxes:tuple[Detection,...], ego_path, cipo_distance_m, road_curvature_1pm, cipo_flag)` — all optional/defaulted
  - `Scene` gains optional `ego_path`, `road_curvature_1pm`, `cipo_distance_m`, `cipo_flag` (default `None`)
  - `Channel.PERCEPTION_EGO = "perception.ego"`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_types.py  (append)
from cruze.core.types import EgoEstimate, Scene, Detection, BBox, ObjectClass
from cruze.core.bus import Channel


def test_ego_estimate_defaults():
    ego = EgoEstimate()
    assert ego.cipo_boxes == ()
    assert ego.ego_path is None
    assert ego.cipo_distance_m is None
    assert ego.road_curvature_1pm is None
    assert ego.cipo_flag is None


def test_ego_estimate_carries_boxes():
    d = Detection(BBox(0, 0, 1, 1), 0.9, ObjectClass.CAR)
    ego = EgoEstimate(cipo_boxes=(d,), cipo_distance_m=42.0, cipo_flag=True)
    assert ego.cipo_boxes[0].cls is ObjectClass.CAR
    assert ego.cipo_distance_m == 42.0


def test_scene_ego_fields_default_none():
    s = Scene()
    assert s.ego_path is None
    assert s.road_curvature_1pm is None
    assert s.cipo_distance_m is None
    assert s.cipo_flag is None


def test_channel_ego_constant():
    assert Channel.PERCEPTION_EGO == "perception.ego"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_types.py -k "ego or channel_ego" -v`
Expected: FAIL — `ImportError: cannot import name 'EgoEstimate'`

- [ ] **Step 3: Add `PERCEPTION_EGO` to `bus.py`**

In `src/cruze/core/bus.py`, add the line after `PERCEPTION_LANES = "perception.lanes"`:

```python
    PERCEPTION_EGO = "perception.ego"
```

- [ ] **Step 4: Add `EgoEstimate` and Scene fields to `types.py`**

After the `Lanes` dataclass, add:

```python
@dataclass(frozen=True)
class EgoEstimate:
    """Per-frame bundle of vision_pilot ONNX net outputs (raw, pre-fusion).
    Mirrors vision_pilot's InferenceFrameResult. Any field may be None/empty
    when its net is disabled or produced no output this frame."""

    timestamp: float = field(default_factory=time.monotonic)
    frame_id: int = 0
    # AutoSpeed vehicle boxes — CIPO/lead candidates (lead SELECTION is SP3).
    cipo_boxes: tuple[Detection, ...] = ()
    # AutoSteer ego-path polyline in raw image px, bottom-first; None if masked out.
    ego_path: tuple[tuple[float, float], ...] | None = None
    # AutoDrive scalars (domain-converted). None until the 2-frame buffer fills.
    cipo_distance_m: float | None = None
    road_curvature_1pm: float | None = None
    cipo_flag: bool | None = None
```

In the `Scene` dataclass, add these optional fields after `required_accel_mps2`:

```python
    # vision_pilot ONNX net outputs, folded from the PERCEPTION_EGO bundle by
    # SceneAssembler. All None when the nets are disabled (the default).
    ego_path: tuple[tuple[float, float], ...] | None = None
    road_curvature_1pm: float | None = None
    cipo_distance_m: float | None = None
    cipo_flag: bool | None = None
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_types.py -v`
Expected: PASS (existing + 4 new)

- [ ] **Step 6: Commit**

```bash
git add src/cruze/core/types.py src/cruze/core/bus.py tests/test_types.py
git commit -m "feat(core): add EgoEstimate type, Scene ego fields, PERCEPTION_EGO channel"
```

---

### Task 4: AutoSpeed estimator

**Files:**
- Create: `src/cruze/perception/backends/autospeed.py`
- Test: `tests/test_autospeed.py`

**Interfaces:**
- Consumes: `preprocess.decode_yolo`, `preprocess.nms`, `preprocess.unmap_point`; `OnnxSession`
- Produces: `AutoSpeedEstimator(model_path, provider="cpu", conf_threshold=0.6, iou_threshold=0.45, class_map=None, session=None)` with `.infer(chw, sx, sy, crop_top) -> tuple[Detection,...]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_autospeed.py
"""AutoSpeed decode/unmap with a fake session — no onnxruntime."""
import numpy as np
import pytest

from cruze.core.types import ObjectClass
from cruze.perception.backends.autospeed import AutoSpeedEstimator


class _FakeSession:
    def __init__(self, out):
        self._out = out
        self.input_names = ["images"]
        self.output_names = ["output"]

    def run(self, feeds):
        return [self._out]


def test_autospeed_decodes_and_unmaps():
    raw = np.zeros((1, 5, 1), dtype=np.float32)     # 4 box + 1 class
    raw[0, :4, 0] = [512, 256, 100, 80]             # cx,cy,w,h in net px
    raw[0, 4, 0] = 10.0                             # logit → keep
    est = AutoSpeedEstimator(model_path="", conf_threshold=0.5, session=_FakeSession(raw))
    dets = est.infer(np.zeros((1, 3, 512, 1024), np.float32), sx=1.25, sy=1.25, crop_top=80)
    assert len(dets) == 1
    d = dets[0]
    assert d.bbox.x1 == pytest.approx(462 * 1.25)
    assert d.bbox.y1 == pytest.approx(216 * 1.25 + 80)
    assert d.cls is ObjectClass.CAR


def test_autospeed_unknown_class_maps_to_unknown():
    raw = np.zeros((1, 5, 1), dtype=np.float32)
    raw[0, :4, 0] = [10, 10, 4, 4]
    raw[0, 4, 0] = 10.0
    est = AutoSpeedEstimator(model_path="", conf_threshold=0.5, class_map={}, session=_FakeSession(raw))
    dets = est.infer(np.zeros((1, 3, 512, 1024), np.float32), 1.0, 1.0, 0)
    assert dets[0].cls is ObjectClass.UNKNOWN
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_autospeed.py -v`
Expected: FAIL — `ModuleNotFoundError: ...backends.autospeed`

- [ ] **Step 3: Implement `autospeed.py`**

```python
# src/cruze/perception/backends/autospeed.py
"""AutoSpeed (vision_pilot) — YOLO-style vehicle detector, run alongside YOLO
as a CIPO/lead candidate source. Emits Detections in raw-frame pixels."""

from __future__ import annotations

import logging

import numpy as np

from cruze.core.types import BBox, Detection, ObjectClass
from cruze.perception import preprocess
from cruze.perception.onnx_runtime import OnnxSession

logger = logging.getLogger(__name__)

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
            out.append(Detection(BBox(x1, y1, x2, y2), float(scores[i]), cls))
        return tuple(out)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_autospeed.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/cruze/perception/backends/autospeed.py tests/test_autospeed.py
git commit -m "feat(perception): add AutoSpeed estimator"
```

---

### Task 5: AutoDrive estimator + homography loader

**Files:**
- Create: `src/cruze/perception/autodrive.py`
- Create: `calibration/vision_pilot_C.yaml` (placeholder identity homography)
- Modify: `models/README.md` (weights provenance)
- Test: `tests/test_autodrive.py`

**Interfaces:**
- Consumes: `preprocess.preprocess_bev`; `OnnxSession`
- Produces:
  - `load_homography(path:str) -> np.ndarray` (3x3)
  - `AutoDriveResult(cipo_distance_m:float, road_curvature_1pm:float, cipo_flag:bool)` (frozen dataclass)
  - `AutoDriveEstimator(model_path, homography, provider="cpu", curv_scale=1.0, flag_threshold=0.5, session=None)` with `.infer(image_bgr) -> AutoDriveResult | None`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_autodrive.py
"""AutoDrive buffer + domain conversion — cv2 warp monkeypatched away."""
import numpy as np
import pytest

import cruze.perception.autodrive as ad


class _FakeSession:
    def __init__(self, outputs):
        self._outputs = outputs
        self.input_names = ["prev", "curr"]
        self.output_names = ["dist", "curv", "flag"]

    def run(self, feeds):
        return self._outputs


def test_first_frame_returns_none_then_converts(monkeypatch):
    monkeypatch.setattr(ad.preprocess, "preprocess_bev",
                        lambda img, H: np.zeros((1, 3, 512, 1024), np.float32))
    fake = _FakeSession([np.array([[0.4]]), np.array([[0.02]]), np.array([[0.9]])])
    est = ad.AutoDriveEstimator(model_path="", homography=np.eye(3, dtype=np.float32),
                                curv_scale=2.0, flag_threshold=0.5, session=fake)
    img = np.zeros((720, 1280, 3), np.uint8)
    assert est.infer(img) is None                      # first frame buffers
    res = est.infer(img)
    assert res.cipo_distance_m == pytest.approx(150.0 * (1 - 0.4))  # 90.0
    assert res.road_curvature_1pm == pytest.approx(0.02 * 2.0)      # 0.04
    assert res.cipo_flag is True


def test_flag_below_threshold_is_false(monkeypatch):
    monkeypatch.setattr(ad.preprocess, "preprocess_bev",
                        lambda img, H: np.zeros((1, 3, 512, 1024), np.float32))
    fake = _FakeSession([np.array([[0.0]]), np.array([[0.0]]), np.array([[0.3]])])
    est = ad.AutoDriveEstimator(model_path="", homography=np.eye(3, dtype=np.float32), session=fake)
    img = np.zeros((720, 1280, 3), np.uint8)
    est.infer(img)
    assert est.infer(img).cipo_flag is False


def test_load_homography(tmp_path):
    p = tmp_path / "C.yaml"
    p.write_text("C: [1, 0, 0, 0, 1, 0, 0, 0, 1]")
    H = ad.load_homography(str(p))
    assert H.shape == (3, 3)
    assert H[0, 0] == pytest.approx(1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_autodrive.py -v`
Expected: FAIL — `ModuleNotFoundError: cruze.perception.autodrive`

- [ ] **Step 3: Implement `autodrive.py`**

```python
# src/cruze/perception/autodrive.py
"""AutoDrive (vision_pilot) — two-frame BEV net producing lead distance, road
curvature, and a CIPO in-path flag. Requires a camera-matched homography; see
the spec §11 — defaults OFF, the vendored C only fits vision_pilot's camera."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from cruze.perception import preprocess
from cruze.perception.onnx_runtime import OnnxSession

logger = logging.getLogger(__name__)

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
                 curv_scale: float = 1.0, flag_threshold: float = 0.5, session=None) -> None:
        self._homography = np.asarray(homography, dtype=np.float32)
        self._curv_scale = curv_scale
        self._flag_threshold = flag_threshold
        self._session = session if session is not None else OnnxSession(model_path, provider)
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
```

- [ ] **Step 4: Create the placeholder homography + weights provenance doc**

Create `calibration/vision_pilot_C.yaml`:

```yaml
# raw-pixel → 1024x512 BEV homography 'C' for AutoDrive.
# PLACEHOLDER identity — replace with vision_pilot's Calibration/homography_C_matrix.yaml
# (valid only for vision_pilot's camera) OR a Cruze recalibration. AutoDrive stays
# geometrically meaningless until this matches the deployed camera (spec §11).
C: [1, 0, 0, 0, 1, 0, 0, 0, 1]
```

Append to `models/README.md`:

```markdown
## vision_pilot ONNX weights (SP2)

AutoSpeed / AutoSteer / AutoDrive weights are vendored (Apache-2.0) in the
reference submodule. Copy them into `models/` (gitignored):

    cp reference/vision_pilot/VisionPilot/modules/models/weights/*.onnx models/

Upstream: autowarefoundation/{auto_speed,auto_steer,auto_drive}. `*_fp32.onnx`
for desktop; `*_int8.onnx` for edge (Jetson/Pi). AutoDrive additionally needs a
camera-matched homography in `calibration/vision_pilot_C.yaml`.
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_autodrive.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Commit**

```bash
git add src/cruze/perception/autodrive.py tests/test_autodrive.py calibration/vision_pilot_C.yaml models/README.md
git commit -m "feat(perception): add AutoDrive estimator + homography loader"
```

---

### Task 6: AutoSteer estimator

**Files:**
- Create: `src/cruze/perception/autosteer.py`
- Test: `tests/test_autosteer.py`

**Interfaces:**
- Consumes: `preprocess.NET_W`, `preprocess.NET_H`, `preprocess.unmap_point`; `OnnxSession`
- Produces: `AutoSteerEstimator(model_path, provider="cpu", session=None)` with `.infer(chw, sx, sy, crop_top) -> tuple[tuple[float,float],...] | None`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_autosteer.py
"""AutoSteer ego-path masking + mapping with a fake session."""
import numpy as np
import pytest

from cruze.perception.autosteer import AutoSteerEstimator


class _FakeSession:
    def __init__(self, xp, h):
        self._out = [xp, h]
        self.input_names = ["input"]
        self.output_names = ["xp", "h"]

    def run(self, feeds):
        return self._out


def test_autosteer_keeps_masked_points_and_maps():
    xp = np.full(64, 0.5, np.float32)          # centre column → u = 512
    h = np.zeros(64, np.float32); h[0] = 1.0   # only first sample passes mask
    est = AutoSteerEstimator(model_path="", session=_FakeSession(xp, h))
    path = est.infer(np.zeros((1, 3, 512, 1024), np.float32), sx=1.25, sy=1.25, crop_top=80)
    assert path is not None and len(path) == 1
    assert path[0][0] == pytest.approx(512 * 1.25)   # u=0.5*1024 → raw x
    assert path[0][1] == pytest.approx(0 * 1.25 + 80)  # row 0 → raw y


def test_autosteer_none_when_all_masked():
    est = AutoSteerEstimator(model_path="", session=_FakeSession(
        np.full(64, 0.5, np.float32), np.zeros(64, np.float32)))
    assert est.infer(np.zeros((1, 3, 512, 1024), np.float32), 1.25, 1.25, 80) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_autosteer.py -v`
Expected: FAIL — `ModuleNotFoundError: cruze.perception.autosteer`

- [ ] **Step 3: Implement `autosteer.py`**

```python
# src/cruze/perception/autosteer.py
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
    def __init__(self, model_path: str, provider: str = "cpu", session=None) -> None:
        self._session = session if session is not None else OnnxSession(model_path, provider)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_autosteer.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/cruze/perception/autosteer.py tests/test_autosteer.py
git commit -m "feat(perception): add AutoSteer estimator"
```

---

### Task 7: Config fields + hardware profile

**Files:**
- Modify: `src/cruze/core/config.py` (`PerceptionConfig`)
- Modify: `config/default.yaml` (document new keys under `perception:`)
- Create: `config/hardware/vision_pilot.yaml`
- Test: `tests/test_config.py` (create if absent)

**Interfaces:**
- Produces: `PerceptionConfig` fields — `onnx_provider`, `autospeed_enabled`, `autosteer_enabled`, `autodrive_enabled`, `autospeed_model_path`, `autosteer_model_path`, `autodrive_model_path`, `autospeed_conf_threshold`, `autospeed_iou_threshold`, `autodrive_homography_path`, `autodrive_curv_scale`, `autodrive_flag_threshold`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_config.py  (create or append)
import pathlib
import yaml

from cruze.core.config import PerceptionConfig


def test_vision_net_defaults_off():
    c = PerceptionConfig()
    assert c.autospeed_enabled is False
    assert c.autosteer_enabled is False
    assert c.autodrive_enabled is False
    assert c.onnx_provider == "cpu"
    assert c.autospeed_model_path.endswith("autospeed_fp32.onnx")
    assert c.autospeed_conf_threshold == 0.6
    assert c.autospeed_iou_threshold == 0.45


def test_vision_pilot_profile_enables_all_three():
    data = yaml.safe_load(pathlib.Path("config/hardware/vision_pilot.yaml").read_text())
    p = data["perception"]
    assert p["autospeed_enabled"] is True
    assert p["autosteer_enabled"] is True
    assert p["autodrive_enabled"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL — `AttributeError: ... has no attribute 'autospeed_enabled'` and missing profile file

- [ ] **Step 3: Add fields to `PerceptionConfig`**

In `src/cruze/core/config.py`, inside `PerceptionConfig`, append after `camera_pitch_deg`:

```python
    # --- vision_pilot ONNX nets (SP2). All OFF by default; a hardware profile
    # (config/hardware/vision_pilot.yaml) turns them on. ---
    onnx_provider: str = "cpu"           # cpu | cuda | tensorrt (shared EP)
    autospeed_enabled: bool = False
    autosteer_enabled: bool = False
    autodrive_enabled: bool = False
    autospeed_model_path: str = "models/autospeed_fp32.onnx"
    autosteer_model_path: str = "models/autosteer_fp32.onnx"
    autodrive_model_path: str = "models/autodrive_fp32.onnx"
    autospeed_conf_threshold: float = 0.6   # vision_pilot default
    autospeed_iou_threshold: float = 0.45   # vision_pilot NMS default
    # AutoDrive BEV homography (raw px → 1024x512). Camera-specific — see spec §11.
    autodrive_homography_path: str = "calibration/vision_pilot_C.yaml"
    # raw curvature → 1/m. Empirical scale (pin exact value from upstream).
    autodrive_curv_scale: float = 1.0
    autodrive_flag_threshold: float = 0.5   # CIPO in-path probability cutoff
```

- [ ] **Step 4: Document keys in `config/default.yaml`**

Under the existing `perception:` section, add:

```yaml
  # vision_pilot ONNX nets (SP2) — all off by default; see config/hardware/vision_pilot.yaml.
  onnx_provider: cpu            # cpu | cuda | tensorrt
  autospeed_enabled: false
  autosteer_enabled: false
  autodrive_enabled: false
  autospeed_model_path: models/autospeed_fp32.onnx
  autosteer_model_path: models/autosteer_fp32.onnx
  autodrive_model_path: models/autodrive_fp32.onnx
  autospeed_conf_threshold: 0.6
  autospeed_iou_threshold: 0.45
  autodrive_homography_path: calibration/vision_pilot_C.yaml
  autodrive_curv_scale: 1.0
  autodrive_flag_threshold: 0.5
```

- [ ] **Step 5: Create `config/hardware/vision_pilot.yaml`**

```yaml
# "Every aspect of vision_pilot" profile — enables all three ONNX nets (fp32).
# AutoDrive needs a camera-matched homography (calibration/vision_pilot_C.yaml);
# the vendored C fits vision_pilot's camera only (spec §11). For edge hardware,
# point the *_model_path keys at the *_int8.onnx weights and set onnx_provider:
# tensorrt.
perception:
  onnx_provider: cpu
  autospeed_enabled: true
  autosteer_enabled: true
  autodrive_enabled: true
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_config.py -v`
Expected: PASS (2 passed)

- [ ] **Step 7: Commit**

```bash
git add src/cruze/core/config.py config/default.yaml config/hardware/vision_pilot.yaml tests/test_config.py
git commit -m "feat(config): add vision_pilot ONNX net settings + hardware profile"
```

---

### Task 8: VisionNets holder + PerceptionService wiring

**Files:**
- Create: `src/cruze/perception/vision_nets.py`
- Modify: `src/cruze/perception/pipeline.py` (`PerceptionService.__init__`, `_analyze`, `_process_frame`)
- Modify: `src/cruze/orchestrator.py:83`
- Test: `tests/test_vision_nets.py`

**Interfaces:**
- Consumes: `EgoEstimate`, `Frame`, `preprocess.preprocess_crop2_1`; the three estimators; `Channel.PERCEPTION_EGO`
- Produces:
  - `VisionNets(autospeed=None, autosteer=None, autodrive=None)` with `.any_enabled: bool` and `.infer(frame:Frame) -> EgoEstimate | None`
  - `build_vision_nets(cfg) -> VisionNets`
  - `PerceptionService(cfg, bus, detector, vision_nets=None)` now publishes `PERCEPTION_EGO`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_vision_nets.py
import numpy as np
import pytest

import cruze.perception.vision_nets as vn
from cruze.core.types import BBox, Detection, Frame, ObjectClass


class _FakeAutoSpeed:
    def infer(self, chw, sx, sy, crop_top):
        return (Detection(BBox(0, 0, 10, 10), 0.9, ObjectClass.CAR),)


def test_infer_none_when_no_nets():
    assert vn.VisionNets().infer(Frame(image=None)) is None
    assert vn.VisionNets().any_enabled is False


def test_infer_bundles_autospeed(monkeypatch):
    monkeypatch.setattr(vn.preprocess, "preprocess_crop2_1",
                        lambda img: (np.zeros((1, 3, 512, 1024), np.float32), 1.25, 1.25, 80))
    nets = vn.VisionNets(autospeed=_FakeAutoSpeed())
    frame = Frame(image=np.zeros((720, 1280, 3), np.uint8), frame_id=7)
    ego = nets.infer(frame)
    assert ego is not None
    assert ego.frame_id == 7
    assert len(ego.cipo_boxes) == 1
    assert ego.cipo_boxes[0].cls is ObjectClass.CAR


@pytest.mark.asyncio
async def test_perception_publishes_ego():
    from cruze.core.bus import Channel, EventBus
    from cruze.core.config import Config
    from cruze.core.types import EgoEstimate
    from cruze.perception import detector as det_mod
    from cruze.perception.pipeline import PerceptionService

    class _FakeNets:
        any_enabled = True
        def infer(self, frame):
            return EgoEstimate(frame_id=frame.frame_id, cipo_flag=True)

    bus = EventBus()
    svc = PerceptionService(Config(), bus, det_mod.load("stub"), vision_nets=_FakeNets())
    ego_q = bus.subscribe(Channel.PERCEPTION_EGO, maxsize=2)
    await svc._process_frame(Frame(image=np.zeros((720, 1280, 3), np.uint8), frame_id=3))
    ego = ego_q.get_nowait()
    assert ego.frame_id == 3 and ego.cipo_flag is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_vision_nets.py -v`
Expected: FAIL — `ModuleNotFoundError: cruze.perception.vision_nets`

- [ ] **Step 3: Implement `vision_nets.py`**

```python
# src/cruze/perception/vision_nets.py
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

    @property
    def any_enabled(self) -> bool:
        return any((self._autospeed, self._autosteer, self._autodrive))

    def infer(self, frame: Frame) -> EgoEstimate | None:
        if not self.any_enabled:
            return None
        cipo_boxes: tuple = ()
        ego_path = None
        dist = curv = flag = None

        if self._autospeed is not None or self._autosteer is not None:
            chw, sx, sy, crop_top = preprocess.preprocess_crop2_1(frame.image)
            if self._autospeed is not None:
                cipo_boxes = self._autospeed.infer(chw, sx, sy, crop_top)
            if self._autosteer is not None:
                ego_path = self._autosteer.infer(chw, sx, sy, crop_top)

        if self._autodrive is not None:
            res = self._autodrive.infer(frame.image)
            if res is not None:
                dist, curv, flag = res.cipo_distance_m, res.road_curvature_1pm, res.cipo_flag

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
```

- [ ] **Step 4: Wire `PerceptionService`**

In `src/cruze/perception/pipeline.py`:

4a. Add the import near the top (with the other core imports):

```python
from cruze.core.types import Detection, EgoEstimate, Frame, Lanes
```

4b. Change `__init__` signature and store the holder. Replace the `def __init__(self, cfg, bus, detector):` line and set-up:

```python
    def __init__(self, cfg: "Config", bus: EventBus, detector: Detector,
                 vision_nets=None) -> None:
        self._cfg = cfg
        self._bus = bus
        self._detector = detector
        self._vision_nets = vision_nets
```

(Leave the rest of `__init__` — tracker, kalman, metrics — unchanged.)

4c. Change `_analyze` to also run the nets and return the bundle:

```python
    def _analyze(self, frame: Frame):
        """All CPU-bound classical/ML work for one frame — runs in the executor."""
        detections = self._detector.detect(frame)
        detections = lights_mod.annotate_lights(detections, frame.image)
        lane_result = (
            lane_mod.detect_lanes(frame.image)
            if self._cfg.perception.lane_detection_enabled
            else None
        )
        ego = self._vision_nets.infer(frame) if self._vision_nets is not None else None
        return detections, lane_result, ego
```

4d. In `_process_frame`, update the unpack and publish the bundle. Change the executor call line to:

```python
        detections, lane_result, ego = await asyncio.get_event_loop().run_in_executor(
            None, self._analyze, frame
        )
```

and add, right after `await self._bus.publish(Channel.PERCEPTION_TRACKS, tracks)`:

```python
        if ego is not None:
            await self._bus.publish(Channel.PERCEPTION_EGO, ego)
```

- [ ] **Step 5: Wire the orchestrator**

In `src/cruze/orchestrator.py`, add the import (near line 34):

```python
from cruze.perception.vision_nets import build_vision_nets
```

and change line 83 from `perception_svc = PerceptionService(cfg, bus, detector)` to:

```python
        perception_svc = PerceptionService(cfg, bus, detector, vision_nets=build_vision_nets(cfg))
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_vision_nets.py -v`
Expected: PASS (3 passed)

- [ ] **Step 7: Run the full suite to confirm no regressions**

Run: `python -m pytest tests/ -q`
Expected: PASS (all existing + new)

- [ ] **Step 8: Commit**

```bash
git add src/cruze/perception/vision_nets.py src/cruze/perception/pipeline.py src/cruze/orchestrator.py tests/test_vision_nets.py
git commit -m "feat(perception): wire VisionNets into pipeline, publish PERCEPTION_EGO"
```

---

### Task 9: SceneAssembler folds the ego bundle

**Files:**
- Modify: `src/cruze/reasoning/scene.py`
- Test: `tests/test_scene.py` (extend)

**Interfaces:**
- Consumes: `Channel.PERCEPTION_EGO`, `EgoEstimate`
- Produces: `Scene.ego_path`, `Scene.road_curvature_1pm`, `Scene.cipo_distance_m`, `Scene.cipo_flag` populated when a fresh `EgoEstimate` is present

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_scene.py  (append)
import time

from cruze.core.bus import EventBus
from cruze.core.config import Config
from cruze.core.types import EgoEstimate
from cruze.reasoning.scene import SceneAssembler


def test_scene_folds_fresh_ego():
    asm = SceneAssembler(Config(), EventBus(), image_width=1280)
    asm._latest_ego = EgoEstimate(
        cipo_distance_m=42.0, road_curvature_1pm=0.01, cipo_flag=True, ego_path=((1.0, 2.0),))
    asm._ego_seen_at = time.monotonic()
    scene = asm._build_scene([])
    assert scene.cipo_distance_m == 42.0
    assert scene.road_curvature_1pm == 0.01
    assert scene.cipo_flag is True
    assert scene.ego_path == ((1.0, 2.0),)


def test_scene_drops_stale_ego():
    asm = SceneAssembler(Config(), EventBus(), image_width=1280)
    asm._latest_ego = EgoEstimate(cipo_distance_m=42.0)
    asm._ego_seen_at = time.monotonic() - 5.0   # older than the freshness window
    scene = asm._build_scene([])
    assert scene.cipo_distance_m is None
    assert scene.cipo_flag is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_scene.py -k ego -v`
Expected: FAIL — `AttributeError: 'SceneAssembler' object has no attribute '_latest_ego'`

- [ ] **Step 3: Implement the fold in `scene.py`**

3a. Add the import: change the types import line to include `EgoEstimate`:

```python
from cruze.core.types import EgoEstimate, Lanes, ObjectClass, Scene, Track, VehicleState
```

3b. Add a freshness constant next to `_LANES_MAX_AGE_S`:

```python
# EgoEstimate older than this is dropped from the Scene — same rationale and
# window as lanes: stale neural-net geometry would mislead downstream.
_EGO_MAX_AGE_S = 0.5
```

3c. In `__init__`, after `self._lanes_seen_at = 0.0`, add:

```python
        self._latest_ego: EgoEstimate | None = None
        self._ego_seen_at = 0.0  # monotonic arrival time of the last EgoEstimate
```

3d. In `run`, subscribe alongside the others (after the `lanes_q` line):

```python
        ego_q = self._bus.subscribe(Channel.PERCEPTION_EGO, maxsize=2)
```

3e. In the inner `drain_aux`, add an ego drain alongside the lanes drain (inside the `while self._running:` loop, before the `if not drained` check):

```python
                if not ego_q.empty():
                    self._latest_ego = ego_q.get_nowait()
                    self._ego_seen_at = time.monotonic()
                    drained = True
```

3f. In `_build_scene`, fold the ego fields. Replace the `scene = Scene(...)` construction with:

```python
        ego = self._latest_ego
        if ego is not None and now - self._ego_seen_at > _EGO_MAX_AGE_S:
            ego = None
        scene = Scene(
            timestamp=now,
            tracks=tuple(tracks),
            vehicle_state=self._latest_vehicle_state,
            lead_track=lead,
            lanes=lanes,
            ego_path=ego.ego_path if ego else None,
            road_curvature_1pm=ego.road_curvature_1pm if ego else None,
            cipo_distance_m=ego.cipo_distance_m if ego else None,
            cipo_flag=ego.cipo_flag if ego else None,
        )
```

(The trailing `dataclasses.replace(scene, required_accel_mps2=...)` stays as-is and preserves the new fields.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_scene.py -v`
Expected: PASS (existing + 2 new)

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest tests/ -q`
Expected: PASS (all)

- [ ] **Step 6: Commit**

```bash
git add src/cruze/reasoning/scene.py tests/test_scene.py
git commit -m "feat(reasoning): fold EgoEstimate onto Scene in SceneAssembler"
```

---

## Self-Review

**1. Spec coverage:**
- §6.1 ONNX engine → Task 2 ✓ · §6.2 preprocess → Task 1 ✓ · §6.3 AutoSpeed → Task 4 ✓ · §6.4 AutoSteer/AutoDrive bundle → Tasks 5,6 + type in Task 3 ✓ · §6.5 AutoDrive → Task 5 ✓ · §6.6 type+channel → Task 3 ✓ · §6.7 PerceptionService wiring → Task 8 ✓ · §6.8 SceneAssembler fold → Task 9 ✓ · §7 config+profile → Task 7 ✓ · §8 weights/README → Task 5 step 4 ✓ · §9 deps → Task 2 step 4 ✓ · §10 tests → every task ✓ · §11 caveats → homography placeholder (Task 5) + AutoSpeed class map comment (Task 4) ✓ · §12 build order → task order matches ✓ · §13 additive → only optional fields + one channel ✓.
- No `serialize.py` task — correct, web deferred (spec §4 note).

**2. Placeholder scan:** No "TBD/TODO-implement-later" gaps. The two spec-acknowledged unknowns (AutoSpeed taxonomy, homography values) ship as a documented default map + a documented placeholder YAML with AutoDrive off by default — these are intended states, not plan gaps.

**3. Type consistency:** `preprocess.unmap_point`, `decode_yolo`, `nms`, `chw_from_rgb01` signatures identical across Tasks 1/4/6. `OnnxSession.input_names/output_names/run` consistent Tasks 2/4/5/6. `EgoEstimate` field names identical in Tasks 3/8/9. `VisionNets.infer`/`any_enabled` consistent Tasks 8/9-callers. `AutoDriveResult` fields (`cipo_distance_m`, `road_curvature_1pm`, `cipo_flag`) consistent Tasks 5/8. Estimator `.infer(chw, sx, sy, crop_top)` signature matches the `VisionNets` call site.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-16-sp2-vision-pilot-onnx-models.md`.
