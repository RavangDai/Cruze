# SP2 — Port vision_pilot's ONNX perception models into Cruze

**Status:** Design (approved to draft) · **Date:** 2026-07-16
**Part of:** the vision_pilot full-port effort. Companion doc: [`docs/vision_pilot-mapping.md`](../../vision_pilot-mapping.md).

This is **sub-project 2 of 4**. The full port decomposes into:

| # | Sub-project | State |
|---|---|---|
| SP1 | Extend the data contract (`types.py`, `bus.py`) | folded into SP2 where needed |
| **SP2** | **ONNX model backends (this doc)** | **now** |
| SP3 | Bayesian fusion upgrades (CIPO particle filter, lateral RANSAC/Kalman → yaw/curvature smoothing) | later |
| SP4 | ~~Web dashboard redesign~~ | **deferred** — running on-device, not web, for now |

> **Runtime target: on-device, no web.** The web dashboard (SP4) is deferred. On-device
> visualization, if/when wanted, is the existing OpenCV HUD (`hmi/hud.py`, vision_pilot's
> `debug_draw` analogue) reading `Scene` directly — no JSON serialization needed. SP2's new
> `Scene` fields therefore feed **SP3 fusion + an on-device HUD**, not a browser. On-device
> also means edge hardware (Jetson/Pi): use `onnx_provider: tensorrt` (or `cuda`) + the
> `*_int8.onnx` weights via a hardware profile — already covered by §7's provider/weights config.

---

## 1. Goal

Run vision_pilot's three ONNX nets — **AutoSpeed**, **AutoSteer**, **AutoDrive** — as native, config-selectable Cruze components, emitting their **raw** outputs into the data contract so SP3 (fusion) and SP4 (dashboard) can consume them.

The three nets are the part of vision_pilot that is genuinely deeper than Cruze's perception (purpose-trained for driving). SP2 makes them available inside Cruze without disturbing anything that already works.

## 2. Non-goals (the SP2 boundary)

SP2 stops at **raw model outputs + the naive domain conversions vision_pilot itself documents**. The following are explicitly **out of scope** and belong to SP3/SP4:

- CIPO longitudinal **particle filter**; lead *selection* by fusing AutoSpeed boxes + AutoDrive flag + tracker.
- Lateral **RANSAC** path fit, **yaw** estimation, dual-Kalman smoothing.
- Steering-wheel widget, curvature/yaw HUD readouts, any `hmi/` change.
- BEV homography **recalibration** for Cruze's own camera (see §11).

## 3. Design principles honoured

- **Bus only** — the nets publish onto a channel; no module imports another (CLAUDE.md hard rule).
- **Hardware = config** — every net is opt-in via YAML; a `vision_pilot.yaml` profile turns them on. Execution provider is config-driven.
- **Graceful degradation** — missing `onnxruntime`, missing weights, an unavailable EP, or a first frame with no history all log-and-continue; they never crash the loop.
- **Additive contract** — only **new optional** fields on `types.py` and **one new channel**. Nothing renamed or removed.
- **No ML deps in tests** — all real logic lives in pure-numpy helpers; the net wrappers are tested with fake sessions.

---

## 4. Background: exact model contracts (from the vendored source)

Verified against `reference/vision_pilot/VisionPilot/modules/models/*` and `.../sensing/image_preprocessing/*`. **The runtime pipeline (`inference.cpp` + `image_preprocessor.cpp`) is authoritative over the header comments**, which are simplified and, for AutoSpeed, misleading.

All three nets take a `1×3×512×1024` (NCHW) float32 tensor, RGB, at net size **W=1024 H=512**.

### 4.1 The two preprocessed images (`ImagePreprocessor::preprocess`)

From one raw frame the preprocessor derives **two** 1024×512 images:

1. **`warped`** — `warpPerspective(raw, C, (1024,512))`, a **BEV** view via homography matrix `C` (loaded from `homography_C_matrix.yaml`). Border mode `BORDER_REFLECT_101`.
2. **`resized`** — **top-crop to 2:1 aspect, then resize** to 1024×512:
   ```
   crop_top = max(0, round(H_raw − W_raw / 2))      # drop the sky, keep bottom W/2 rows
   resized  = resize( raw[crop_top:H_raw, 0:W_raw], (1024, 512) )
   ```

### 4.2 Which net eats which image

From `inference.cpp::process(warped, resized)`:

| Net | Spatial input | Normalisation | Frames |
|---|---|---|---|
| **AutoDrive** | `warped` (BEV) | ImageNet mean/std | **two** (prev + curr) |
| **AutoSteer** | `resized` (crop-2:1) | `/255` only | one (curr) |
| **AutoSpeed** | `resized` (crop-2:1, *same buffer as AutoSteer*) | `/255` only | one (curr) |

ImageNet: `mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`, applied after `/255`, per channel.

> **Correction vs. the brainstorm:** AutoSpeed is **not** letterboxed and AutoDrive is **not** a plain resize. AutoSpeed/AutoSteer share the crop-2:1 image; AutoDrive uses the BEV warp. The `preprocess.py` port must implement crop-2:1 and the homography warp, **not** letterbox.

### 4.3 Outputs

- **AutoSpeed** → raw `[1, C, N]`, `C = 4 + K`. Decode per anchor `n`: `cx,cy,w,h = data[0..3, n]`; class prob `= sigmoid(data[4+c, n])`; keep argmax-class if `> conf_thres`; corner box `= (cx±w/2, cy±h/2)`; then **NMS** at `iou_thres`. Boxes are in **1024×512 (resized) space** — caller reverses the crop+resize (§6.2).
  - `conf_thres = 0.6`, `iou_thres = 0.45` (vision_pilot defaults).
  - **Class taxonomy is unknown** — the visualizer only labels `L1/L2/L3`. Resolved at implementation (§11).
- **AutoSteer** → `xp[64]`, `h_vector[64]`. Ego-path sample `i`: row `v = linspace(0, 511, 64)[i]`, col `u = xp[i]·1024`; **keep only where `h_vector[i] ≥ 0.5`**. In resized space — caller reverses crop+resize.
- **AutoDrive** → `dist_normalized`, `curvature_raw`, `flag_prob`. Domain conversion (vision_pilot):
  ```
  cipo_distance_m   = 150.0 · (1.0 − dist_normalized)     # D_MAX_M = 150 (empirical, vision_pilot)
  road_curvature_1pm = curvature_raw · curv_scale         # curv_scale from config
  cipo_flag          = flag_prob ≥ flag_threshold         # CIPO in-path probability
  ```

### 4.4 Resized-space → raw-frame un-mapping (for AutoSpeed boxes & AutoSteer path)

Reverses the crop-2:1 + resize so overlays land on Cruze's original frame (`sx = W_raw/1024`, `sy = (H_raw − crop_top)/512`):

```
x_raw = x_resized · sx
y_raw = y_resized · sy + crop_top
```

---

## 5. Architecture — how it fits Cruze

All three nets become **optional estimators layered on top of the existing pipeline**. YOLO stays the primary `Detector`, unchanged; lights/signs/pedestrians keep working. AutoSpeed runs *alongside* YOLO as a CIPO/lead detector (its boxes are carried as candidates; lead selection is SP3).

```
PerceptionService._analyze(frame)                     [executor thread]
  ├─ YOLO.detect         → list[Detection]   (unchanged → PERCEPTION_DETECTIONS)
  ├─ lights, lanes       (unchanged)
  └─ VisionNets.infer(frame)  (new, when enabled)
        ├─ AutoSpeed  → cipo_boxes (crop-2:1 preproc, shared)
        ├─ AutoSteer  → ego_path   (crop-2:1 preproc, shared)
        └─ AutoDrive  → distance / curvature / flag (BEV warp, 2-frame buffer)
                              │
                              ▼  EgoEstimate  → Channel.PERCEPTION_EGO
SceneAssembler (drain_aux) subscribes PERCEPTION_EGO, folds onto new optional Scene fields.
```

The three nets share one preprocessing of the crop-2:1 image (AutoSpeed + AutoSteer) and one BEV warp (AutoDrive), computed once per frame.

---

## 6. Component specs

### 6.1 `perception/onnx_runtime.py` — shared ONNX engine

`class OnnxSession` wrapping `onnxruntime.InferenceSession`.

- Guarded import → `ImportError("onnxruntime required. Install with: pip install 'cruze[onnx]'")`.
- `__init__(model_path, provider="cpu")` — provider mapped: `cpu→CPUExecutionProvider`, `cuda→CUDAExecutionProvider`, `tensorrt→[TensorrtExecutionProvider, CUDA, CPU]`. **If the requested EP is unavailable, log a warning and fall back to CPU** (never raise).
- Introspects and stores input/output names from the session (models have arbitrary tensor names — never hard-code).
- `run(feeds: dict[str, np.ndarray]) -> list[np.ndarray]`.
- **Pure/testable seam:** `select_providers(provider, available) -> list[str]` is a free function (no onnxruntime import) → unit-tested directly.

### 6.2 `perception/preprocess.py` — pure numpy (the real test surface)

Free functions, numpy only (numpy is allowed in tests):

- `top_crop_2_1(h, w) -> crop_top` — `max(0, round(h - w/2))`.
- `preprocess_crop2_1(image_bgr) -> (chw_f32, sx, sy, crop_top)` — crop-2:1 → resize 1024×512 → BGR→RGB → `/255` → HWC→CHW. Returns the transform params for un-mapping.
- `preprocess_bev(image_bgr, C) -> chw_f32` — `warpPerspective(·, C, (1024,512), BORDER_REFLECT_101)` → RGB → `/255` → **ImageNet norm** → CHW. (cv2 guarded; if cv2 absent, AutoDrive stays disabled — graceful.)
- `decode_yolo(raw[1,C,N], conf_thres) -> (boxes, scores, class_ids)` — sigmoid + threshold + argmax + corner conversion.
- `nms(boxes, scores, iou_thres) -> keep_idx`.
- `unmap_point(x, y, sx, sy, crop_top) -> (x_raw, y_raw)` and `unmap_boxes(...)`.

### 6.3 `perception/backends/autospeed.py` — `AutoSpeedEstimator`

- `__init__(model_path, provider, conf_thres=0.6, iou_thres=0.45, class_map=_AUTOSPEED_CLASSES)`.
- `infer(frame, chw, sx, sy, crop_top) -> tuple[Detection, ...]` — accepts the **shared** crop-2:1 tensor (computed once by the service), runs the session, `decode_yolo` + `nms`, `unmap_boxes` to raw frame, maps `class_id → ObjectClass`, wraps as `Detection`. Unknown ids → `ObjectClass.UNKNOWN`.
- **Not** registered in `detector.load()` — it is an estimator, not the primary detector.

### 6.4 `perception/autosteer.py` — `AutoSteerEstimator`

- `infer(frame, chw, sx, sy, crop_top) -> tuple[tuple[float,float], ...] | None` — shared crop-2:1 tensor → `xp,h_vector` → build path points where `h_vector ≥ 0.5`, `u = xp·1024`, `v = linspace(0,511,64)`, un-map to raw px. `None`/empty when nothing passes the mask.

### 6.5 `perception/autodrive.py` — `AutoDriveEstimator`

- Holds a **one-frame history buffer** of the previous BEV tensor.
- `infer(frame) -> AutoDriveResult | None` — computes the BEV tensor (`preprocess_bev`); on the **first** frame stores it and returns `None` (graceful); thereafter runs `session.run(prev, curr)`, applies the §4.3 domain conversions, returns `AutoDriveResult(cipo_distance_m, road_curvature_1pm, cipo_flag)`.
- Needs the homography `C` (config, §7) and cv2. Disabled cleanly if either is missing.

### 6.6 New type + channel (folded SP1)

`core/bus.py`: add `PERCEPTION_EGO = "perception.ego"`.

`core/types.py` — one new frozen dataclass + new optional `Scene` fields (never rename/remove):

```python
@dataclass(frozen=True)
class EgoEstimate:
    """Per-frame bundle of vision_pilot's ONNX net outputs (raw, pre-fusion).
    Mirrors vision_pilot's InferenceFrameResult. Any field may be None when its
    net is disabled or produced no output this frame."""
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

# Scene gains (optional, default None):
#   ego_path:            tuple[tuple[float, float], ...] | None = None
#   road_curvature_1pm:  float | None = None
#   cipo_distance_m:     float | None = None
#   cipo_flag:           bool | None = None
```

### 6.7 `PerceptionService` wiring

- New collaborator `VisionNets` (small holder that owns the three estimators + shared preprocessing), constructed from config; each estimator independently on/off.
- In `_analyze` (executor): when any net is enabled, compute the shared crop-2:1 tensor once, run enabled estimators, build an `EgoEstimate`.
- In `_process_frame`: `await bus.publish(Channel.PERCEPTION_EGO, ego)` when produced. No change to the existing DETECTIONS/LANES/TRACKS publishes.
- Latency: nets run in the existing executor hop (never block the loop). Over-budget still only logs (advisory budget rule).

### 6.8 `SceneAssembler` fold

- Subscribe `PERCEPTION_EGO` (maxsize 2) in `run`; drain in the existing `drain_aux` with a freshness window (reuse the `_LANES_MAX_AGE_S = 0.5 s` pattern → `_ego_seen_at`).
- In `_build_scene`, copy fresh `EgoEstimate` fields onto the new optional `Scene` fields via `dataclasses.replace`. Stale → treated as absent (`None`).

---

## 7. Config additions (`core/config.py` `PerceptionConfig`)

```python
onnx_provider: str = "cpu"                # cpu | cuda | tensorrt (shared EP)
autospeed_enabled: bool = False
autosteer_enabled: bool = False
autodrive_enabled: bool = False
autospeed_model_path: str = "models/autospeed_fp32.onnx"
autosteer_model_path: str = "models/autosteer_fp32.onnx"
autodrive_model_path: str = "models/autodrive_fp32.onnx"
autospeed_conf_threshold: float = 0.6     # vision_pilot default
autospeed_iou_threshold: float = 0.45     # vision_pilot default
# AutoDrive BEV homography (raw px → 1024×512 BEV). Camera-specific — see §11.
autodrive_homography_path: str = "calibration/vision_pilot_C.yaml"
# raw curvature → 1/m. Empirical scale from vision_pilot's vehicle config.
autodrive_curv_scale: float = 1.0         # TODO: pin exact value from upstream at impl
autodrive_flag_threshold: float = 0.5     # CIPO in-path probability cutoff
```

New hardware profile **`config/hardware/vision_pilot.yaml`** — enables all three (fp32), documents that AutoDrive requires a matching homography. Edge profiles may point the paths at the `*_int8.onnx` weights.

## 8. Weights & provenance

The six `.onnx` files live in the submodule (`reference/vision_pilot/.../weights/`), Apache-2.0, fp32 + int8. `models/` is gitignored. Document in `models/README.md`:

- Provenance + licence + upstream repos (`auto_speed`, `auto_steer`, `auto_drive`).
- A copy step (`cp reference/vision_pilot/VisionPilot/modules/models/weights/*.onnx models/`) — no build coupling to the submodule (it stays reference-only).
- The vision_pilot homography `C` extracted to `calibration/vision_pilot_C.yaml`.

## 9. Dependencies (`pyproject.toml`)

New optional extra `onnx = ["onnxruntime"]` (and a note that `onnxruntime-gpu` replaces it for `cuda`/`tensorrt`). `cv2` is already an optional perception dep (used by lanes/lights). Tests import neither.

## 10. Testing plan (all ML-dep-free)

| File | Covers |
|---|---|
| `tests/test_preprocess.py` | `top_crop_2_1` formula; crop-2:1 shape + `sx/sy/crop_top`; un-map round-trips a known point; ImageNet norm values; `decode_yolo` on synthetic `[1,C,N]`; `nms` suppression |
| `tests/test_autospeed.py` | decode→class-map→un-map to raw px with a **fake session** (canned array); unknown id → UNKNOWN |
| `tests/test_autodrive.py` | history buffer: first frame → `None`; domain conversion math; disabled when cv2/homography missing |
| `tests/test_autosteer.py` | `h_vector < 0.5` dropped; `u/v` mapping; empty → `None` |
| `tests/test_onnx_runtime.py` | `select_providers` fallback logic (no onnxruntime import) |
| extend `tests/test_scene.py` | `EgoEstimate` fold onto `Scene`; staleness → `None` |

Fake session = a tiny object with `.run()` returning canned numpy arrays and the same in/out-name surface — no `onnxruntime` needed.

## 11. Known caveats / prerequisites

1. **AutoDrive is only geometrically valid with a homography matching the camera.** The vendored `C` is calibrated for vision_pilot's camera. On Cruze's own camera, AutoDrive distance/curvature are meaningless until recalibrated (the `Calibration/calc_front_camera_homography.py` path — a separate follow-up, mapping-doc §7.4). Therefore AutoDrive defaults **off**; `vision_pilot.yaml` enables it for replay of vision_pilot-style footage only. This is documented, not silently wrong.
2. **AutoSpeed class taxonomy** — resolved at implementation by reading ONNX metadata (`C−4 = K`) + the upstream `auto_speed` repo; encoded as `_AUTOSPEED_CLASSES`, unknowns → `UNKNOWN`. If it turns out to encode lane-tier (L1/L2/L3) rather than object categories, the map instead marks them all `CAR` with the tier kept in a comment — decided once the taxonomy is known.
3. **`autodrive_curv_scale`** — exact empirical value to be pinned from upstream vehicle config at implementation (placeholder `1.0`).

## 12. Build order (for the implementation plan)

1. `onnx_runtime.py` + `preprocess.py` (+ their tests) — no Cruze wiring yet.
2. `types.py` (`EgoEstimate` + Scene fields) + `bus.py` (`PERCEPTION_EGO`). (No `serialize.py` change — web is deferred; on-device HUD reads `Scene` directly.)
3. `backends/autospeed.py` + test.
4. `autodrive.py` (+ homography load, buffer) + test.
5. `autosteer.py` + test.
6. `VisionNets` holder + `PerceptionService` wiring + `SceneAssembler` fold + `test_scene.py` extension.
7. `PerceptionConfig` fields + `config/hardware/vision_pilot.yaml` + `models/README.md` provenance + `pyproject` extra.

## 13. Data-contract compliance

Only **additive** changes: one new channel, one new dataclass, new **optional** `Scene` fields, new config fields with defaults. No existing field renamed or removed; every existing subscriber and test is unaffected when the nets are disabled (the default).
