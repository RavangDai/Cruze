# Cruze ↔ Autoware `vision_pilot` — Reference Mapping

Reference analysis mapping [`autowarefoundation/vision_pilot`](https://github.com/autowarefoundation/vision_pilot)
onto Cruze, piece by piece. `vision_pilot` is vendored as a git **submodule** at
`reference/vision_pilot/` — read-only reference, not built into Cruze.

> **No Cruze source changes are implied by this document.** It exists so a future
> session can see exactly which `vision_pilot` idea corresponds to which Cruze
> module, what Cruze already has, what's missing, and what's fundamentally
> different — before deciding whether to borrow anything.

---

## 1. TL;DR — the one difference that explains everything

**`vision_pilot` is a *control* system. Cruze is an *advisory co-pilot*.**

- `vision_pilot` closes the loop: camera → AI → fusion → **planner → `vehicle_interface->write(steering, accel)`** over CAN. It steers and brakes the car (L2 ADAS: ACC, LKAS, AEB, Autopilot).
- Cruze never actuates. It ends at **voice + HUD**: camera → perception → Scene → events → personality → TTS / web dashboard. It *advises* the human driver.

Every other difference below flows from this. `vision_pilot`'s entire `safety_guardian/planning` layer and CAN write-path have **no Cruze equivalent by design** — and Cruze's whole `personality` / `voice` / `maps` / traffic-light / stop-sign stack has no `vision_pilot` equivalent.

A literal "match each and every file" is therefore impossible: different language (C++/ROS 2 vs Python/asyncio), different runtime, different *purpose*. What *is* portable is a set of perception ideas and the three ONNX models — see §7.

---

## 2. Architecture at a glance

### `vision_pilot` (single C++ loop, from `VisionPilot/app/vision_pilot.cpp`)

```
camera_interface (V4L2 / file / ROS2)      vehicle_interface (CAN / file / ROS2)
            │                                          │
            ▼                                          │ ego_v
   ImagePreprocessor.preprocess ──► warped (BEV) + resized
            │
            ▼   InferencePipeline.process(warped, resized)
      ┌─────┴───────────────┬───────────────────┐
      ▼                     ▼                   ▼
  AutoDrive(t-1,t)     AutoSteer            AutoSpeed
  dist,curv,flag       64 waypoints        YOLO boxes (CIPO)
      │                     │                   │
      └──────► LongitudinalFusion (CIPO particle filter) ◄─┘
      └──────► LateralFusion (RANSAC poly + PF + Kalman)
                        │
                        ▼   InferenceFrameResult {cipo, lateral, ...}
                   Planner.compute_plan(cte, epsi, kappa, ego_v, cipo…)
                        │
                        ▼   Plan {acceleration, steering[], warnings[]}
            vehicle_interface->write(steering, accel)   +   Visualization (local / WebRTC)
```

### Cruze (asyncio, event-bus fan-out; from `orchestrator.py` + `core/bus.py`)

```
camera.py ─► PERCEPTION_FRAME
                │
      PerceptionService (perception/pipeline.py)
      detector backend ─► tracker ─► depth ─► lane ─► lights
                │
   ├─► PERCEPTION_DETECTIONS   ├─► PERCEPTION_TRACKS   ├─► PERCEPTION_LANES
                │
telemetry/* ─► TELEMETRY_VEHICLE_STATE
                │
      SceneAssembler (reasoning/scene.py) ─► REASONING_SCENE  (adds IDM required_accel)
                │
      EventEngine (reasoning/events.py, threat.py) ─► REASONING_EVENT
                │
      personality/responses.py ─► VOICE_UTTERANCE ─► voice/tts.py   (speaks)
                │
      hmi/{serialize,webapp}.py ─► web dashboard (WebSocket)         (shows)
```

The key structural contrast: `vision_pilot` is a **synchronous dataflow graph** (each stage calls the next in one `while(true)`); Cruze is a **publish/subscribe bus** where modules never import each other (CLAUDE.md hard rule).

---

## 3. Module-by-module mapping

Status legend: ✅ Cruze has an equivalent · 🟡 partial / different shape · ❌ missing · ➖ N/A by design

| `vision_pilot` module | Purpose | Cruze counterpart | Status | Notes |
|---|---|---|---|---|
| `app/vision_pilot.cpp` | Main loop wiring all stages | `orchestrator.py` + `__main__.py` | 🟡 | Cruze wires services onto a bus instead of calling stages inline. |
| `modules/common/types.hpp` | Shared `Warning`, `Plan` structs | `core/types.py` | 🟡 | Cruze's contract is far richer (Scene, Track, VehicleState, Utterance…). |
| `modules/common/utils` | Misc helpers, matrix load | scattered / `core/` | 🟡 | No single utils module in Cruze. |
| `modules/config` | `.conf` loader | `core/config.py` | ✅ | Cruze uses layered YAML + env vars; `vision_pilot` uses flat `.conf`. |
| `modules/logging` | `VP_INFO/VP_ERROR` logger | `core/logging.py` | ✅ | Direct analogue. |
| `modules/engine` (`onnx_engine`) | ONNX Runtime session wrapper | `perception/detector.py` factory + backends | 🟡 | Same "swappable inference engine" idea; Cruze's is a `Detector` protocol. |
| `modules/models/auto_speed` | YOLO CIPO detector (ONNX) | `perception/backends/yolo.py` | 🟡 | Both YOLO-style; Cruze's is general multi-class, AutoSpeed is CIPO-focused. |
| `modules/models/auto_steer` | Ego-path 64-waypoint net | `perception/lane.py` | 🟡 | Very different method (learned waypoints vs classical lane fit) → both yield lateral offset. |
| `modules/models/auto_drive` | E2E distance + curvature + in-path flag | `perception/depth.py` (distance only) | 🟡 | Cruze has monocular depth; **no curvature / E2E in-path flag**. |
| `modules/models/inference` | Two-frame buffer → parallel ONNX → fusion | `perception/pipeline.py` | 🟡 | Cruze's pipeline is detector→tracker→depth→lane→lights, no Bayesian fusion. |
| `modules/models/trt_engine` | TensorRT execution provider | `perception/backends/tensorrt.py` | ✅ | Same role. |
| `modules/safety_guardian/fusion/longitudinal_fusion` | CIPO particle filter (dist+vel) | `perception/tracker.py` (SORT + scalar Kalman) | 🟡 | Cruze tracks closing speed with a scalar Kalman, not a 500-particle filter. |
| `modules/safety_guardian/fusion/lateral_fusion` | RANSAC poly + PF + Kalman → cte/yaw/curv | `perception/lane.py` (`cte_m`, polys) | 🟡 | Cruze computes `cte_m` from ground-plane geometry; no PF/Kalman smoothing, no yaw/curvature. |
| `modules/safety_guardian/planning/longitudinal_planning` | Accel command (ACC/AEB) | `reasoning/scene.py` IDM `required_accel_mps2` | 🟡 | Cruze *computes* required accel but only to **advise**, never to actuate. |
| `modules/safety_guardian/planning/lateral_planning` | Steering command (LKAS) MPC | — | ➖ | No steering control in Cruze by design. |
| `modules/safety_guardian/planning/planning` | Unified `compute_plan` → `Plan` | `reasoning/events.py` (warnings only) | 🟡 | Cruze emits `DrivingEvent`s, not actuation commands. |
| `modules/sensing/camera_interface` (v4l2/file) | Frame source | `perception/camera.py` | ✅ | Both support live device + video-file replay. |
| `modules/sensing/image_preprocessing` | BEV warp + resize (homography) | inside `depth.py` / `lane.py` (ground-plane) | 🟡 | Cruze has no explicit BEV warp stage; geometry is per-module. |
| `modules/sensing/vehicle_interface` (CAN/file) | Ego speed in, control out | `telemetry/obd.py` (+ `gps`, `imu`, `fusion`) | 🟡 | Cruze reads OBD/GPS/IMU and **fuses** them; never writes control back. |
| `modules/middleware_interfaces/ros2_interface` | ROS 2 camera + vehicle nodes | `core/bus.py` (asyncio) | 🟡 | Both are the "middleware"; ROS 2 topics ↔ Cruze `Channel`s. |
| `modules/visualization` (local + WebRTC) | Live HUD / stream | `hmi/hud.py` + `hmi/webapp.py` + `hmi/web/` | ✅ | Cruze streams a HUD to a browser via WebSocket; `vision_pilot` via WebRTC. |
| `modules/debug` (`debug_draw`) | Annotated telemetry overlay | `hmi/hud.py` (debug draw) | ✅ | Both draw diagnostic overlays. |
| `Calibration/` | Camera intrinsics + homography | `calibration/camera_params.npz` (gitignored) | 🟡 | Cruze stores intrinsics; no scripted homography solver checked in. |
| `Simulation/CARLA`, `Simulation/SODA.Sim` | Sim bridges (ROS 2 / Zenoh) | video replay + `backends/stub.py` | 🟡 | Cruze can replay recorded drives; no live-sim bridge. |
| — | Wake word / STT / LLM dialog / TTS | `voice/{wake,stt,dialog,tts}.py` | ➖ | **Cruze-only.** No voice layer in `vision_pilot`. |
| — | Personality / canned responses | `personality/{persona,responses}.py` | ➖ | **Cruze-only.** |
| — | Speed-limit maps / cache | `maps/{speed_limits,cache}.py` | ➖ | **Cruze-only** ISA source (OSM/cache); `vision_pilot` uses a static config `speed_limit`. |
| — | Traffic-light + stop-sign perception | `perception/lights.py`, `ObjectClass.STOP_SIGN` | ➖ | **Cruze-only.** `vision_pilot` does not classify lights/signs. |

---

## 4. Data-contract mapping (`types.hpp` ↔ `types.py`)

| `vision_pilot` type | Fields | Closest Cruze type | Notes |
|---|---|---|---|
| `Warning` enum | `None, FCW, AEB, LLDW, RLDW` | `DrivingEvent.kind` (str) + `EventLevel` | Cruze kinds: `fcw`, `brake_hard`/`brake_advised` (≈AEB), `lane_departure` (≈LLDW/RLDW), plus `tailgating`, `speeding`, `slow_lead`, `cut_in`, `stop_sign`. |
| `Plan` | `acceleration`, `steering[]`, `warnings[]` | *(none — actuation)* | Nearest partial: `Scene.required_accel_mps2` (the accel number) + `DrivingEvent` (the warnings). No steering equivalent. |
| `models::Detection` | `x1,y1,x2,y2,score,class_id` | `Detection{bbox: BBox, confidence, cls}` | Cruze wraps coords in `BBox` (with `.iou/.cx/.cy`) and uses an `ObjectClass` enum. |
| `AutoSpeedOutput` | `detections[]`, `valid` | `list[Detection]` on `PERCEPTION_DETECTIONS` | Direct analogue. |
| `AutoSteerOutput` | `xp[64]`, `h_vector[64]`, `valid` | `Lanes{left_poly, right_poly, cte_m}` | Different representation of "where the path goes." AutoSteer = ego-path waypoints; Cruze = left/right boundaries + offset. |
| `AutoDriveOutput` | `dist_normalized`, `curvature_raw`, `flag_prob` | `Track.distance_m` (dist only) | Cruze has **no** road-curvature or E2E in-path-object probability. |
| `CIPOFusionEstimate` | `distance_m`, `velocity_ms`, `distance_stddev_m`, `cut_in_detected` | `Track{distance_m, closing_speed_mps}` + `Scene.lead_track` | Cruze's lead is chosen by tracker+scene; `vision_pilot`'s by particle-filter posterior. `cut_in_detected` ↔ Cruze `is_cut_in` / `"cut_in"` event. |
| `LateralFusionEstimate` | `cte_m`, `yaw_rad`, `curvature`, `path_a/b/c`, stddevs | `Lanes.cte_m` (+ `left_poly/right_poly`) | Cruze exposes CTE and polylines; **no** `yaw_rad`, `curvature`, or covariance. |
| `InferenceFrameResult` | bundles all model + fusion outputs | `Scene` | Both are "the per-frame aggregate handed downstream." |
| `LatencyStats` | `pre/ad/as/asp/wall` | `perception.latency_budget_ms` logging | Same spirit (per-stage timing); Cruze logs against a budget. |

**Cruze contract rule (CLAUDE.md):** add optional fields freely, never rename/remove. So *if* you later borrow AutoDrive's curvature or AutoSteer's yaw, the clean move is new **optional** fields on `Lanes` / `Scene` (e.g. `curvature_1pm`, `yaw_rad`), never a rewrite.

---

## 5. ADAS feature parity

| Feature | `vision_pilot` | Cruze | Gap |
|---|---|---|---|
| **ISA** – speed assist | Static `speed_limit` config | ✅ `maps/speed_limits.py` + `is_speeding` → `"speeding"` event | Cruze richer (real limits via maps). |
| **FCW** – forward collision warn | `Warning::FCW` | ✅ `is_forward_collision_warning` → `"fcw"` (CRITICAL) | Parity (advisory). |
| **LDW** – lane departure warn | `Warning::LLDW/RLDW` | ✅ `is_lane_departure` → `"lane_departure"` | Parity (advisory). |
| **AEB** – emergency braking | `Warning::AEB` + accel actuation | 🟡 `"brake_hard"` event (voice only) | Cruze warns, does **not** brake. |
| **ACC** – adaptive cruise | Longitudinal planner actuates throttle | 🟡 IDM `required_accel_mps2` computed, **not** actuated | Advisory only. |
| **LKAS** – lane keep assist | Lateral planner actuates steering | ❌ lane detected, no steering | No control path. |
| **Autopilot** – hands-free | Full E2E closed loop | ❌ | Out of scope: Cruze is a co-pilot, not an autopilot. |
| Traffic-light awareness | — | ✅ `perception/lights.py` (RED/YELLOW/GREEN) | Cruze-only. |
| Stop-sign awareness | — | ✅ `"stop_sign"` event | Cruze-only. |
| Voice / conversational | — | ✅ `voice/` + `personality/` | Cruze-only. |

**Reading:** Cruze already achieves feature-parity on the three *warning* features (FCW, LDW, ISA) and exceeds `vision_pilot` on driver interaction (voice, lights, signs, maps). The gap is entirely the **control** features (ACC/LKAS/AEB/Autopilot), which Cruze deliberately does not do.

---

## 6. Fundamental differences (why this can't be a 1:1 port)

1. **Control vs advice** — `vision_pilot` actuates; Cruze speaks. (§1)
2. **Language / runtime** — C++/ROS 2 synchronous dataflow vs Python/asyncio pub-sub bus. No source drops across.
3. **Perception philosophy** — `vision_pilot` = 3 bespoke E2E ONNX nets + heavy Bayesian fusion (particle filters, RANSAC, dual Kalman smoothers). Cruze = general YOLO detector + SORT tracker + classical depth/lane/HSV. `vision_pilot` is deeper on *estimation*; Cruze is broader on *scene understanding* (classes, lights, signs).
4. **Mapless, single-purpose vs sensor-fusion co-pilot** — `vision_pilot` uses one monocular camera + ego speed. Cruze fuses OBD + GPS + IMU + maps and layers personality/LLM dialog on top.
5. **Config surface** — flat `.conf` vs Cruze's layered YAML + hardware profiles + env overrides.

---

## 7. What Cruze *could* borrow (mapped to Cruze conventions)

Ordered roughly by value-for-effort. Each respects the CLAUDE.md rules (bus-only, hardware = config, additive types).

1. **Wrap the three ONNX models as Cruze backends/services.** The weights are Apache-2.0 and ship in the submodule (`modules/models/weights/*.onnx`), with upstream repos [`auto_speed`](https://github.com/autowarefoundation/auto_speed), [`auto_steer`](https://github.com/autowarefoundation/auto_steer), [`auto_drive`](https://github.com/autowarefoundation/auto_drive).
   - **AutoSpeed** → a new `perception/backends/autospeed.py` implementing the `Detector` protocol (§ "add a new detector backend" in CLAUDE.md) + a hardware YAML.
   - **AutoSteer** → a lane/path service publishing `Lanes` (path waypoints → `left_poly/right_poly` + `cte_m`).
   - **AutoDrive** → a service publishing distance + a *new optional* `Scene.road_curvature_1pm`.
2. **CIPO longitudinal particle filter** → upgrade `perception/tracker.py`'s closing-speed estimate (currently scalar Kalman) with a particle-filter fusion of AutoDrive distance + bbox-projected distance. Improves lead distance/velocity robustness through cut-in/cut-out.
3. **Lateral RANSAC + curvature** → enrich `Lanes` with optional `yaw_rad` and `curvature_1pm` fields; useful for a smarter `is_lane_departure` and future corridor HUD.
4. **BEV homography preprocessing** (`Calibration/calc_front_camera_homography.py`) → a documented ground-plane homography for Cruze's `depth.py`/`lane.py`, improving metric CTE and distance.
5. **CARLA / SODA sim bridge** → a simulated camera+telemetry source for Cruze (it already has video replay + `stub` backend); handy for repeatable testing without hardware.

Everything above is **additive** — new backends/services + new YAML + new optional `types.py` fields. None requires touching existing modules' internals, per the data-contract rule.

---

## 8. Working with the submodule

- **Location:** `reference/vision_pilot/` (registered in `.gitmodules`).
- **Not committed yet** — `git submodule add` staged `.gitmodules` + the pointer; commit is left to you.
- **Fresh clone of Cruze elsewhere:** `git submodule update --init reference/vision_pilot`.
- **Update to upstream latest:** `git -C reference/vision_pilot pull origin main` then commit the moved pointer.
- **Remove:** `git submodule deinit reference/vision_pilot && git rm reference/vision_pilot`.
- The submodule is reference-only; nothing in Cruze imports or builds it.
