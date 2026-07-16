# Cruze — CLAUDE.md

Everything a future Claude Code session needs to add to this codebase without reading every file.

---

## Mission and design principles

Cruze is a voice-activated, vision-first AI co-pilot for car dashboards. It watches the road, knows the vehicle state, and speaks when it matters. It has a personality — warm, situationally witty — but safety always comes first.

**Three non-negotiable design principles:**

1. **Event bus only** — modules never import each other. All communication goes through `core/bus.py`. Violations cause tight coupling that breaks hardware swappability.

2. **Hardware is a config change, not a code change** — detector backend, camera resolution, telemetry source, TTS engine: all driven by YAML in `config/hardware/`. Adding a new deployment target means adding a new YAML file, not touching source.

3. **Graceful degradation always** — if a sensor or API is unavailable, the system logs a warning and continues. No hard crashes on missing hardware.

---

## The data contract: `src/cruze/core/types.py`

**Every shared dataclass lives here. Every module imports from here.**

Key types:
- `BBox` — pixel bounding box with `.iou()`, `.width`, `.height`, `.cx`, `.cy`
- `Detection` — single detector output: bbox + confidence + class + optional distance, mask polygon (`mask_xy`), traffic-light state
- `Track` — tracked object with stable ID, closing speed, distance, mask, debounced light state
- `TrafficLightState` — RED / YELLOW / GREEN / UNKNOWN lamp state
- `LaneLine` / `Lanes` — per-frame lane boundaries (either side may be None), plus curved `left_poly`/`right_poly` polylines and metre-space cross-track error `cte_m`
- `Frame` — raw camera frame with timestamp and focal length
- `VehicleState` — fused OBD + GPS + IMU snapshot
- `Scene` — tracks + vehicle state + identified lead vehicle + fresh lanes + `required_accel_mps2` (IDM urgency scalar; the HUD corridor colour and brake events both derive from it)
- `DrivingEvent` — emitted by rule engine: kind (string), level (EventLevel), context dict
- `Utterance` — what TTS should say: text + priority + interrupt flag

**Backwards-compat rule:** add optional fields freely. Never rename or remove. Existing subscribers break silently.

---

## Channel names: `src/cruze/core/bus.py`

```
Channel.PERCEPTION_FRAME          raw camera frames
Channel.PERCEPTION_DETECTIONS     list[Detection] per frame
Channel.PERCEPTION_TRACKS         list[Track] per frame
Channel.PERCEPTION_LANES          Lanes per frame (left/right LaneLine or None)
Channel.TELEMETRY_VEHICLE_STATE   VehicleState snapshots
Channel.REASONING_SCENE           Scene snapshots
Channel.REASONING_EVENT           DrivingEvent
Channel.VOICE_WAKE                wake word detected (True)
Channel.VOICE_QUERY               transcribed user query (str)
Channel.VOICE_UTTERANCE           Utterance to speak
```

---

## How to add a new module

1. Create `src/cruze/<subsystem>/<module>.py`.
2. Subscribe to whatever channels you need:
   ```python
   queue = bus.subscribe(Channel.REASONING_SCENE, maxsize=4)
   scene = await asyncio.wait_for(queue.get(), timeout=1.0)
   ```
3. Publish results to a channel:
   ```python
   await bus.publish(Channel.VOICE_UTTERANCE, utterance)
   ```
4. Expose an async `run(self)` and `stop(self)` method.
5. Register in `orchestrator.py`:
   - Instantiate the service.
   - Add to `self._services` list.
   - Add `asyncio.create_task(svc.run(), name=...)` to `self._tasks`.

No other files need to change.

---

## How to add a new detector backend

1. Create `src/cruze/perception/backends/<name>.py`.
2. Implement the `Detector` protocol (one method):
   ```python
   def detect(self, frame: Frame) -> list[Detection]: ...
   ```
3. Guard heavy imports inside `__init__` with a clear `ImportError` message:
   ```python
   try:
       import mylib
   except ImportError as exc:
       raise ImportError("mylib required. Install with: pip install mylib") from exc
   ```
4. Register in `src/cruze/perception/detector.py` `load()` factory.
5. Add a hardware profile YAML in `config/hardware/` that sets `perception.backend: <name>`.

---

## How to add a new event/warning

1. **Rule** — add a pure function to `src/cruze/reasoning/threat.py`:
   ```python
   def is_my_warning(scene: Scene, cfg: ReasoningConfig) -> bool: ...
   ```
   Pure functions only: no side effects, no bus access. Unit test it in `tests/test_threat.py`.

2. **Event** — add a rule to `EventEngine._rules()` in `src/cruze/reasoning/events.py`:
   ```python
   if threat.is_my_warning(scene, cfg):
       results.append(("my_warning", EventLevel.WARNING, {"key": value}))
   ```
   The string kind must be unique and stable (it appears in canned responses and logs).

3. **Canned response** — add lines to `src/cruze/personality/responses.py`:
   ```python
   "my_warning": [
       "Something happened up front.",
       "Watch out for the thing.",
   ],
   ```
   Use `{context_key}` placeholders matching the context dict from step 2.

4. **Optionally** — update `voice/dialog.py`'s system prompt if the LLM needs context about the new event type.

---

## Config system: `src/cruze/core/config.py`

Load order (later wins):
1. Hardcoded defaults in the dataclass
2. `config/default.yaml`
3. `config/hardware/<profile>.yaml`
4. User YAML (`--config` flag)
5. Env vars: `CRUZE_<SECTION>_<KEY>=value`  
   e.g. `CRUZE_PERCEPTION_BACKEND=tflite`

All sections are dataclasses (type-safe, IDE-complete). To add a new config key: add the field with a default to the relevant dataclass in `config.py`, then document it in `config/default.yaml`.

---

## Testing conventions

- **No ML deps in tests.** `pytest` must pass with only `pytest`, `pytest-asyncio`, stdlib, and numpy. Heavy deps (ultralytics, faster-whisper, etc.) are behind lazy imports inside factory functions.
- Stub detector (`backend=stub`) and simulated telemetry must always be test-accessible.
- Tracker tests pass explicit `timestamp=` to `update()` for deterministic closing-speed assertions.
- Async tests use `@pytest.mark.asyncio` with `await asyncio.sleep(0)` before publishing to give service tasks time to subscribe.
- `tests/test_threat.py` — pure function tests for all threat predicates.
- `tests/test_tracker.py` — SORT algorithm correctness: ID stability, cross-class non-matching, age-out, closing speed sign.
- `tests/test_events.py` — cooldown logic, event firing, multi-rule interaction.
- `tests/test_bus.py` — drop-oldest, fan-out, cross-channel isolation.

Run: `python -m pytest tests/`

---

## Where to put things

| Thing | Location |
|---|---|
| Model weights | `models/` (gitignored; see `models/README.md`) |
| Camera calibration | `calibration/camera_params.npz` (gitignored) |
| Recorded drives | `drives/<timestamp>/` (gitignored) |
| Speed limit cache | `cache/speed_limits.db` (gitignored) |
| Logs | `logs/cruze.log` (rotated) |

---

## Latency budgets

| Module | Budget | Notes |
|---|---|---|
| Full perception pipeline | 150 ms desktop (seg) / 25 ms Jetson | Configurable: `perception.latency_budget_ms`; advisory — over-budget logs, nothing dropped |
| Detector | ~70% of budget | Biggest cost; -seg weights ≈ 1.5–2× box-only; TensorRT halves it |
| Tracker + depth + lanes + light HSV | ~20% of budget | Near-constant regardless of backend |
| Scene + event engine | < 5 ms | Pure Python, no ML |
| TTS first-sentence | < 800 ms | Sentence-boundary buffering in `tts.py` |

---

## Deploy to Jetson Orin Nano

1. Flash JetPack 6.x.
2. Copy repo to `/opt/cruze`.
3. Run `deploy/jetson/setup.sh` — installs deps, converts YOLO to TensorRT, installs systemd service.
4. Set `ANTHROPIC_API_KEY` in `/opt/cruze/.env`.
5. `systemctl status cruze` to verify.

## Deploy to Raspberry Pi 5

Same steps but use `config/hardware/pi5.yaml` (TFLite backend, tiny Whisper, offline TTS).  
Set `CRUZE_HARDWARE=pi5` in `/opt/cruze/.env`.

---

## Hard rules (never break these)

- **No module-to-module imports outside `core/`.** Modules may only import from `cruze.core.*`. Everything else goes through the bus.
- **No blocking calls in the asyncio loop.** CPU-bound work (detector, STT, synthesis) goes in `run_in_executor(None, ...)`.
- **No demographic-based jokes** in `personality/responses.py` or dialog prompts. Roast situations, never people.
- **No magic numbers without a comment** explaining the physical or empirical basis.
- **Safety events (FCW, stop sign) always bypass the speech cooldown** or use half the normal cooldown — see `EventEngine._evaluate()`.
