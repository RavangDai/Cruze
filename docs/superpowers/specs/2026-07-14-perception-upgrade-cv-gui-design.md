# Perception Upgrade + CV GUI — Design

**Date:** 2026-07-14
**Status:** Approved (user selected all four design options; plan approved via plan mode)
**Branch:** feature/web-dashboard

## Goal

Cruze should detect lanes, traffic lights (with red/yellow/green state), pedestrians,
cars/trucks/buses/motorcycles/bicycles, and stop signs — and render all of it
beautifully in **both** GUIs:

1. **Web dashboard (primary)** — the aviation-HUD frontend from the approved
   2026-06-11 web-dashboard spec, extended with instance-mask overlays, lane
   polylines, and traffic-light state designators.
2. **cv2 HUD (secondary)** — backport masks, lanes, and light state to
   `hmi/hud.py`.

## User decisions

| Decision | Choice |
|---|---|
| GUI platform | Both (web primary, cv2 backport) |
| "Environment" | Instance segmentation masks (yolov8n-seg), translucent per-class colors |
| Traffic-light state | Yes — classical HSV crop analysis, no new model |
| Detector model | yolov8n-seg (backward compatible with box-only weights) |

## Architecture

All changes follow the three non-negotiables: bus-only communication, hardware
as config, graceful degradation. Data-contract changes are strictly additive.

### Data contract (`core/types.py`)

- `TrafficLightState` enum: RED / YELLOW / GREEN / UNKNOWN.
- `LaneLine` frozen dataclass (x1, y1, x2, y2 pixel segment) — moved to core so
  reasoning and HMI can consume lanes without importing `perception.lane`.
- `Lanes` frozen dataclass: `left`/`right: LaneLine | None`, timestamp, frame_id.
- `Detection.mask_xy: tuple[tuple[float, float], ...] | None` — instance-mask
  outline polygon in image pixels, downsampled to ≤32 points at the detector.
- `Detection.light_state: TrafficLightState | None` — set only for TRAFFIC_LIGHT.
- `Track.mask_xy` / `Track.light_state` — same, carried by the tracker.
- `Scene.lanes: Lanes | None`.

### New bus channel

`Channel.PERCEPTION_LANES` — `Lanes` per frame, published by PerceptionService.

### Pipeline changes (`perception/`)

- `backends/yolo.py`: extract `results.masks.xy` polygons (index-aligned with
  boxes), downsample via `_downsample_polygon` (≤32 points); `None` when running
  box-only weights. Default weights: `models/yolov8n-seg.pt`.
- **New `perception/lights.py`**: pure-numpy HSV classifier. Lit-pixel mask
  (V > 0.45, S > 0.35) → hue histogram (red/yellow/green bands) → winner needs
  ≥2% of crop lit and ≥1.5× runner-up, else UNKNOWN. Orientation-agnostic
  (works on loose/clipped/horizontal boxes, unlike thirds-position methods).
- `pipeline.py`: one combined executor call runs detector + light annotation +
  `lane.detect_lanes` (gated by `perception.lane_detection_enabled`); the depth
  annotation rebuild switches to `dataclasses.replace` so new optional fields
  survive. Publishes detections → lanes → tracks.
- `tracker.py`: `_TrackState` carries mask (fresh every match) and light state
  with debounce — UNKNOWN holds the last confident state for up to 15 frames
  (~0.5 s at 30 fps) before reverting to UNKNOWN.
- `lane.py`: reuses core `LaneLine`; unchanged algorithm (Canny + Hough).

### Scene fold-in (`reasoning/scene.py`)

SceneAssembler drains `PERCEPTION_LANES` alongside vehicle state and attaches
lanes to the Scene only when fresh (< 0.5 s old ≈ 15 frames — staleness means
lane detection stopped or is disabled).

### Serialization (`hmi/serialize.py`)

- `track_to_dict` gains `"light"` (state string or null) and `"mask"` (flat
  `[x1, y1, x2, y2, ...]` list — half the JSON overhead of nested pairs).
- New `lanes_to_dict` → `{"left": [x1,y1,x2,y2] | null, "right": ...} | null`.
- `scene_to_dict` gains `"lanes"`.
- Wire budget: 32-pt mask ≈ 450 B/track; 10 tracks at 10 Hz ≈ 45 KB/s —
  negligible next to the ~600 KB/s JPEG stream.

### HMI

- `hmi.backend` gains `"both"`; each service gates itself:
  DashboardService runs for `web`/`both`, HUDService for `opencv`/`both`
  (HUDService previously ran unconditionally — fixed).
- `hud.py`: translucent `fillPoly` masks (existing `addWeighted` blend),
  double-stroked glowing lane lines from `PERCEPTION_LANES` (fresh < 1 s),
  light-state colors + label suffix for traffic lights. New config toggles
  `hmi.show_masks` / `hmi.show_lanes`.
- **New frontend** `hmi/web/` (index.html, style.css, app.js, render.js):
  vanilla JS, no build step. Two stacked canvases sized to the JPEG's natural
  resolution (overlay coords = frame pixels, CSS scales both). rAF render loop
  draws latest scene: mask Path2D fills, corner-bracket designators with
  `CLASS-ID · MPH · M` labels, lead = bright green + LEAD tag, pedestrians =
  dashed amber, traffic lights get state-colored brackets + mini 3-lamp stack,
  lanes = glow-stroked segments + 6%-alpha corridor fill. Status bar (FPS,
  link state, clock), right column (ego speed + source, limit, heading, lead
  card, Leaflet map with numeric NAV fallback), alert ticker, voice line.
  Phosphor-green aviation aesthetic per the 2026-06-11 spec.

### Orchestration / packaging

- `orchestrator.py` registers `DashboardService`.
- `pyproject.toml`: new `dashboard` extra (fastapi, uvicorn), added to `all`;
  `httpx` in dev (TestClient); `cruze.hmi` package-data ships `web/*`.
- Config: `model_path: models/yolov8n-seg.pt`, `lane_detection_enabled: true`,
  desktop `latency_budget_ms: 150` (seg ≈ 1.5–2× box-only on CPU; budget is
  advisory — over-budget logs at DEBUG, nothing dropped).
- Docs: models/README.md (seg weights), CLAUDE.md (new channel + types).

## Error handling / degradation

- Box-only weights → masks simply absent, GUI draws brackets only.
- No cv2 → lanes return (None, None); no fastapi → dashboard idles with a
  warning; no clients connected → no JPEG encoding.
- Unlit/ambiguous/occluded lights → UNKNOWN (grey), never a guess.

## Testing

Repo rule: suite passes with pytest + pytest-asyncio + stdlib + numpy only.

- `tests/test_lights.py` (new) — synthetic BGR crops: red/green/amber patches,
  dark, tiny, unsaturated-bright, mixed-color; annotate_lights pass-through
  and bbox clamping.
- `tests/test_masks.py` (new) — `_downsample_polygon`: cap at 32, pass-through,
  degenerate < 3 points → None, rounding.
- `tests/test_tracker.py` — mask/light propagation, fresh-mask replacement,
  debounce hold and expiry (explicit timestamps).
- `tests/test_scene.py` — fresh lanes attached, stale lanes dropped, none → None.
- `tests/test_serialize.py` — mask/light/lanes wire format incl. all-None case.
- `tests/test_webapp.py` — new HMIConfig fields; index page still serves.

## Verification

1. `pip install -e .[vision,dashboard,dev]`, `python -m pytest tests/`.
2. `python -m cruze --video tests/solidWhiteRight.mp4 --loop` →
   http://127.0.0.1:8484 shows lanes + masks + brackets.
3. `CRUZE_HMI_BACKEND=opencv` → cv2 window shows same; `both` → both.
4. Traffic-light state needs a dashcam clip with lights (no bundled clip has
   them); verify no false designators on the lane videos.

## Non-goals

- Learned lane models (UFLD), road-surface segmentation (COCO has no road
  class), traffic-sign content recognition, dashboard auth, offline map tiles.
