# Cruze Web Dashboard + Real-World Speed Tracking — Design

Date: 2026-06-11
Status: Approved by user (visual direction and full design validated via visual companion)

## Goal

Replace the basic OpenCV HUD with a world-class web dashboard (Aviation HUD aesthetic),
make ego speed work in real life via GPS, show absolute speed for other vehicles, and
substantially improve distance/speed accuracy.

## Decisions made during brainstorming

| Decision | Choice |
|---|---|
| GUI platform | Web dashboard: FastAPI + WebSocket + static single-page frontend |
| Visual direction | Aviation HUD — phosphor green on black, monospace, target designators |
| GPS hardware | Both serial NMEA (USB dongle) and network NMEA (phone), config-driven |
| Map tiles | Online OSM via Leaflet, dark/green CSS filter; numeric lat/lon fallback offline |
| Accuracy scope | Full package: Kalman tracker, ground-plane distance, ego-speed fusion |

## 1. Web dashboard — `src/cruze/hmi/webapp.py`

New `DashboardService` following the standard module pattern (async `run()`/`stop()`,
bus-only communication, registered in `orchestrator.py`).

**Subscriptions:**
- `Channel.PERCEPTION_FRAME` — video frames
- `Channel.REASONING_SCENE` — tracks + vehicle state + lead vehicle in one snapshot
- `Channel.REASONING_EVENT` — alerts for the ticker
- `Channel.VOICE_UTTERANCE` — show what Cruze says in the voice line

**Transport:**
- FastAPI + uvicorn run in-process (uvicorn `Server` with `Config`, served as an asyncio task).
- One WebSocket endpoint (`/ws`):
  - Binary messages: JPEG-encoded frames (cv2.imencode in `run_in_executor` — no blocking
    in the loop). Drop-oldest when the client is slow (mirror bus philosophy).
  - Text messages: JSON scene/state/event/utterance payloads at ~10 Hz.
- Static frontend served from `src/cruze/hmi/web/` (`index.html`, `app.js`, `style.css`).
  No build step — vanilla JS + Leaflet from CDN with graceful degradation if CDN
  unreachable (map panel falls back to numeric NAV readout).

**Frontend (Aviation HUD):**
- Top status bar: CRUZE wordmark, CAM FPS, GPS fix status, OBD status, clock.
- Camera panel: `<video>`-less approach — draw JPEG frames to `<canvas>`, overlay layer
  on a second canvas: corner-bracket target designators per track, label
  `CLASS-ID · SPEED MPH · DIST M`, lead vehicle emphasized (bright green), pedestrians
  dashed amber. Latency readout.
- Alert ticker under camera: active events with level colors (green/amber/red), recent
  cleared events dimmed.
- Right column: large ego speed readout with source tag (`SRC OBD` / `SRC GPS-DOPPLER` /
  `SRC GPS-POS`), posted limit, heading; lead-vehicle card; Leaflet map with glowing
  heading arrow, breadcrumb trail (last ~5 min), dark/green filter; Cruze voice line.
- Responsive: usable on a dash-mounted tablet.

**Config (`HMIConfig`):**
- `backend: str = "web"` — `web` | `opencv` | `none` (old cv2 HUD stays available)
- `web_host: str = "127.0.0.1"`, `web_port: int = 8484`
- `jpeg_quality: int = 75` (bandwidth vs quality)
- Existing fields unchanged.

**Graceful degradation:** if fastapi/uvicorn not installed, log a warning and idle
(same pattern as current HUD with cv2).

## 2. GPS speed — `telemetry/gps.py`, `telemetry/vehicle_state.py`

**Parsing:** extract speed-over-ground from RMC field 7 (knots → m/s, ×0.514444) —
Doppler-derived, ~0.3 m/s accuracy. Callback signature gains
`speed_mps: float | None` (backwards-compatible keyword).

**Transports, selected by `gps_port` scheme:**
- `COM5` / `/dev/ttyUSB0` — serial NMEA (existing path)
- `udp://0.0.0.0:10110` or `tcp://host:port` — network NMEA (phone apps like ShareGPS)
- `sim` — existing simulator (also emits simulated speed now)

**Fusion in `VehicleStateService` (priority with 2 s staleness window):**
1. OBD wheel speed (`source="obd"`)
2. GPS Doppler speed (`source="gps"`)
3. GPS position-derived: haversine Δposition/Δt with sanity clamps — reject fixes
   implying > 90 m/s or teleports (`source="gps_pos"`)

`VehicleState.speed_mps_source` documents which source is live; the dashboard shows it.

## 3. Other-vehicle absolute speed + accuracy

**Kalman tracker (`perception/tracker.py`):** replace the EMA closing-speed smoother
with a per-track 1-D Kalman filter, state `[distance, closing_speed]`,
constant-velocity model, pure numpy (no scipy hard dependency — assignment fallback
stays). Tuned so closing speed converges within ~0.5 s and rejects single-frame
distance spikes. Tracker `update(detections, timestamp=)` API unchanged.

**Ground-plane distance (`perception/depth.py`):** for road-contact classes
(car/truck/bus/motorcycle/bicycle/person), estimate distance from the bbox bottom edge:

```
Z = f · H_cam / (y_bottom − y_horizon)
```

where `H_cam` is camera height above road and `y_horizon` derives from camera pitch.
New `PerceptionConfig` fields: `camera_height_m: float = 1.2`,
`camera_pitch_deg: float = 0.0` (documented physical basis). Falls back to the
existing bbox-height method when the bbox bottom is above the horizon (cresting hill,
clipped box) or config absent. Expected accuracy: ±5–10 % under 50 m vs ±20–30 % today.

**Absolute speed (`reasoning/scene.py`):** when building the Scene, compute
`other_speed = ego_speed − closing_speed` for each track with both inputs available;
attach via new optional field `Track.speed_mps: float | None = None`
(`dataclasses.replace`, backwards-compatible). Known limitation: correct for
same-direction traffic; oncoming vehicles read high (lane-gating is future work).

## 4. Data contract changes (`core/types.py`)

Additive only:
- `Track.speed_mps: float | None = None` — absolute ground speed estimate
- `VehicleState.gps_speed_mps: float | None = None` — raw GPS Doppler speed (debugging/fusion transparency)

## 5. Testing (no ML deps, per CLAUDE.md)

- `tests/test_gps.py` — RMC speed parsing, knots conversion, dm→decimal existing cases,
  haversine speed, staleness/priority fusion logic (pure functions extracted).
- `tests/test_tracker.py` — extend: Kalman convergence with explicit timestamps,
  spike rejection, closing-speed sign conventions preserved.
- `tests/test_depth.py` — ground-plane geometry cases (known camera height/pitch →
  known distances), horizon-edge fallback.
- `tests/test_scene.py` — absolute speed math incl. missing-input cases.
- `tests/test_webapp.py` — JSON payload serialization (pure), endpoint tests behind
  `pytest.importorskip("fastapi")`.

## 6. Dependencies

New optional: `fastapi`, `uvicorn` (desktop/jetson/pi profiles). Core test suite still
passes with stdlib + numpy + pytest only.

## 7. Non-goals

- Offline map tiles (revisit if dead zones become a problem)
- Lane-gating oncoming traffic for speed display
- Stereo/LiDAR depth
- Authentication on the dashboard (LAN-local, bind 127.0.0.1 by default)
