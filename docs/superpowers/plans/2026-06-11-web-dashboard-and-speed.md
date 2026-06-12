# Web Dashboard + Real-World Speed Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the OpenCV HUD with an Aviation-HUD web dashboard, make ego speed work from real GPS (serial or phone-over-network), show absolute speed for other vehicles, and upgrade distance/speed accuracy (Kalman tracker + ground-plane depth).

**Architecture:** New `DashboardService` in `hmi/` follows the standard bus-only module pattern (FastAPI + uvicorn in-process, one WebSocket carrying binary JPEG frames + JSON data). GPS gains RMC Doppler speed + UDP/TCP NMEA transports; `VehicleStateService` fuses OBD → GPS-Doppler → GPS-position with staleness checks via a new pure `telemetry/fusion.py`. Tracker EMA is replaced with a per-track 2-state Kalman filter; `depth.py` gains a ground-plane estimator; `reasoning/scene.py` computes absolute speed per track.

**Tech Stack:** Python 3.11 asyncio, FastAPI + uvicorn (optional dep), vanilla JS + Leaflet (CDN, no build step), numpy-free pure-Python Kalman, pytest (suite must pass with stdlib + numpy only).

**Spec:** `docs/superpowers/specs/2026-06-11-web-dashboard-and-speed-design.md`

**Conventions (from CLAUDE.md — read it first):**
- Modules never import each other; only `cruze.core.*` + same-subsystem files.
- No blocking calls in the asyncio loop — CPU/IO-bound work goes in `run_in_executor`.
- Heavy deps (fastapi, uvicorn, cv2) are lazy-imported with graceful degradation.
- Tests must pass with only pytest, pytest-asyncio, stdlib, numpy.
- Run tests with: `python -m pytest tests/ -v`
- No magic numbers without a comment explaining the physical/empirical basis.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/cruze/core/types.py` | Modify | Add `Track.speed_mps`, `VehicleState.gps_speed_mps` (additive only) |
| `src/cruze/telemetry/gps.py` | Modify | RMC speed parsing, UDP/TCP NMEA transports, non-blocking reads |
| `src/cruze/telemetry/fusion.py` | Create | Pure functions: haversine, position speed, priority fusion |
| `src/cruze/telemetry/vehicle_state.py` | Modify | Use fusion; publish `gps_speed_mps` |
| `src/cruze/perception/tracker.py` | Modify | Kalman filter replaces EMA |
| `src/cruze/perception/depth.py` | Modify | Ground-plane distance + fused estimator |
| `src/cruze/perception/pipeline.py` | Modify | Use fused distance estimator |
| `src/cruze/core/config.py` | Modify | `camera_height_m`, `camera_pitch_deg`, HMI web fields |
| `src/cruze/reasoning/scene.py` | Modify | Absolute speed per track |
| `src/cruze/hmi/serialize.py` | Create | Dataclass → JSON-safe dict (no heavy deps) |
| `src/cruze/hmi/webapp.py` | Create | `ClientHub` + `DashboardService` + `build_app` |
| `src/cruze/hmi/web/index.html` | Create | Dashboard page |
| `src/cruze/hmi/web/style.css` | Create | Aviation HUD theme |
| `src/cruze/hmi/web/app.js` | Create | WS client, canvas overlays, Leaflet map |
| `src/cruze/orchestrator.py` | Modify | Select HMI backend |
| `config/default.yaml` | Modify | Document new keys |
| `pyproject.toml` | Modify | `dashboard` optional dependency group |
| Tests | Create/Modify | `test_gps.py`, `test_fusion.py`, `test_tracker.py` (extend), `test_depth.py`, `test_scene.py`, `test_serialize.py`, `test_webapp.py` |

---

### Task 1: Data contract additions

**Files:**
- Modify: `src/cruze/core/types.py`
- Test: `tests/test_types.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_types.py`:

```python
"""Data-contract tests for additive fields (backwards compatibility)."""

import dataclasses

from cruze.core.types import BBox, ObjectClass, Track, VehicleState


def test_track_speed_mps_defaults_none():
    t = Track(track_id=1, bbox=BBox(0, 0, 10, 10), cls=ObjectClass.CAR)
    assert t.speed_mps is None


def test_track_speed_mps_settable_via_replace():
    t = Track(track_id=1, bbox=BBox(0, 0, 10, 10), cls=ObjectClass.CAR)
    t2 = dataclasses.replace(t, speed_mps=25.0)
    assert t2.speed_mps == 25.0
    assert t2.track_id == 1


def test_vehicle_state_gps_speed_defaults_none():
    v = VehicleState()
    assert v.gps_speed_mps is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_types.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'speed_mps'` (or AttributeError).

- [ ] **Step 3: Add the fields (at the END of each dataclass — never reorder)**

In `src/cruze/core/types.py`, `Track` gains one field after `timestamp`:

```python
    # Timestamp of the frame this track was last updated.
    timestamp: float = field(default_factory=time.monotonic)
    # Absolute ground speed estimate (ego speed − closing speed); None when
    # either input is unavailable. Valid for same-direction traffic only.
    speed_mps: float | None = None
```

`VehicleState` gains one field after `posted_speed_limit_mps`:

```python
    posted_speed_limit_mps: float | None = None  # from maps module; None if unknown
    # Raw GPS Doppler speed (RMC speed-over-ground), kept separate from the
    # fused speed_mps for transparency/debugging.
    gps_speed_mps: float | None = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_types.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cruze/core/types.py tests/test_types.py
git commit -m "feat: add Track.speed_mps and VehicleState.gps_speed_mps to data contract"
```

---

### Task 2: GPS Doppler speed from RMC sentences

**Files:**
- Modify: `src/cruze/telemetry/gps.py`
- Modify: `src/cruze/telemetry/vehicle_state.py` (callback signature only)
- Test: `tests/test_gps.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gps.py`:

```python
"""GPS NMEA parsing tests — pure functions, no hardware."""

import pytest

from cruze.telemetry.gps import _dm_to_decimal, _parse_gga_or_rmc

# Standard NMEA reference sentence (Wikipedia/u-blox docs): 22.4 knots SOG.
RMC = "$GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W*6A"
GGA = "$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47"


def test_rmc_speed_over_ground_knots_to_mps():
    fix = _parse_gga_or_rmc(RMC)
    assert fix is not None
    lat, lon, alt_m, heading_deg, speed_mps = fix
    # 22.4 knots × 0.514444 = 11.52 m/s
    assert speed_mps == pytest.approx(11.5235, abs=0.001)
    assert heading_deg == pytest.approx(84.4)
    assert lat == pytest.approx(48.1173, abs=0.0001)


def test_rmc_empty_speed_field_gives_none():
    sentence = "$GPRMC,123519,A,4807.038,N,01131.000,E,,084.4,230394,003.1,W*6A"
    fix = _parse_gga_or_rmc(sentence)
    assert fix is not None
    assert fix[4] is None


def test_gga_has_no_speed():
    fix = _parse_gga_or_rmc(GGA)
    assert fix is not None
    assert fix[4] is None
    assert fix[2] == pytest.approx(545.4)  # altitude


def test_garbage_returns_none():
    assert _parse_gga_or_rmc("not nmea") is None
    assert _parse_gga_or_rmc("$GPXTE,A,A,0.67,L,N*6F") is None


def test_dm_to_decimal_south_west_negative():
    assert _dm_to_decimal("4807.038", "S") == pytest.approx(-48.1173, abs=0.0001)
    assert _dm_to_decimal("01131.000", "W") == pytest.approx(-11.5167, abs=0.0001)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_gps.py -v`
Expected: FAIL — the 5-tuple unpacking fails (parser currently returns 4-tuple).

- [ ] **Step 3: Update the parser, simulator, and callback**

In `src/cruze/telemetry/gps.py`:

Add near the top constants:

```python
# 1 knot = 1852 m / 3600 s.
_KNOTS_TO_MPS = 0.514444

# Tangential speed of the simulated circular route: circumference / period.
# 0.005° ≈ 555 m radius → 2π·555 m / 120 s ≈ 29 m/s (≈ 65 mph).
_SIM_SPEED_MPS = 2 * math.pi * (_SIM_RADIUS_DEG * 111_000.0) / _SIM_PERIOD_S
```

Replace `_parse_gga_or_rmc` entirely:

```python
def _parse_gga_or_rmc(sentence: str) -> tuple[float, float, float, float, float | None] | None:
    """
    Minimal NMEA GGA / RMC parser.
    Returns (lat, lon, alt_m, heading_deg, speed_mps) or None on parse failure.
    speed_mps is Doppler speed-over-ground from RMC; None for GGA (no speed field).
    """
    if not sentence.startswith("$"):
        return None
    parts = sentence.split(",")
    try:
        if sentence.startswith("$GPGGA") or sentence.startswith("$GNGGA"):
            lat = _dm_to_decimal(parts[2], parts[3])
            lon = _dm_to_decimal(parts[4], parts[5])
            alt_m = float(parts[9]) if parts[9] else 0.0
            return lat, lon, alt_m, 0.0, None
        if sentence.startswith("$GPRMC") or sentence.startswith("$GNRMC"):
            lat = _dm_to_decimal(parts[3], parts[4])
            lon = _dm_to_decimal(parts[5], parts[6])
            speed_mps = float(parts[7]) * _KNOTS_TO_MPS if parts[7] else None
            heading_deg = float(parts[8]) if parts[8] else 0.0
            return lat, lon, 0.0, heading_deg, speed_mps
    except (IndexError, ValueError):
        pass
    return None
```

Replace `_simulated_position` (returns 5-tuple now):

```python
    def _simulated_position(self, t: float) -> tuple[float, float, float, float, float]:
        angle = 2 * math.pi * t / _SIM_PERIOD_S
        lat = _SIM_LAT + _SIM_RADIUS_DEG * math.sin(angle)
        lon = _SIM_LON + _SIM_RADIUS_DEG * math.cos(angle)
        alt_m = 15.0
        heading_deg = (math.degrees(angle) + 90) % 360
        return lat, lon, alt_m, heading_deg, _SIM_SPEED_MPS
```

Update the `run()` docstring line to:

```python
        Continuously read GPS and call callback(lat, lon, alt_m, heading_deg, speed_mps).
```

In `src/cruze/telemetry/vehicle_state.py`, update `_on_gps` and add the field in `__init__`:

```python
        self._accel_mps2: float | None = None
        self._gps_speed_mps: float | None = None
```

```python
    def _on_gps(
        self,
        lat: float,
        lon: float,
        alt_m: float,
        heading_deg: float,
        speed_mps: float | None = None,
    ) -> None:
        self._lat = lat
        self._lon = lon
        self._alt_m = alt_m
        self._heading_deg = heading_deg
        self._gps_speed_mps = speed_mps
```

(Fusion comes in Task 4 — for now the GPS speed is just stored.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_gps.py tests/ -v`
Expected: new tests PASS; full suite still green.

- [ ] **Step 5: Commit**

```bash
git add src/cruze/telemetry/gps.py src/cruze/telemetry/vehicle_state.py tests/test_gps.py
git commit -m "feat: parse GPS Doppler speed from RMC sentences"
```

---

### Task 3: Network NMEA transports (UDP/TCP) + non-blocking reads

**Files:**
- Modify: `src/cruze/telemetry/gps.py`
- Test: `tests/test_gps.py` (extend)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gps.py`:

```python
from cruze.telemetry.gps import _parse_endpoint, _parse_nmea_burst


def test_parse_endpoint_sim():
    assert _parse_endpoint("sim") == ("sim", "", 0)


def test_parse_endpoint_serial():
    assert _parse_endpoint("COM5") == ("serial", "COM5", 0)
    assert _parse_endpoint("/dev/ttyUSB0") == ("serial", "/dev/ttyUSB0", 0)


def test_parse_endpoint_udp():
    assert _parse_endpoint("udp://0.0.0.0:10110") == ("udp", "0.0.0.0", 10110)


def test_parse_endpoint_tcp():
    assert _parse_endpoint("tcp://192.168.1.50:2947") == ("tcp", "192.168.1.50", 2947)


def test_parse_nmea_burst_prefers_rmc_speed():
    burst = (GGA + "\r\n" + RMC + "\r\n").encode("ascii")
    fix = _parse_nmea_burst(burst)
    assert fix is not None
    assert fix[4] is not None  # RMC speed wins over GGA's None


def test_parse_nmea_burst_garbage_returns_none():
    assert _parse_nmea_burst(b"\xff\xfe garbage\r\n") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_gps.py -v`
Expected: FAIL — `ImportError: cannot import name '_parse_endpoint'`.

- [ ] **Step 3: Implement transports**

In `src/cruze/telemetry/gps.py`, add module-level functions:

```python
def _parse_endpoint(gps_port: str) -> tuple[str, str, int]:
    """
    Classify a gps_port config value.
    Returns (transport, host_or_path, port):
      "sim"    — simulator
      "udp"    — listen for NMEA datagrams (phone forwarder apps)
      "tcp"    — connect to an NMEA stream (gpsd raw, ESP32 bridges)
      "serial" — local serial device path
    """
    if gps_port == "sim":
        return ("sim", "", 0)
    for scheme in ("udp", "tcp"):
        prefix = scheme + "://"
        if gps_port.startswith(prefix):
            host, _, port = gps_port[len(prefix):].rpartition(":")
            return (scheme, host, int(port))
    return ("serial", gps_port, 0)


def _parse_nmea_burst(data: bytes) -> tuple[float, float, float, float, float | None] | None:
    """
    Parse a chunk possibly containing several NMEA sentences.
    Returns the best fix: an RMC fix (carries speed) wins over GGA;
    otherwise the last parseable fix. None if nothing parsed.
    """
    fix = None
    for line in data.decode("ascii", errors="ignore").splitlines():
        parsed = _parse_gga_or_rmc(line.strip())
        if parsed is not None and (fix is None or parsed[4] is not None):
            fix = parsed
    return fix
```

Update `GPSReader.__init__` (replace the `_simulated` line and add socket state):

```python
    def __init__(self, cfg: "TelemetryConfig", poll_interval_s: float = 1.0) -> None:
        self._cfg = cfg
        self._poll_interval = poll_interval_s
        self._transport, self._host, self._port = _parse_endpoint(cfg.gps_port)
        self._simulated = cfg.simulated or self._transport == "sim"
        self._serial = None
        self._sock = None
        self._tcp_buf = b""
```

Replace `connect()` with a dispatcher plus two helpers:

```python
    def connect(self) -> bool:
        if self._simulated:
            logger.info("GPS: running in simulated mode")
            return True
        if self._transport in ("udp", "tcp"):
            return self._connect_socket()
        return self._connect_serial()

    def _connect_serial(self) -> bool:
        try:
            import serial  # type: ignore
        except ImportError:
            logger.warning("pyserial not installed — GPS will be unavailable")
            self._simulated = True
            return False
        try:
            self._serial = serial.Serial(self._cfg.gps_port, baudrate=9600, timeout=2)
            logger.info("GPS: opened %s", self._cfg.gps_port)
            return True
        except Exception as exc:
            logger.warning("GPS: could not open %s: %s — simulating", self._cfg.gps_port, exc)
            self._simulated = True
            return False

    def _connect_socket(self) -> bool:
        import socket
        try:
            if self._transport == "udp":
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.bind((self._host or "0.0.0.0", self._port))
            else:
                sock = socket.create_connection((self._host, self._port), timeout=5.0)
            # 2 s timeout: NMEA arrives at ≥1 Hz, so 2 s of silence means no fix.
            sock.settimeout(2.0)
            self._sock = sock
            logger.info("GPS: %s endpoint %s:%d ready", self._transport, self._host, self._port)
            return True
        except OSError as exc:
            logger.warning("GPS: could not open %s (%s) — simulating", self._cfg.gps_port, exc)
            self._simulated = True
            return False
```

Replace `run()` and `_read_real()` — all real reads block (socket/serial timeouts), so they
must run in the executor per the no-blocking-calls rule:

```python
    async def run(self, callback) -> None:
        """
        Continuously read GPS and call callback(lat, lon, alt_m, heading_deg, speed_mps).
        """
        t_start = time.monotonic()
        loop = asyncio.get_running_loop()
        while True:
            if self._simulated:
                fix = self._simulated_position(time.monotonic() - t_start)
            else:
                # Reads block up to the 2 s socket/serial timeout — keep them
                # off the event loop.
                fix = await loop.run_in_executor(None, self._read_fix)
            if fix is not None:
                callback(*fix)
            await asyncio.sleep(self._poll_interval)

    def _read_fix(self) -> tuple[float, float, float, float, float | None] | None:
        try:
            if self._transport == "udp":
                return self._read_udp()
            if self._transport == "tcp":
                return self._read_tcp()
            return self._read_serial()
        except Exception as exc:
            logger.debug("GPS read error: %s", exc)
            return None

    def _read_serial(self) -> tuple[float, float, float, float, float | None] | None:
        if self._serial is None:
            return None
        line = self._serial.readline().decode("ascii", errors="ignore").strip()
        return _parse_gga_or_rmc(line)

    def _read_udp(self) -> tuple[float, float, float, float, float | None] | None:
        import socket
        if self._sock is None:
            return None
        try:
            data, _addr = self._sock.recvfrom(4096)
        except (socket.timeout, OSError):
            return None
        return _parse_nmea_burst(data)

    def _read_tcp(self) -> tuple[float, float, float, float, float | None] | None:
        import socket
        if self._sock is None:
            return None
        try:
            chunk = self._sock.recv(4096)
        except (socket.timeout, OSError):
            return None
        if not chunk:
            return None
        self._tcp_buf += chunk
        complete, sep, self._tcp_buf = self._tcp_buf.rpartition(b"\n")
        if not sep:
            return None
        return _parse_nmea_burst(complete)
```

Replace `disconnect()`:

```python
    def disconnect(self) -> None:
        if self._serial is not None:
            self._serial.close()
            self._serial = None
        if self._sock is not None:
            self._sock.close()
            self._sock = None
```

Delete the old `_read_real()` method (replaced by `_read_fix`/`_read_serial`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_gps.py tests/ -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cruze/telemetry/gps.py tests/test_gps.py
git commit -m "feat: UDP/TCP NMEA transports for GPS, non-blocking reads"
```

---

### Task 4: Speed fusion (OBD → GPS-Doppler → GPS-position)

**Files:**
- Create: `src/cruze/telemetry/fusion.py`
- Modify: `src/cruze/telemetry/vehicle_state.py`
- Test: `tests/test_fusion.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_fusion.py`:

```python
"""Speed-fusion pure-function tests."""

import pytest

from cruze.telemetry.fusion import (
    STALENESS_S,
    SpeedSample,
    fuse_speed,
    haversine_m,
    position_speed_mps,
)


def test_haversine_known_distance():
    # 0.001° of latitude ≈ 111.2 m anywhere on Earth.
    d = haversine_m(37.7749, -122.4194, 37.7759, -122.4194)
    assert d == pytest.approx(111.2, rel=0.01)


def test_haversine_zero():
    assert haversine_m(37.0, -122.0, 37.0, -122.0) == 0.0


def test_position_speed_basic():
    # 111.2 m in 10 s ≈ 11.12 m/s.
    v = position_speed_mps(37.7749, -122.4194, 0.0, 37.7759, -122.4194, 10.0)
    assert v == pytest.approx(11.12, rel=0.01)


def test_position_speed_rejects_teleport():
    # 1 full degree (~111 km) in 1 s is a GPS glitch, not motion.
    assert position_speed_mps(37.0, -122.0, 0.0, 38.0, -122.0, 1.0) is None


def test_position_speed_rejects_nonpositive_dt():
    assert position_speed_mps(37.0, -122.0, 5.0, 37.001, -122.0, 5.0) is None


def test_fuse_priority_obd_wins():
    now = 100.0
    samples = [
        (SpeedSample(20.0, 99.5), "obd"),
        (SpeedSample(21.0, 99.9), "gps"),
        (SpeedSample(22.0, 99.9), "gps_pos"),
    ]
    assert fuse_speed(samples, now) == (20.0, "obd")


def test_fuse_falls_through_stale_sources():
    now = 100.0
    samples = [
        (SpeedSample(20.0, now - STALENESS_S - 1.0), "obd"),   # stale
        (SpeedSample(21.0, 99.9), "gps"),
    ]
    assert fuse_speed(samples, now) == (21.0, "gps")


def test_fuse_all_missing_or_stale():
    now = 100.0
    samples = [
        (None, "obd"),
        (SpeedSample(21.0, 0.0), "gps"),
    ]
    assert fuse_speed(samples, now) == (None, "unknown")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_fusion.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cruze.telemetry.fusion'`.

- [ ] **Step 3: Implement `fusion.py`**

Create `src/cruze/telemetry/fusion.py`:

```python
"""
Speed-source fusion — pure functions, no I/O, fully unit-tested.

Priority: OBD wheel speed > GPS Doppler (RMC speed-over-ground) >
GPS position-derived (haversine Δposition/Δtime). A source is used only if
its sample is fresher than STALENESS_S.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

EARTH_RADIUS_M = 6_371_000.0

# A source older than this is considered dead and skipped (GPS modules emit
# at ≥1 Hz, OBD at ~10 Hz — 2 s of silence means the source dropped out).
STALENESS_S = 2.0

# Fixes implying faster than ~200 mph are receiver glitches, not driving.
MAX_PLAUSIBLE_SPEED_MPS = 90.0


@dataclass(frozen=True)
class SpeedSample:
    speed_mps: float
    timestamp: float  # time.monotonic() at receipt


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two WGS-84 points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def position_speed_mps(
    lat1: float, lon1: float, t1: float,
    lat2: float, lon2: float, t2: float,
) -> float | None:
    """
    Speed implied by two consecutive position fixes, or None when the pair is
    unusable (non-positive Δt, or implausibly fast → GPS glitch).
    """
    dt = t2 - t1
    if dt <= 0.0:
        return None
    speed = haversine_m(lat1, lon1, lat2, lon2) / dt
    if speed > MAX_PLAUSIBLE_SPEED_MPS:
        return None
    return speed


def fuse_speed(
    samples: Sequence[tuple[SpeedSample | None, str]],
    now: float,
    staleness_s: float = STALENESS_S,
) -> tuple[float | None, str]:
    """
    Pick the highest-priority fresh sample.

    samples: (sample, source_label) pairs in priority order.
    Returns (speed_mps, source) — (None, "unknown") when nothing is fresh.
    """
    for sample, source in samples:
        if sample is not None and now - sample.timestamp <= staleness_s:
            return sample.speed_mps, source
    return None, "unknown"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_fusion.py -v`
Expected: 8 PASS.

- [ ] **Step 5: Wire fusion into `VehicleStateService`**

In `src/cruze/telemetry/vehicle_state.py`:

Add import:

```python
from cruze.telemetry.fusion import SpeedSample, STALENESS_S, fuse_speed, position_speed_mps
```

Replace the mutable-state block in `__init__` (the lines from `self._speed_mps` through `self._accel_mps2`, including Task 2's `_gps_speed_mps`) with:

```python
        # Latest sample per speed source (fusion picks at publish time).
        self._obd_sample: SpeedSample | None = None
        self._obd_source: str = "obd"            # may be "simulated"
        self._gps_doppler_sample: SpeedSample | None = None
        self._gps_position_sample: SpeedSample | None = None
        self._last_fix: tuple[float, float, float] | None = None  # lat, lon, t

        self._lat: float | None = None
        self._lon: float | None = None
        self._alt_m: float | None = None
        self._heading_deg: float | None = None
        self._accel_mps2: float | None = None
```

Replace `_on_obd` and `_on_gps`:

```python
    def _on_obd(self, speed_mps: float | None, source: str) -> None:
        if speed_mps is not None:
            self._obd_sample = SpeedSample(speed_mps, time.monotonic())
            self._obd_source = source

    def _on_gps(
        self,
        lat: float,
        lon: float,
        alt_m: float,
        heading_deg: float,
        speed_mps: float | None = None,
    ) -> None:
        now = time.monotonic()
        self._lat = lat
        self._lon = lon
        self._alt_m = alt_m
        self._heading_deg = heading_deg

        if speed_mps is not None:
            self._gps_doppler_sample = SpeedSample(speed_mps, now)

        if self._last_fix is not None:
            pos_speed = position_speed_mps(
                self._last_fix[0], self._last_fix[1], self._last_fix[2],
                lat, lon, now,
            )
            if pos_speed is not None:
                self._gps_position_sample = SpeedSample(pos_speed, now)
        self._last_fix = (lat, lon, now)
```

Replace `_publish_loop`:

```python
    async def _publish_loop(self) -> None:
        while self._running:
            now = time.monotonic()
            speed, source = fuse_speed(
                [
                    (self._obd_sample, self._obd_source),
                    (self._gps_doppler_sample, "gps"),
                    (self._gps_position_sample, "gps_pos"),
                ],
                now,
            )
            gps_speed = (
                self._gps_doppler_sample.speed_mps
                if self._gps_doppler_sample is not None
                and now - self._gps_doppler_sample.timestamp <= STALENESS_S
                else None
            )
            state = VehicleState(
                timestamp=now,
                speed_mps=speed,
                speed_mps_source=source,
                heading_deg=self._heading_deg,
                latitude=self._lat,
                longitude=self._lon,
                altitude_m=self._alt_m,
                acceleration_mps2=self._accel_mps2,
                posted_speed_limit_mps=self._posted_speed_limit_mps,
                gps_speed_mps=gps_speed,
            )
            await self._bus.publish(Channel.TELEMETRY_VEHICLE_STATE, state)
            await asyncio.sleep(self._publish_interval)
```

- [ ] **Step 6: Run the full suite**

Run: `python -m pytest tests/ -v`
Expected: all PASS (existing tests don't construct VehicleStateService internals directly).

- [ ] **Step 7: Commit**

```bash
git add src/cruze/telemetry/fusion.py src/cruze/telemetry/vehicle_state.py tests/test_fusion.py
git commit -m "feat: priority speed fusion OBD > GPS Doppler > GPS position"
```

---

### Task 5: Kalman-filtered tracker

**Files:**
- Modify: `src/cruze/perception/tracker.py`
- Test: `tests/test_tracker.py` (extend — existing tests must keep passing)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tracker.py`:

```python
# --- Kalman filtering ---

def test_closing_speed_none_until_second_distance():
    tracker = Tracker(iou_threshold=0.3, max_age=5)
    tracks = tracker.update([_det(10, 10, 50, 50, dist=30.0)], timestamp=0.0)
    assert tracks[0].closing_speed_mps is None
    assert tracks[0].distance_m == pytest.approx(30.0)


def test_kalman_rejects_distance_spike():
    """A single wild distance measurement must not yank the filtered distance."""
    tracker = Tracker(iou_threshold=0.3, max_age=5)
    d = _det(10, 10, 50, 50)
    tracker.update([_det(10, 10, 50, 50, dist=30.0)], timestamp=0.0)
    tracker.update([_det(10, 10, 50, 50, dist=29.5)], timestamp=0.1)
    tracker.update([_det(10, 10, 50, 50, dist=29.0)], timestamp=0.2)
    # Spike: monocular depth glitches to 60 m for one frame.
    tracks = tracker.update([_det(10, 10, 50, 50, dist=60.0)], timestamp=0.3)
    assert tracks[0].distance_m < 40.0, "filter swallowed the 60 m spike"
    # Next normal frame pulls it back.
    tracks = tracker.update([_det(10, 10, 50, 50, dist=28.5)], timestamp=0.4)
    assert tracks[0].distance_m < 33.0


def test_kalman_converges_to_constant_closing_speed():
    """Constant 5 m/s approach at 1 Hz → closing speed near 5 within a few frames."""
    tracker = Tracker(iou_threshold=0.3, max_age=5)
    for i, dist in enumerate([50.0, 45.0, 40.0, 35.0, 30.0]):
        tracks = tracker.update([_det(10, 10, 50, 50, dist=dist)], timestamp=float(i))
    closing = tracks[0].closing_speed_mps
    assert closing is not None
    assert 4.0 < closing < 6.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_tracker.py -v`
Expected: the three new tests FAIL (`closing_speed_none_until_second_distance` may pass by
accident with the EMA — the spike test must fail since raw assignment uses 60.0 directly).

- [ ] **Step 3: Replace the EMA with a Kalman filter**

In `src/cruze/perception/tracker.py`:

Delete the `_EMA_ALPHA` constant. Add the filter class above `_TrackState`:

```python
class _DistanceKalman:
    """
    Per-track 1-D constant-velocity Kalman filter over (distance, d_dist/dt).

    Replaces EMA smoothing: rejects single-frame monocular-depth spikes while
    converging to a constant closing speed within a few observations.
    """

    # Process noise driver: relative acceleration between ego and target can
    # reach ~3 m/s² under hard braking.
    _ACCEL_NOISE = 3.0
    # Measurement σ as a fraction of distance — monocular depth error grows
    # roughly linearly with range (~7 % with ground-plane estimation).
    _MEAS_FRAC = 0.07

    def __init__(self, distance_m: float) -> None:
        self.d = distance_m
        self.v = 0.0  # d(distance)/dt; negative = approaching
        # Initial covariance: distance fairly trusted (σ≈2 m), velocity unknown (σ≈5 m/s).
        self.p11, self.p12, self.p22 = 4.0, 0.0, 25.0
        self.updates = 0

    @property
    def ready(self) -> bool:
        """True once at least one correction happened (≥2 distance measurements)."""
        return self.updates >= 1

    def step(self, measured_m: float, dt: float) -> None:
        # --- Predict (constant velocity model) ---
        d = self.d + self.v * dt
        v = self.v
        q = self._ACCEL_NOISE ** 2
        q11 = q * dt ** 4 / 4.0
        q12 = q * dt ** 3 / 2.0
        q22 = q * dt ** 2
        p11 = self.p11 + 2.0 * dt * self.p12 + dt * dt * self.p22 + q11
        p12 = self.p12 + dt * self.p22 + q12
        p22 = self.p22 + q22
        # --- Update with the measured distance (H = [1, 0]) ---
        r = (self._MEAS_FRAC * measured_m) ** 2
        s = p11 + r
        k1 = p11 / s
        k2 = p12 / s
        innovation = measured_m - d
        self.d = d + k1 * innovation
        self.v = v + k2 * innovation
        self.p11 = (1.0 - k1) * p11
        self.p12 = (1.0 - k1) * p12
        self.p22 = p22 - k2 * p12
        self.updates += 1
```

Replace `_TrackState` (distance/closing fields become the filter):

```python
@dataclass
class _TrackState:
    """Mutable internal state for a single tracked object."""

    track_id: int
    bbox: BBox
    cls: ObjectClass
    kalman: _DistanceKalman | None = None
    age_missed: int = 0
    last_updated: float = field(default_factory=time.monotonic)

    def to_track(self) -> Track:
        return Track(
            track_id=self.track_id,
            bbox=self.bbox,
            cls=self.cls,
            distance_m=self.kalman.d if self.kalman is not None else None,
            closing_speed_mps=(
                -self.kalman.v if self.kalman is not None and self.kalman.ready else None
            ),
            age_missed=self.age_missed,
            timestamp=self.last_updated,
        )
```

Replace `_new_track` and `_update_track`:

```python
    def _new_track(self, det: Detection, now: float) -> _TrackState:
        tid = self._next_id
        self._next_id += 1
        return _TrackState(
            track_id=tid,
            bbox=det.bbox,
            cls=det.cls,
            kalman=_DistanceKalman(det.distance_m) if det.distance_m is not None else None,
            last_updated=now,
        )

    def _update_track(self, trk: _TrackState, det: Detection, now: float) -> None:
        dt = now - trk.last_updated
        trk.bbox = det.bbox
        trk.age_missed = 0
        trk.last_updated = now

        if det.distance_m is None:
            return  # keep last filtered state; no measurement this frame
        if trk.kalman is None:
            trk.kalman = _DistanceKalman(det.distance_m)
        elif dt > 0:
            trk.kalman.step(det.distance_m, dt)
```

Also update the module docstring step 6 line to:

```
  6. Estimate distance + closing speed with a per-track constant-velocity
     Kalman filter — rejects single-frame depth spikes, converges in a few
     frames (replaces the earlier EMA smoothing).
```

- [ ] **Step 4: Run the tracker suite**

Run: `python -m pytest tests/test_tracker.py -v`
Expected: ALL pass — both the pre-existing tests (ID stability, ageing, closing-speed signs)
and the three new Kalman tests. The sign tests still pass because at dt=1 s the gains are
high (first correction yields closing ≈ ±4.3 m/s for a 5 m/s step).
If a numeric bound in the new tests misses by <1 m or <0.5 m/s due to float details, widen
the bound — the behavioral claims are: spike of 60 m filtered to <40 m, and convergence
near 5 m/s.

- [ ] **Step 5: Run the full suite (threat/events depend on closing speed)**

Run: `python -m pytest tests/ -v`
Expected: all PASS — `test_threat.py`/`test_events.py` build `Track` objects directly and
are unaffected by tracker internals.

- [ ] **Step 6: Commit**

```bash
git add src/cruze/perception/tracker.py tests/test_tracker.py
git commit -m "feat: per-track Kalman filter replaces EMA closing-speed smoothing"
```

---

### Task 6: Ground-plane distance estimation

**Files:**
- Modify: `src/cruze/perception/depth.py`
- Modify: `src/cruze/core/config.py` (PerceptionConfig)
- Modify: `src/cruze/perception/pipeline.py`
- Modify: `config/default.yaml`
- Test: `tests/test_depth.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_depth.py`:

```python
"""Monocular depth tests — bbox-height and ground-plane methods."""

import pytest

from cruze.core.types import BBox, ObjectClass
from cruze.perception.depth import (
    estimate_distance,
    estimate_distance_fused,
    estimate_distance_ground_plane,
    focal_length_px,
    horizon_y_px,
)


def test_horizon_at_image_center_with_zero_pitch():
    assert horizon_y_px(720, 1000.0, 0.0) == pytest.approx(360.0)


def test_horizon_rises_with_downward_pitch():
    # Camera pitched 5° down → horizon moves up (smaller y).
    assert horizon_y_px(720, 1000.0, 5.0) < 360.0


def test_ground_plane_known_geometry():
    # f=1000 px, camera 1.5 m above road, horizon at y=360.
    # bbox bottom at y=460 → dy=100 px → Z = 1000·1.5/100 = 15 m.
    bbox = BBox(100, 380, 200, 460)
    z = estimate_distance_ground_plane(bbox, 1000.0, 1.5, 360.0)
    assert z == pytest.approx(15.0)


def test_ground_plane_rejects_bbox_above_horizon():
    bbox = BBox(100, 100, 200, 200)  # bottom edge above horizon
    assert estimate_distance_ground_plane(bbox, 1000.0, 1.5, 360.0) is None


def test_ground_plane_clamps_far_distance():
    # 9 px below horizon → raw 166 m; just inside validity margin, must clamp ≤200.
    bbox = BBox(100, 300, 200, 369.0)
    z = estimate_distance_ground_plane(bbox, 1000.0, 1.5, 360.0)
    assert z is not None and z <= 200.0


def test_fused_uses_ground_plane_for_vehicles():
    bbox = BBox(100, 380, 200, 460)
    z = estimate_distance_fused(bbox, ObjectClass.CAR, 1000.0, 1.5, 360.0)
    assert z == pytest.approx(15.0)


def test_fused_falls_back_to_bbox_height_above_horizon():
    bbox = BBox(100, 100, 200, 200)
    z = estimate_distance_fused(bbox, ObjectClass.CAR, 1000.0, 1.5, 360.0)
    assert z == pytest.approx(estimate_distance(bbox, ObjectClass.CAR, 1000.0))


def test_fused_uses_bbox_height_for_traffic_lights():
    # Traffic lights don't touch the road — ground-plane geometry is invalid.
    bbox = BBox(100, 380, 200, 460)
    z = estimate_distance_fused(bbox, ObjectClass.TRAFFIC_LIGHT, 1000.0, 1.5, 360.0)
    assert z == pytest.approx(estimate_distance(bbox, ObjectClass.TRAFFIC_LIGHT, 1000.0))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_depth.py -v`
Expected: FAIL — `ImportError: cannot import name 'estimate_distance_ground_plane'`.

- [ ] **Step 3: Implement ground-plane estimation**

In `src/cruze/perception/depth.py`, append:

```python
# Classes whose bbox bottom edge touches the road surface — ground-plane
# geometry applies. Signs and lights are elevated; bbox-height only.
_GROUND_CONTACT_CLASSES = {
    ObjectClass.CAR,
    ObjectClass.TRUCK,
    ObjectClass.BUS,
    ObjectClass.MOTORCYCLE,
    ObjectClass.BICYCLE,
    ObjectClass.PERSON,
}

# Bbox bottoms closer to the horizon than this are unusable: at <8 px a
# single-pixel error changes the estimate by >12 %, swamping the geometry.
_MIN_HORIZON_OFFSET_PX = 8.0


def horizon_y_px(image_height: int, focal_length: float, pitch_deg: float) -> float:
    """
    Image row of the horizon. pitch_deg > 0 = camera tilted down, which moves
    the horizon up (smaller y) by f·tan(pitch) from the principal point.
    """
    return image_height / 2.0 - focal_length * math.tan(math.radians(pitch_deg))


def estimate_distance_ground_plane(
    bbox: BBox,
    focal_length: float,
    camera_height_m: float,
    horizon_y: float,
) -> float | None:
    """
    Flat-road distance from the bbox bottom edge:

        Z = f · H_cam / (y_bottom − y_horizon)

    Pure similar-triangles on the ground plane: independent of object size,
    so it avoids the ±20-30 % class-height error of the bbox-height method.
    Returns None when the bbox bottom is at/above the horizon (crest, clipped
    box, elevated object) — caller should fall back to bbox-height.
    """
    dy = bbox.y2 - horizon_y
    if dy < _MIN_HORIZON_OFFSET_PX:
        return None
    distance = focal_length * camera_height_m / dy
    # Same plausibility clamp as estimate_distance().
    return max(0.5, min(distance, 200.0))


def estimate_distance_fused(
    bbox: BBox,
    cls: ObjectClass,
    focal_length: float,
    camera_height_m: float,
    horizon_y: float,
) -> float | None:
    """Ground-plane estimate for road-contact classes; bbox-height otherwise."""
    if cls in _GROUND_CONTACT_CLASSES:
        gp = estimate_distance_ground_plane(bbox, focal_length, camera_height_m, horizon_y)
        if gp is not None:
            return gp
    return estimate_distance(bbox, cls, focal_length)
```

- [ ] **Step 4: Run depth tests**

Run: `python -m pytest tests/test_depth.py -v`
Expected: 8 PASS.

- [ ] **Step 5: Add config fields and wire into the pipeline**

In `src/cruze/core/config.py`, append to `PerceptionConfig`:

```python
    # Camera mounting geometry for ground-plane distance estimation.
    # Height of the lens above the road surface (typical dash mount ≈ 1.2 m).
    camera_height_m: float = 1.2
    # Downward tilt of the camera in degrees (0 = level with the road).
    camera_pitch_deg: float = 0.0
```

In `config/default.yaml`, extend the `perception:` section:

```yaml
perception:
  backend: yolo
  model_path: models/yolov8n.pt
  confidence_threshold: 0.4
  max_track_age: 5
  iou_threshold: 0.3
  latency_budget_ms: 50.0
  camera_height_m: 1.2     # lens height above road surface (metres)
  camera_pitch_deg: 0.0    # positive = tilted down
```

In `src/cruze/perception/pipeline.py`, replace the depth-annotation block inside
`_process_frame` with:

```python
        # Annotate detections with monocular depth if focal length is known.
        if frame.focal_length_px is not None:
            image_height = frame.image.shape[0]
            horizon_y = depth_mod.horizon_y_px(
                image_height,
                frame.focal_length_px,
                self._cfg.perception.camera_pitch_deg,
            )
            detections = [
                Detection(
                    bbox=d.bbox,
                    confidence=d.confidence,
                    cls=d.cls,
                    distance_m=depth_mod.estimate_distance_fused(
                        d.bbox,
                        d.cls,
                        frame.focal_length_px,
                        self._cfg.perception.camera_height_m,
                        horizon_y,
                    ),
                )
                for d in detections
            ]
```

- [ ] **Step 6: Run the full suite**

Run: `python -m pytest tests/ -v`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add src/cruze/perception/depth.py src/cruze/perception/pipeline.py src/cruze/core/config.py config/default.yaml tests/test_depth.py
git commit -m "feat: ground-plane distance estimation with bbox-height fallback"
```

---

### Task 7: Absolute speed for other vehicles

**Files:**
- Modify: `src/cruze/reasoning/scene.py`
- Test: `tests/test_scene.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_scene.py`:

```python
"""SceneAssembler absolute-speed tests — no bus traffic needed."""

import pytest

from cruze.core.bus import EventBus
from cruze.core.config import Config
from cruze.core.types import BBox, ObjectClass, Track, VehicleState
from cruze.reasoning.scene import SceneAssembler


def _assembler(ego_speed=None):
    sa = SceneAssembler(Config(), EventBus(), image_width=1280)
    sa._latest_vehicle_state = VehicleState(speed_mps=ego_speed)
    return sa


def _track(cls=ObjectClass.CAR, closing=None, dist=30.0, tid=1):
    return Track(
        track_id=tid,
        bbox=BBox(600, 300, 700, 400),
        cls=cls,
        distance_m=dist,
        closing_speed_mps=closing,
    )


def test_absolute_speed_ego_minus_closing():
    sa = _assembler(ego_speed=30.0)
    scene = sa._build_scene([_track(closing=5.0)])
    # Lead closing at 5 m/s while ego does 30 → lead absolute 25 m/s.
    assert scene.tracks[0].speed_mps == pytest.approx(25.0)


def test_absolute_speed_clamped_at_zero():
    sa = _assembler(ego_speed=3.0)
    scene = sa._build_scene([_track(closing=10.0)])
    # Math gives −7 (oncoming/stopped edge case) — clamp to 0 for display sanity.
    assert scene.tracks[0].speed_mps == 0.0


def test_no_ego_speed_means_no_absolute_speed():
    sa = _assembler(ego_speed=None)
    scene = sa._build_scene([_track(closing=5.0)])
    assert scene.tracks[0].speed_mps is None


def test_no_closing_speed_means_no_absolute_speed():
    sa = _assembler(ego_speed=30.0)
    scene = sa._build_scene([_track(closing=None)])
    assert scene.tracks[0].speed_mps is None


def test_person_gets_no_absolute_speed():
    sa = _assembler(ego_speed=30.0)
    scene = sa._build_scene([_track(cls=ObjectClass.PERSON, closing=5.0)])
    assert scene.tracks[0].speed_mps is None


def test_lead_track_carries_absolute_speed():
    sa = _assembler(ego_speed=30.0)
    scene = sa._build_scene([_track(closing=5.0)])
    assert scene.lead_track is not None
    assert scene.lead_track.speed_mps == pytest.approx(25.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_scene.py -v`
Expected: FAIL — `speed_mps` is None everywhere (no computation yet).

- [ ] **Step 3: Compute absolute speed in `_build_scene`**

In `src/cruze/reasoning/scene.py`:

Add `import dataclasses` to the imports.

Replace `_build_scene` with:

```python
    def _build_scene(self, tracks: list[Track]) -> Scene:
        tracks = self._with_absolute_speed(tracks)
        lead = self._find_lead(tracks)
        return Scene(
            timestamp=time.monotonic(),
            tracks=tuple(tracks),
            vehicle_state=self._latest_vehicle_state,
            lead_track=lead,
        )

    def _with_absolute_speed(self, tracks: list[Track]) -> list[Track]:
        """
        Attach absolute ground speed: ego speed − closing speed.

        Valid for same-direction traffic (the dominant case for vehicles
        ahead); oncoming vehicles read high. Clamped at 0 — a negative value
        would mean the target is reversing toward us, which at our accuracy
        is indistinguishable from noise.
        """
        ego = self._latest_vehicle_state.speed_mps
        if ego is None:
            return tracks
        out: list[Track] = []
        for t in tracks:
            if t.cls in _VEHICLE_CLASSES and t.closing_speed_mps is not None:
                out.append(
                    dataclasses.replace(t, speed_mps=max(0.0, ego - t.closing_speed_mps))
                )
            else:
                out.append(t)
        return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_scene.py tests/ -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cruze/reasoning/scene.py tests/test_scene.py
git commit -m "feat: absolute speed for tracked vehicles in scene assembly"
```

---

### Task 8: HMI config for the web dashboard

**Files:**
- Modify: `src/cruze/core/config.py` (HMIConfig)
- Modify: `config/default.yaml`
- Test: `tests/test_webapp.py` (create — config part only)

- [ ] **Step 1: Write the failing test**

Create `tests/test_webapp.py`:

```python
"""Dashboard service tests. FastAPI-dependent tests are skipped when the
optional [dashboard] extra is not installed — the core suite must pass with
stdlib + numpy only."""

from cruze.core.config import HMIConfig


def test_hmi_web_defaults():
    cfg = HMIConfig()
    assert cfg.backend == "web"
    assert cfg.web_host == "127.0.0.1"
    assert cfg.web_port == 8484
    assert 1 <= cfg.jpeg_quality <= 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_webapp.py -v`
Expected: FAIL — `AttributeError: 'HMIConfig' object has no attribute 'backend'`.

- [ ] **Step 3: Add the config fields**

In `src/cruze/core/config.py`, replace `HMIConfig` with:

```python
@dataclass
class HMIConfig:
    enabled: bool = True
    # "web" = browser dashboard, "opencv" = legacy cv2 window, "none" = headless.
    backend: str = "web"
    # Bind loopback by default; set 0.0.0.0 to reach the dashboard from a
    # tablet on the car's hotspot (no auth — LAN-trusted only).
    web_host: str = "127.0.0.1"
    web_port: int = 8484
    # JPEG quality for the WebSocket video stream (75 ≈ 60 KB/frame at 720p).
    jpeg_quality: int = 75
    window_title: str = "Cruze HUD"
    overlay_alpha: float = 0.7
    show_track_ids: bool = True
    show_distance: bool = True
```

In `config/default.yaml`, replace the `hmi:` section with:

```yaml
hmi:
  enabled: true
  backend: web            # web | opencv | none
  web_host: 127.0.0.1     # 0.0.0.0 to allow a dash-mounted tablet
  web_port: 8484
  jpeg_quality: 75        # WebSocket JPEG stream quality (1-100)
  window_title: Cruze HUD
  overlay_alpha: 0.7
  show_track_ids: true
  show_distance: true
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_webapp.py tests/ -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cruze/core/config.py config/default.yaml tests/test_webapp.py
git commit -m "feat: HMI config for web dashboard backend"
```

---

### Task 9: JSON serialization module

**Files:**
- Create: `src/cruze/hmi/serialize.py`
- Test: `tests/test_serialize.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_serialize.py`:

```python
"""hmi.serialize tests — everything must be json.dumps-able with no heavy deps."""

import json

from cruze.core.types import (
    BBox,
    DrivingEvent,
    EventLevel,
    ObjectClass,
    Scene,
    Track,
    Utterance,
    VehicleState,
)
from cruze.hmi.serialize import event_to_dict, scene_to_dict, utterance_to_dict


def _track(tid=1, speed=25.0):
    return Track(
        track_id=tid,
        bbox=BBox(10.5, 20.5, 110.5, 120.5),
        cls=ObjectClass.CAR,
        distance_m=34.27,
        closing_speed_mps=4.789,
        speed_mps=speed,
    )


def test_scene_to_dict_roundtrips_json():
    scene = Scene(
        tracks=(_track(),),
        vehicle_state=VehicleState(speed_mps=27.7, speed_mps_source="gps",
                                   latitude=37.7, longitude=-122.4,
                                   posted_speed_limit_mps=29.0),
        lead_track=_track(),
    )
    d = scene_to_dict(scene)
    payload = json.dumps(d)  # must not raise
    back = json.loads(payload)
    assert back["type"] == "scene"
    assert back["lead_id"] == 1
    assert back["tracks"][0]["cls"] == "car"
    assert back["tracks"][0]["speed_mps"] == 25.0
    assert back["state"]["source"] == "gps"
    assert back["state"]["lat"] == 37.7


def test_scene_to_dict_handles_all_none():
    d = scene_to_dict(Scene())
    json.dumps(d)
    assert d["lead_id"] is None
    assert d["tracks"] == []
    assert d["state"]["speed_mps"] is None


def test_event_to_dict():
    ev = DrivingEvent(kind="fcw", level=EventLevel.CRITICAL, context={"ttc_s": 1.2})
    d = event_to_dict(ev)
    json.dumps(d)
    assert d == {"type": "event", "kind": "fcw", "level": "critical",
                 "ts": ev.timestamp, "context": {"ttc_s": 1.2}}


def test_utterance_to_dict():
    u = Utterance(text="Watch the gap.", priority=2)
    d = utterance_to_dict(u)
    json.dumps(d)
    assert d["type"] == "utterance"
    assert d["text"] == "Watch the gap."
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_serialize.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cruze.hmi.serialize'`.

- [ ] **Step 3: Implement `serialize.py`**

Create `src/cruze/hmi/serialize.py`:

```python
"""
Dataclass → JSON-safe dict converters for the dashboard WebSocket protocol.

Pure functions, no heavy imports — unit-tested without fastapi installed.
Every payload carries a "type" discriminator the frontend switches on.
"""

from __future__ import annotations

from typing import Any

from cruze.core.types import DrivingEvent, Scene, Track, Utterance, VehicleState


def _r(value: float | None, ndigits: int = 2) -> float | None:
    """Round for wire compactness; pass None through."""
    return None if value is None else round(value, ndigits)


def track_to_dict(t: Track) -> dict[str, Any]:
    return {
        "id": t.track_id,
        "cls": t.cls.value,
        "bbox": [round(t.bbox.x1, 1), round(t.bbox.y1, 1),
                 round(t.bbox.x2, 1), round(t.bbox.y2, 1)],
        "distance_m": _r(t.distance_m, 1),
        "closing_mps": _r(t.closing_speed_mps),
        "speed_mps": _r(t.speed_mps),
    }


def state_to_dict(v: VehicleState) -> dict[str, Any]:
    return {
        "speed_mps": _r(v.speed_mps),
        "source": v.speed_mps_source,
        "gps_speed_mps": _r(v.gps_speed_mps),
        "heading_deg": _r(v.heading_deg, 1),
        "lat": v.latitude,
        "lon": v.longitude,
        "limit_mps": _r(v.posted_speed_limit_mps),
    }


def scene_to_dict(s: Scene) -> dict[str, Any]:
    return {
        "type": "scene",
        "ts": s.timestamp,
        "tracks": [track_to_dict(t) for t in s.tracks],
        "lead_id": s.lead_track.track_id if s.lead_track is not None else None,
        "state": state_to_dict(s.vehicle_state),
    }


def event_to_dict(e: DrivingEvent) -> dict[str, Any]:
    return {
        "type": "event",
        "kind": e.kind,
        "level": e.level.value,
        "ts": e.timestamp,
        "context": e.context,
    }


def utterance_to_dict(u: Utterance) -> dict[str, Any]:
    return {"type": "utterance", "text": u.text, "priority": u.priority}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_serialize.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cruze/hmi/serialize.py tests/test_serialize.py
git commit -m "feat: JSON serialization for dashboard WebSocket protocol"
```

---

### Task 10: ClientHub + DashboardService

**Files:**
- Create: `src/cruze/hmi/webapp.py`
- Test: `tests/test_webapp.py` (extend)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_webapp.py`:

```python
import asyncio

import pytest

from cruze.hmi.webapp import ClientHub


async def test_hub_broadcast_reaches_all_clients():
    hub = ClientHub(maxsize=4)
    q1, q2 = hub.register(), hub.register()
    hub.broadcast(("txt", "hello"))
    assert q1.get_nowait() == ("txt", "hello")
    assert q2.get_nowait() == ("txt", "hello")


async def test_hub_drop_oldest_when_client_slow():
    hub = ClientHub(maxsize=2)
    q = hub.register()
    for i in range(4):
        hub.broadcast(("bin", i))
    assert q.qsize() == 2
    # Oldest items (0, 1) were dropped.
    assert q.get_nowait() == ("bin", 2)
    assert q.get_nowait() == ("bin", 3)


async def test_hub_unregister_stops_delivery():
    hub = ClientHub()
    q = hub.register()
    hub.unregister(q)
    hub.broadcast(("txt", "x"))
    assert q.qsize() == 0
    assert hub.client_count == 0


def test_index_page_served():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")  # required by starlette's TestClient
    from fastapi.testclient import TestClient

    from cruze.hmi.webapp import _WEB_DIR, ClientHub, build_app

    client = TestClient(build_app(_WEB_DIR, ClientHub()))
    resp = client.get("/")
    assert resp.status_code == 200
    assert "CRUZE" in resp.text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_webapp.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cruze.hmi.webapp'`.

- [ ] **Step 3: Implement `webapp.py`**

Create `src/cruze/hmi/webapp.py`:

```python
"""
Web dashboard service — Aviation HUD in the browser.

Subscribes to:
  Channel.PERCEPTION_FRAME   — frames → JPEG over WebSocket (binary)
  Channel.REASONING_SCENE    — tracks + vehicle state + lead (JSON, ≤10 Hz)
  Channel.REASONING_EVENT    — alerts for the ticker (JSON)
  Channel.VOICE_UTTERANCE    — Cruze's voice line (JSON)

FastAPI + uvicorn run in-process as asyncio tasks. fastapi/uvicorn/cv2 are
optional: missing deps log a warning and the service idles (graceful
degradation, same pattern as the cv2 HUD).
"""

from __future__ import annotations

import asyncio
import json
import logging
import pathlib
import time
from typing import TYPE_CHECKING, Any

from cruze.core.bus import Channel, EventBus
from cruze.core.types import DrivingEvent, Frame, Scene, Utterance
from cruze.hmi import serialize

if TYPE_CHECKING:
    from cruze.core.config import Config

logger = logging.getLogger(__name__)

_WEB_DIR = pathlib.Path(__file__).parent / "web"

# Scene JSON cap — full track lists at camera rate would saturate the socket;
# 10 Hz is smooth for gauges and overlays.
_SCENE_MAX_HZ = 10.0


class ClientHub:
    """
    Outbound fan-out to connected browsers with drop-oldest semantics.

    Each client gets its own queue of ("bin", bytes) / ("txt", str) items;
    a slow client loses old frames instead of stalling the pumps.
    """

    def __init__(self, maxsize: int = 16) -> None:
        self._clients: set[asyncio.Queue] = set()
        self._maxsize = maxsize

    def register(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(self._maxsize)
        self._clients.add(q)
        return q

    def unregister(self, q: asyncio.Queue) -> None:
        self._clients.discard(q)

    def broadcast(self, item: tuple[str, Any]) -> None:
        for q in list(self._clients):
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            q.put_nowait(item)

    @property
    def client_count(self) -> int:
        return len(self._clients)


def build_app(web_dir: pathlib.Path, hub: ClientHub):
    """Build the FastAPI app. Imports fastapi lazily so the module imports clean."""
    from fastapi import FastAPI, WebSocket
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="Cruze Dashboard")

    @app.get("/")
    async def index():
        return FileResponse(web_dir / "index.html")

    app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        q = hub.register()
        try:
            while True:
                kind, payload = await q.get()
                if kind == "bin":
                    await ws.send_bytes(payload)
                else:
                    await ws.send_text(payload)
        except Exception:
            pass  # client disconnected (WebSocketDisconnect or transport error)
        finally:
            hub.unregister(q)

    return app


class DashboardService:
    def __init__(self, cfg: "Config", bus: EventBus) -> None:
        self._cfg = cfg
        self._bus = bus
        self._hub = ClientHub()
        self._server = None
        self._running = False

    async def run(self) -> None:
        self._running = True
        hmi = self._cfg.hmi

        if not hmi.enabled or hmi.backend != "web":
            logger.info("Dashboard: disabled (backend=%s)", hmi.backend)
            while self._running:
                await asyncio.sleep(1.0)
            return

        try:
            import uvicorn
        except ImportError:
            logger.warning(
                "Dashboard: fastapi/uvicorn not installed — disabled. "
                "Install with: pip install cruze[dashboard]"
            )
            while self._running:
                await asyncio.sleep(1.0)
            return

        app = build_app(_WEB_DIR, self._hub)
        config = uvicorn.Config(
            app, host=hmi.web_host, port=hmi.web_port,
            log_level="warning", access_log=False,
        )
        self._server = uvicorn.Server(config)

        pumps = [
            asyncio.create_task(self._pump_frames(), name="dash-frames"),
            asyncio.create_task(self._pump_scenes(), name="dash-scenes"),
            asyncio.create_task(self._pump_events(), name="dash-events"),
            asyncio.create_task(self._pump_utterances(), name="dash-voice"),
        ]
        logger.info("Dashboard: serving on http://%s:%d", hmi.web_host, hmi.web_port)
        try:
            await self._server.serve()
        finally:
            for t in pumps:
                t.cancel()

    async def stop(self) -> None:
        self._running = False
        if self._server is not None:
            self._server.should_exit = True

    async def _pump_frames(self) -> None:
        try:
            import cv2  # type: ignore
        except ImportError:
            logger.warning("Dashboard: opencv not installed — video stream disabled")
            return
        queue = self._bus.subscribe(Channel.PERCEPTION_FRAME, maxsize=2)
        loop = asyncio.get_running_loop()
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self._cfg.hmi.jpeg_quality]
        while self._running:
            try:
                frame: Frame = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            if self._hub.client_count == 0:
                continue  # don't burn CPU encoding for nobody
            ok, buf = await loop.run_in_executor(
                None, cv2.imencode, ".jpg", frame.image, encode_params
            )
            if ok:
                self._hub.broadcast(("bin", buf.tobytes()))

    async def _pump_scenes(self) -> None:
        queue = self._bus.subscribe(Channel.REASONING_SCENE, maxsize=4)
        last_sent = 0.0
        while self._running:
            try:
                scene: Scene = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            now = time.monotonic()
            if now - last_sent < 1.0 / _SCENE_MAX_HZ:
                continue
            last_sent = now
            self._hub.broadcast(("txt", json.dumps(serialize.scene_to_dict(scene))))

    async def _pump_events(self) -> None:
        queue = self._bus.subscribe(Channel.REASONING_EVENT, maxsize=8)
        while self._running:
            try:
                ev: DrivingEvent = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            self._hub.broadcast(("txt", json.dumps(serialize.event_to_dict(ev))))

    async def _pump_utterances(self) -> None:
        queue = self._bus.subscribe(Channel.VOICE_UTTERANCE, maxsize=8)
        while self._running:
            try:
                utt: Utterance = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            self._hub.broadcast(("txt", json.dumps(serialize.utterance_to_dict(utt))))
```

- [ ] **Step 4: Create a placeholder `index.html` so the served-page test can pass**

Create `src/cruze/hmi/web/index.html` containing just (Task 11 replaces this with the real page):

```html
<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>CRUZE</title></head>
<body>CRUZE dashboard placeholder</body></html>
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_webapp.py tests/ -v`
Expected: hub tests PASS; `test_index_page_served` PASSES if fastapi+httpx are installed,
otherwise SKIPS. Full suite green.

- [ ] **Step 6: Commit**

```bash
git add src/cruze/hmi/webapp.py src/cruze/hmi/web/index.html tests/test_webapp.py
git commit -m "feat: DashboardService with WebSocket fan-out hub"
```

---

### Task 11: Aviation HUD frontend

**Files:**
- Modify: `src/cruze/hmi/web/index.html` (replace placeholder)
- Create: `src/cruze/hmi/web/style.css`
- Create: `src/cruze/hmi/web/app.js`
- Test: `tests/test_webapp.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_webapp.py`:

```python
def test_frontend_assets_complete():
    """The static page must contain every element app.js binds to."""
    from cruze.hmi.webapp import _WEB_DIR

    html = (_WEB_DIR / "index.html").read_text(encoding="utf-8")
    for element_id in ("cam", "overlay", "speed", "speed-src", "limit", "hdg",
                       "map", "ticker", "voice-line", "st-fps", "st-gps",
                       "st-clock", "lead-info"):
        assert f'id="{element_id}"' in html, f"missing #{element_id}"
    assert (_WEB_DIR / "app.js").exists()
    assert (_WEB_DIR / "style.css").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_webapp.py::test_frontend_assets_complete -v`
Expected: FAIL — placeholder page has no element IDs.

- [ ] **Step 3: Write the real `index.html`**

Replace `src/cruze/hmi/web/index.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CRUZE</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
      integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin="anonymous">
<link rel="stylesheet" href="/static/style.css">
</head>
<body>
<header id="statusbar">
  <div class="brand">CRUZE <span class="sub">&#9656; CO-PILOT ACTIVE</span></div>
  <div class="status">
    <span>CAM <b id="st-fps">--</b> FPS</span>
    <span>GPS <b id="st-gps">NO FIX</b></span>
    <span id="st-clock">--:--:--</span>
  </div>
</header>
<main>
  <section id="cam-panel">
    <div id="cam-stack">
      <canvas id="cam" width="1280" height="720"></canvas>
      <canvas id="overlay" width="1280" height="720"></canvas>
      <div id="cam-meta">CAM-1 &middot; <span id="cam-res">--</span></div>
    </div>
    <div id="ticker">SYSTEMS NOMINAL</div>
  </section>
  <aside>
    <div class="panel" id="speed-panel">
      <div id="speed">---</div>
      <div class="label">MPH &middot; <span id="speed-src">SRC --</span></div>
      <div class="row">
        <span>LIMIT <b id="limit">--</b></span>
        <span>HDG <b id="hdg">---&deg;</b></span>
      </div>
    </div>
    <div class="panel" id="lead-panel">
      <div class="label">LEAD VEHICLE</div>
      <div id="lead-info">NO TARGET</div>
    </div>
    <div class="panel" id="map-panel">
      <div id="map"></div>
      <div id="nav-fallback">NAV &middot; NO FIX</div>
    </div>
    <div class="panel" id="voice-panel">
      <span class="label">CRUZE &#9656;</span> <span id="voice-line">standing by</span>
    </div>
  </aside>
</main>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
        integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin="anonymous"></script>
<script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 4: Write `style.css`**

Create `src/cruze/hmi/web/style.css`:

```css
/* Aviation HUD theme — phosphor green on black. */
:root {
  --green: #7CFC00;
  --green-dim: #3d6b3d;
  --green-faint: #1f3d1f;
  --green-bg: #030503;
  --amber: #DFFF00;
  --red: #ff4d4d;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  background: #000;
  color: var(--green);
  font-family: Consolas, "Courier New", monospace;
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
#statusbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 16px;
  border-bottom: 1px solid var(--green-faint);
  background: var(--green-bg);
}
.brand { font-weight: bold; letter-spacing: 4px; font-size: 16px; }
.brand .sub { color: var(--green-dim); font-size: 10px; letter-spacing: 1px; }
.status { display: flex; gap: 18px; color: var(--green-dim); font-size: 12px; }
.status b { color: var(--green); }
main { display: flex; gap: 10px; padding: 10px; flex: 1; min-height: 0; }
#cam-panel { flex: 5; display: flex; flex-direction: column; min-width: 0; }
#cam-stack {
  position: relative;
  flex: 1;
  background: #050805;
  border: 1px solid var(--green-faint);
  overflow: hidden;
}
#cam, #overlay {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: contain;
}
#cam-meta {
  position: absolute;
  bottom: 4px;
  left: 8px;
  color: var(--green-dim);
  font-size: 10px;
  z-index: 2;
}
#ticker {
  border: 1px solid var(--green-faint);
  border-top: none;
  background: var(--green-bg);
  padding: 6px 10px;
  font-size: 12px;
  min-height: 30px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
#ticker .critical { color: var(--red); }
#ticker .warning { color: var(--amber); }
#ticker .notice { color: var(--amber); opacity: 0.8; }
#ticker .info { color: var(--green-dim); }
aside { flex: 2; display: flex; flex-direction: column; gap: 10px; min-width: 260px; }
.panel {
  border: 1px solid var(--green-faint);
  background: var(--green-bg);
  padding: 10px;
}
.label { color: var(--green-dim); font-size: 10px; letter-spacing: 2px; }
#speed-panel { text-align: center; }
#speed { font-size: 52px; line-height: 1; }
#speed-panel .row {
  margin-top: 8px;
  display: flex;
  justify-content: center;
  gap: 16px;
  font-size: 12px;
  color: var(--green-dim);
}
#speed-panel .row b { color: var(--green); }
#lead-info { margin-top: 6px; font-size: 13px; }
#map-panel { flex: 1; position: relative; padding: 0; min-height: 160px; }
#map { position: absolute; inset: 0; background: #04140a; }
/* Dark/green map: invert OSM tiles and shove the hue toward phosphor green. */
#map .leaflet-tile {
  filter: grayscale(1) invert(1) sepia(1) hue-rotate(60deg) saturate(3) brightness(0.55);
}
#nav-fallback {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--green-dim);
  font-size: 12px;
  z-index: 1;
}
#map-panel.has-fix #nav-fallback { display: none; }
#voice-panel { font-size: 12px; color: #5a8c5a; }
.ego-marker {
  color: var(--green);
  font-size: 18px;
  text-shadow: 0 0 8px var(--green);
  line-height: 1;
}
@media (max-width: 900px) {
  main { flex-direction: column; }
  aside { flex-direction: row; flex-wrap: wrap; }
  aside .panel { flex: 1 1 40%; }
}
```

- [ ] **Step 5: Write `app.js`**

Create `src/cruze/hmi/web/app.js`:

```javascript
/* Cruze dashboard client: one WebSocket carries binary JPEG frames and JSON
   text messages ({type: "scene" | "event" | "utterance"}). */
"use strict";

const GREEN = "#7CFC00", GREEN_DIM = "#3d6b3d", AMBER = "#DFFF00", RED = "#ff4d4d";
const MPS_TO_MPH = 2.237;

const cam = document.getElementById("cam");
const ovl = document.getElementById("overlay");
const ctx = cam.getContext("2d");
const octx = ovl.getContext("2d");

let tracks = [], leadId = null, state = {};
let frameCount = 0;
let map = null, egoMarker = null, trail = null;

function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onmessage = (e) => {
    if (e.data instanceof ArrayBuffer) drawFrame(e.data);
    else handleJson(JSON.parse(e.data));
  };
  ws.onclose = () => setTimeout(connect, 1000);  // auto-reconnect
}

async function drawFrame(buf) {
  const bmp = await createImageBitmap(new Blob([buf], { type: "image/jpeg" }));
  if (cam.width !== bmp.width || cam.height !== bmp.height) {
    cam.width = ovl.width = bmp.width;
    cam.height = ovl.height = bmp.height;
    document.getElementById("cam-res").textContent = `${bmp.width}x${bmp.height}`;
  }
  ctx.drawImage(bmp, 0, 0);
  bmp.close();
  frameCount++;
  drawOverlay();
}

function handleJson(msg) {
  if (msg.type === "scene") {
    tracks = msg.tracks;
    leadId = msg.lead_id;
    state = msg.state;
    updatePanels();
    drawOverlay();
  } else if (msg.type === "event") {
    addTickerEvent(msg);
  } else if (msg.type === "utterance") {
    document.getElementById("voice-line").textContent = `"${msg.text}"`;
  }
}

/* ---- camera overlay ---- */

function drawOverlay() {
  octx.clearRect(0, 0, ovl.width, ovl.height);
  for (const t of tracks) drawDesignator(t, t.id === leadId);
}

function drawDesignator(t, isLead) {
  const [x1, y1, x2, y2] = t.bbox;
  const w = x2 - x1, h = y2 - y1;
  const arm = Math.min(18, w / 3, h / 3);
  const color = isLead ? GREEN : (t.cls === "person" ? AMBER : GREEN_DIM);
  octx.strokeStyle = color;
  octx.lineWidth = isLead ? 2 : 1;
  octx.setLineDash(t.cls === "person" ? [4, 3] : []);
  // Corner brackets, fighter-HUD style.
  for (const [cx, cy, dx, dy] of [[x1, y1, 1, 1], [x2, y1, -1, 1], [x1, y2, 1, -1], [x2, y2, -1, -1]]) {
    octx.beginPath();
    octx.moveTo(cx + dx * arm, cy);
    octx.lineTo(cx, cy);
    octx.lineTo(cx, cy + dy * arm);
    octx.stroke();
  }
  octx.setLineDash([]);
  // Label: CLASS-ID above, SPEED · DIST below.
  octx.font = "11px Consolas, monospace";
  octx.fillStyle = color;
  const tag = `${t.cls.toUpperCase()}-${String(t.id).padStart(2, "0")}${isLead ? " LEAD" : ""}`;
  octx.fillText(tag, x1, Math.max(10, y1 - 5));
  const parts = [];
  if (t.speed_mps != null) parts.push(`${Math.round(t.speed_mps * MPS_TO_MPH)} MPH`);
  if (t.distance_m != null) parts.push(`${Math.round(t.distance_m)} M`);
  if (t.closing_mps != null && t.closing_mps > 0.5) parts.push("CLOSING");
  if (parts.length) octx.fillText(parts.join(" · "), x1, Math.min(ovl.height - 4, y2 + 13));
}

/* ---- right-hand panels ---- */

function updatePanels() {
  const speedEl = document.getElementById("speed");
  speedEl.textContent = state.speed_mps != null
    ? String(Math.round(state.speed_mps * MPS_TO_MPH)).padStart(3, "0")
    : "---";
  document.getElementById("speed-src").textContent =
    "SRC " + (state.source || "--").toUpperCase().replace("_", "-");
  document.getElementById("limit").textContent =
    state.limit_mps != null ? Math.round(state.limit_mps * MPS_TO_MPH) : "--";
  document.getElementById("hdg").textContent =
    state.heading_deg != null ? `${String(Math.round(state.heading_deg)).padStart(3, "0")}°` : "---°";
  // Speeding tint: red speed readout when >limit.
  speedEl.style.color =
    state.limit_mps != null && state.speed_mps > state.limit_mps ? RED : GREEN;

  const lead = tracks.find((t) => t.id === leadId);
  document.getElementById("lead-info").innerHTML = lead
    ? `${lead.cls.toUpperCase()}-${String(lead.id).padStart(2, "0")} · ` +
      `${lead.speed_mps != null ? Math.round(lead.speed_mps * MPS_TO_MPH) + " MPH" : "-- MPH"} · ` +
      `${lead.distance_m != null ? Math.round(lead.distance_m) + " M" : "-- M"}`
    : "NO TARGET";

  const hasFix = state.lat != null && state.lon != null;
  document.getElementById("st-gps").textContent = hasFix ? "3D FIX" : "NO FIX";
  if (hasFix) updateMap(state.lat, state.lon, state.heading_deg || 0);
}

/* ---- map ---- */

function updateMap(lat, lon, hdg) {
  if (typeof L === "undefined") {  // Leaflet CDN unreachable → numeric fallback
    document.getElementById("nav-fallback").textContent =
      `NAV · ${lat.toFixed(4)} ${lon.toFixed(4)} · HDG ${Math.round(hdg)}°`;
    return;
  }
  document.getElementById("map-panel").classList.add("has-fix");
  if (!map) {
    map = L.map("map", { zoomControl: false, attributionControl: false }).setView([lat, lon], 16);
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 19 }).addTo(map);
    egoMarker = L.marker([lat, lon], {
      icon: L.divIcon({ className: "ego-marker", html: "▲", iconSize: [18, 18] }),
    }).addTo(map);
    trail = L.polyline([], { color: "#7CFC00", weight: 2, opacity: 0.6 }).addTo(map);
  }
  egoMarker.setLatLng([lat, lon]);
  egoMarker.getElement().style.transform += ` rotate(${hdg}deg)`;
  const pts = trail.getLatLngs();
  pts.push([lat, lon]);
  if (pts.length > 300) pts.shift();  // ~5 min of 1 Hz fixes
  trail.setLatLngs(pts);
  map.panTo([lat, lon], { animate: true });
}

/* ---- alert ticker ---- */

const recentEvents = [];

function addTickerEvent(ev) {
  recentEvents.unshift({ ...ev, at: Date.now() });
  if (recentEvents.length > 4) recentEvents.pop();
  renderTicker();
}

function renderTicker() {
  const el = document.getElementById("ticker");
  if (!recentEvents.length) { el.textContent = "SYSTEMS NOMINAL"; return; }
  el.innerHTML = recentEvents
    .map((e) => `<span class="${e.level}">&#9888; ${e.kind.toUpperCase().replace(/_/g, " ")}</span>`)
    .join(" &nbsp;|&nbsp; ");
}

/* ---- status bar ---- */

setInterval(() => {
  document.getElementById("st-fps").textContent = frameCount;
  frameCount = 0;
  document.getElementById("st-clock").textContent = new Date().toLocaleTimeString("en-GB");
  // Age out ticker entries after 10 s.
  const cutoff = Date.now() - 10_000;
  while (recentEvents.length && recentEvents[recentEvents.length - 1].at < cutoff) recentEvents.pop();
  renderTicker();
}, 1000);

connect();
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_webapp.py tests/ -v`
Expected: all PASS (fastapi tests skip when not installed).

- [ ] **Step 7: Commit**

```bash
git add src/cruze/hmi/web/
git commit -m "feat: Aviation HUD dashboard frontend"
```

---

### Task 12: Orchestrator wiring, dependencies, end-to-end verification

**Files:**
- Modify: `src/cruze/orchestrator.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Wire backend selection into the orchestrator**

In `src/cruze/orchestrator.py`, add the import:

```python
from cruze.hmi.webapp import DashboardService
```

Replace the `hud_svc = HUDService(cfg, bus)` line and the `self._services` block with:

```python
        # HMI backend is a config choice: web dashboard, legacy cv2 window, or none.
        hmi_svc: Any | None = None
        if cfg.hmi.backend == "web":
            hmi_svc = DashboardService(cfg, bus)
        elif cfg.hmi.backend == "opencv":
            hmi_svc = HUDService(cfg, bus)

        self._services = [
            camera_svc, perception_svc, telemetry_svc,
            scene_svc, event_svc, persona_svc,
            wake_svc, stt_svc, tts_svc,
        ]
        if hmi_svc is not None:
            self._services.append(hmi_svc)
```

- [ ] **Step 2: Add the optional dependency group**

In `pyproject.toml`, add to `[project.optional-dependencies]`:

```toml
dashboard = [
    "fastapi>=0.110",
    "uvicorn>=0.29",
]
```

and add `"cruze[dashboard]",` to the `all = [...]` list.

- [ ] **Step 3: Run the complete test suite**

Run: `python -m pytest tests/ -v`
Expected: all PASS, zero failures. Note the total count — it should exceed the prior 49 by
roughly 35 new tests.

- [ ] **Step 4: End-to-end smoke test (manual)**

```powershell
pip install -e ".[dashboard]"
$env:CRUZE_PERCEPTION_BACKEND = "stub"
$env:CRUZE_TELEMETRY_SIMULATED = "true"
python -m cruze --no-camera
```

Open http://127.0.0.1:8484 and verify:
- Status bar shows CAM FPS > 0 and GPS `3D FIX`.
- Speed readout animates (simulated OBD sine profile, `SRC SIMULATED`).
- Map shows San Francisco with the ▲ marker circling and a green trail.
- Clock ticks; no console errors in browser dev tools.

If opencv + ultralytics are installed, also verify overlays on a real video:

```powershell
python -m cruze --video tests/solidWhiteRight.mp4
```

Expected: live video in the browser with bracket designators and `MPH · M` chips on
detected vehicles.

- [ ] **Step 5: Commit**

```bash
git add src/cruze/orchestrator.py pyproject.toml
git commit -m "feat: wire web dashboard into orchestrator, add dashboard extra"
```

---

## Self-Review Notes

- **Spec coverage:** dashboard (Tasks 8–12), GPS Doppler + transports (Tasks 2–3),
  fusion (Task 4), Kalman (Task 5), ground-plane (Task 6), absolute speed (Task 7),
  data contract (Task 1), tests throughout, deps (Task 12). Non-goals untouched.
- **Type consistency:** GPS fix is a 5-tuple `(lat, lon, alt_m, heading_deg, speed_mps)`
  everywhere; `SpeedSample(speed_mps, timestamp)`; hub items are `(kind, payload)` tuples;
  WS JSON discriminators `scene|event|utterance` match app.js handlers; serialize keys
  (`distance_m`, `closing_mps`, `speed_mps`, `lead_id`, `limit_mps`) match app.js usage.
- **Known judgment calls:** Kalman test numeric bounds are derived analytically; Step 4 of
  Task 5 explicitly authorizes widening marginal bounds without changing behavioral claims.
```
