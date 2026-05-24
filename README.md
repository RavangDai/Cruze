# Cruze

Vision-first AI co-pilot for the car dashboard. Watches the road, knows vehicle state, warns about hazards, and talks back when spoken to — with personality.

## Quick start (laptop / desktop)

```bash
# Install (ML deps are optional for development)
pip install -e ".[dev]"          # tests only, no camera/ML needed
pip install -e ".[all]"          # everything

# Run with simulated camera (no hardware needed)
python -m cruze --hardware desktop --no-camera

# Run with real webcam
python -m cruze --hardware desktop

# Run tests (no ML deps required)
python -m pytest tests/

# Per-module latency benchmark
python scripts/benchmark.py
```

## Architecture

Async event-driven pub/sub. Every module publishes to and subscribes from a central `EventBus`. **Modules never import each other.**

```
Camera ──► Detector ──► Depth ──► Tracker ──► Scene ──► Rules ──► Personality ──► TTS
                                                   ▲
                                              VehicleState (OBD + GPS + IMU)
```

See `CLAUDE.md` for a complete guide to adding modules, backends, and events.

## Hardware targets

| Profile | Command | Backend |
|---|---|---|
| Laptop / desktop | `--hardware desktop` | YOLOv8n CPU |
| NVIDIA Jetson Orin Nano | `--hardware jetson_orin` | TensorRT |
| Raspberry Pi 5 + Hailo-8 | `--hardware pi5` | TFLite |

Swapping hardware is a one-line config change (or `--hardware` flag). No code changes.

## Warnings Cruze handles

| Warning | Trigger |
|---|---|
| Forward collision (FCW) | TTC to lead vehicle < 3 s |
| Tailgating | Following gap < 2 s headway |
| Speeding | Ego speed > posted limit + 1.4 m/s grace |
| Stop sign | Stop sign detected in tracked objects |
| Slow lead | Lead vehicle at < 60% of posted speed limit |

## Voice

- Wake word: "Cruze" (openWakeWord, stub in Phase 1)
- STT: faster-whisper (offline, CPU-friendly)
- TTS: Piper (offline neural, sentence-boundary streaming)
- Dialog: Anthropic Claude API with scene context; falls back to canned responses offline

## Deploying to production

```bash
# Jetson Orin Nano (run on the device)
sudo bash deploy/jetson/setup.sh

# Docker (Linux with X11)
cd deploy && docker compose up

# systemd (after setup.sh)
systemctl status cruze
journalctl -u cruze -f
```

## Repository layout

```
src/cruze/
  core/         bus, config, logging, types (the contract)
  perception/   camera, detector (YOLOv8/TFLite/TensorRT/stub), tracker, depth, lane
  telemetry/    OBD, GPS, IMU, vehicle_state fuser
  maps/         OSM speed limits + SQLite cache
  reasoning/    scene assembler, threat functions, event rule engine
  voice/        wake word, STT, TTS, Claude dialog
  personality/  persona (decides when/what to say), canned responses
  hmi/          OpenCV HUD overlay
config/
  default.yaml
  hardware/     desktop.yaml  jetson_orin.yaml  pi5.yaml
tests/          test_bus  test_tracker  test_threat  test_events
scripts/        benchmark  calibrate_camera  record_drive
deploy/         Dockerfile  docker-compose  systemd  jetson/setup.sh
models/         README.md (download instructions)
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required for Claude dialog |
| `CRUZE_PERCEPTION_BACKEND` | `yolo` | `stub` / `yolo` / `tflite` / `tensorrt` |
| `CRUZE_VOICE_OFFLINE_MODE` | `false` | Skip API calls, use canned responses |
| `PICOVOICE_ACCESS_KEY` | — | Required for Porcupine wake word |
| `CRUZE_LOGGING_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` |
