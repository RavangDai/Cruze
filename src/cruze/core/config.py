"""
Layered configuration loader.

Load order (later wins):
  1. Built-in defaults (hardcoded below)
  2. config/default.yaml
  3. config/hardware/<profile>.yaml  (selected by --hardware flag)
  4. User YAML passed via --config
  5. Environment variables:  CRUZE_<SECTION>_<KEY>=value
     e.g. CRUZE_PERCEPTION_BACKEND=tflite
          CRUZE_CAMERA_WIDTH=1280

All values exposed as a single Config dataclass so callers get IDE
completion and an AttributeError (not KeyError) on typos.
"""

from __future__ import annotations

import os
import pathlib
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class CameraConfig:
    device_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    # Horizontal field of view in degrees; used to derive focal_length_px.
    hfov_deg: float = 70.0
    simulated: bool = False          # True → StubCamera; no real device opened
    # Path to a video file to replay instead of a live device. Empty → use
    # device_index. When set, frames are paced at the file's native FPS.
    source: str = ""
    loop: bool = False               # When replaying a file, restart at EOF


@dataclass
class PerceptionConfig:
    backend: str = "yolo"            # yolo | tflite | tensorrt | stub
    model_path: str = "models/yolov8n.pt"
    confidence_threshold: float = 0.4
    # Max missed frames before a track is dropped.
    max_track_age: int = 5
    # Min IoU for a detection to be associated with an existing track.
    iou_threshold: float = 0.3
    # Latency budget for the full perception pipeline in milliseconds.
    latency_budget_ms: float = 50.0


@dataclass
class TelemetryConfig:
    obd_port: str = "auto"           # "auto" → python-OBD auto-detect; "sim" → simulator
    gps_port: str = "sim"            # serial port path or "sim"
    imu_port: str = "sim"
    simulated: bool = False          # master switch: all sources simulated


@dataclass
class MapsConfig:
    # Overpass API endpoint.
    overpass_url: str = "https://overpass-api.de/api/interpreter"
    cache_db_path: str = "cache/speed_limits.db"
    # Grid cell size in degrees (~111 km per degree → 0.001° ≈ 111 m).
    cache_grid_deg: float = 0.001
    offline_only: bool = False       # skip Overpass queries entirely


@dataclass
class ReasoningConfig:
    # TTC threshold below which FCW is raised (seconds).
    fcw_ttc_threshold_s: float = 3.0
    # Following gap threshold below which tailgating is raised (seconds).
    tailgating_gap_threshold_s: float = 2.0
    # Speed grace above posted limit before speeding event (m/s ≈ 1.4 mph).
    speeding_grace_mps: float = 1.4
    # Per-event minimum interval between repeat warnings (seconds).
    event_cooldown_s: float = 8.0


@dataclass
class VoiceConfig:
    # Wake word backend: "openwakeword" | "porcupine" | "stub"
    wake_backend: str = "stub"
    wake_word: str = "cruze"
    # STT backend: "faster_whisper" | "stub"
    stt_backend: str = "stub"
    whisper_model: str = "base.en"
    # TTS backend: "piper" | "stub"
    tts_backend: str = "stub"
    piper_model_path: str = "models/en_US-lessac-medium.onnx"
    # Anthropic model for dialog.
    claude_model: str = "claude-sonnet-4-6"
    # Fall back to canned responses when True (no network needed).
    offline_mode: bool = False


@dataclass
class HMIConfig:
    enabled: bool = True
    window_title: str = "Cruze HUD"
    overlay_alpha: float = 0.7
    show_track_ids: bool = True
    show_distance: bool = True


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_dir: str = "logs"
    structured: bool = True          # JSON lines for production; pretty for dev
    max_bytes: int = 10_000_000
    backup_count: int = 5


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    hardware_profile: str = "desktop"
    camera: CameraConfig = field(default_factory=CameraConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    maps: MapsConfig = field(default_factory=MapsConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    hmi: HMIConfig = field(default_factory=HMIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _try_import_yaml() -> Any:
    try:
        import yaml  # type: ignore
        return yaml
    except ImportError:
        return None


def _load_yaml(path: pathlib.Path) -> dict[str, Any]:
    yaml = _try_import_yaml()
    if yaml is None:
        logger.warning("pyyaml not installed — skipping %s", path)
        return {}
    if not path.exists():
        return {}
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    logger.debug("Loaded config from %s", path)
    return data


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge *override* into *base* recursively; *override* wins on conflicts."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Apply CRUZE_<SECTION>_<KEY>=value env vars."""
    prefix = "CRUZE_"
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix):].lower().split("_", 1)
        if len(parts) != 2:
            continue
        section, field_name = parts
        if section in data and isinstance(data[section], dict):
            # Attempt type coercion based on existing value type.
            existing = data[section].get(field_name)
            if isinstance(existing, bool):
                data[section][field_name] = value.lower() in ("1", "true", "yes")
            elif isinstance(existing, int):
                try:
                    data[section][field_name] = int(value)
                except ValueError:
                    pass
            elif isinstance(existing, float):
                try:
                    data[section][field_name] = float(value)
                except ValueError:
                    pass
            else:
                data[section][field_name] = value
    return data


def _dict_to_config(data: dict[str, Any]) -> Config:
    """Populate a Config from a raw dict; unknown keys are silently ignored."""
    def _apply(dc_instance: Any, src: dict) -> None:
        for fname, fval in src.items():
            if not hasattr(dc_instance, fname):
                continue
            existing = getattr(dc_instance, fname)
            if isinstance(existing, (CameraConfig, PerceptionConfig, TelemetryConfig,
                                     MapsConfig, ReasoningConfig, VoiceConfig,
                                     HMIConfig, LoggingConfig)):
                if isinstance(fval, dict):
                    _apply(existing, fval)
            else:
                setattr(dc_instance, fname, fval)

    cfg = Config()
    _apply(cfg, data)
    return cfg


def load_config(
    hardware_profile: str = "desktop",
    extra_config_path: pathlib.Path | None = None,
    repo_root: pathlib.Path | None = None,
) -> Config:
    """
    Build and return the layered Config.

    Parameters
    ----------
    hardware_profile:
        Name of the hardware profile YAML under config/hardware/.
    extra_config_path:
        Optional user-supplied YAML that overlays everything else.
    repo_root:
        Root of the Cruze repo. Defaults to three parents up from this file
        (src/cruze/core/config.py → repo root).
    """
    if repo_root is None:
        repo_root = pathlib.Path(__file__).parent.parent.parent.parent

    config_dir = repo_root / "config"

    raw: dict[str, Any] = {}
    raw = _deep_merge(raw, _load_yaml(config_dir / "default.yaml"))
    raw = _deep_merge(raw, _load_yaml(config_dir / "hardware" / f"{hardware_profile}.yaml"))
    if extra_config_path:
        raw = _deep_merge(raw, _load_yaml(extra_config_path))
    raw = _apply_env_overrides(raw)

    cfg = _dict_to_config(raw)
    cfg.hardware_profile = hardware_profile
    return cfg
