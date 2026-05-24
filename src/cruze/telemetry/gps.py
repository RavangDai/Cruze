"""
GPS reader — parses NMEA sentences from a serial port, or simulates.

Simulated mode: drives a fixed route (looping around a reference lat/lon)
so the speed-limit lookup has a location to work with.

Published values: latitude, longitude, altitude_m, heading_deg.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cruze.core.config import TelemetryConfig

logger = logging.getLogger(__name__)

# Simulated route: small circle around a reference point.
_SIM_LAT = 37.7749    # San Francisco
_SIM_LON = -122.4194
_SIM_RADIUS_DEG = 0.005   # ~550 m radius
_SIM_PERIOD_S = 120.0     # one lap in 2 minutes


class GPSReader:
    """
    Reads GPS position from a serial NMEA source, or simulates it.

    Parameters
    ----------
    cfg:
        Telemetry config. Simulated when cfg.simulated or cfg.gps_port == "sim".
    poll_interval_s:
        Sampling interval. Real GPS modules typically output at 1 Hz.
    """

    def __init__(self, cfg: "TelemetryConfig", poll_interval_s: float = 1.0) -> None:
        self._cfg = cfg
        self._poll_interval = poll_interval_s
        self._simulated = cfg.simulated or cfg.gps_port == "sim"
        self._serial = None

    def connect(self) -> bool:
        if self._simulated:
            logger.info("GPS: running in simulated mode")
            return True
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

    async def run(self, callback) -> None:
        """
        Continuously read GPS and call callback(lat, lon, alt_m, heading_deg).
        """
        t_start = time.monotonic()
        while True:
            t = time.monotonic() - t_start
            if self._simulated:
                pos = self._simulated_position(t)
            else:
                pos = self._read_real()
            if pos is not None:
                callback(*pos)
            await asyncio.sleep(self._poll_interval)

    def _simulated_position(self, t: float) -> tuple[float, float, float, float]:
        angle = 2 * math.pi * t / _SIM_PERIOD_S
        lat = _SIM_LAT + _SIM_RADIUS_DEG * math.sin(angle)
        lon = _SIM_LON + _SIM_RADIUS_DEG * math.cos(angle)
        alt_m = 15.0
        heading_deg = (math.degrees(angle) + 90) % 360
        return lat, lon, alt_m, heading_deg

    def _read_real(self) -> tuple[float, float, float, float] | None:
        if self._serial is None:
            return None
        try:
            line = self._serial.readline().decode("ascii", errors="ignore").strip()
            return _parse_gga_or_rmc(line)
        except Exception as exc:
            logger.debug("GPS read error: %s", exc)
            return None

    def disconnect(self) -> None:
        if self._serial is not None:
            self._serial.close()
            self._serial = None


def _parse_gga_or_rmc(sentence: str) -> tuple[float, float, float, float] | None:
    """
    Minimal NMEA GGA / RMC parser.
    Returns (lat, lon, alt_m, heading_deg) or None on parse failure.
    """
    if not sentence.startswith("$"):
        return None
    parts = sentence.split(",")
    try:
        if sentence.startswith("$GPGGA") or sentence.startswith("$GNGGA"):
            lat = _dm_to_decimal(parts[2], parts[3])
            lon = _dm_to_decimal(parts[4], parts[5])
            alt_m = float(parts[9]) if parts[9] else 0.0
            return lat, lon, alt_m, 0.0
        if sentence.startswith("$GPRMC") or sentence.startswith("$GNRMC"):
            lat = _dm_to_decimal(parts[3], parts[4])
            lon = _dm_to_decimal(parts[5], parts[6])
            heading_deg = float(parts[8]) if parts[8] else 0.0
            return lat, lon, 0.0, heading_deg
    except (IndexError, ValueError):
        pass
    return None


def _dm_to_decimal(dm: str, direction: str) -> float:
    """Convert NMEA DDDMM.MMMM + N/S/E/W to signed decimal degrees."""
    if not dm:
        return 0.0
    dot = dm.index(".")
    degrees = float(dm[: dot - 2])
    minutes = float(dm[dot - 2:])
    value = degrees + minutes / 60.0
    if direction in ("S", "W"):
        value = -value
    return value
