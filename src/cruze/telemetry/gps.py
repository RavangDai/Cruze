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
_SIM_RADIUS_DEG = 0.005   # ~555 m radius (0.005° × 111 km/°)
_SIM_PERIOD_S = 120.0     # one lap in 2 minutes

# 1 knot = 1852 m / 3600 s.
_KNOTS_TO_MPS = 0.514444

# Tangential speed of the simulated circular route: circumference / period.
# 0.005° ≈ 555 m radius → 2π·555 m / 120 s ≈ 29 m/s (≈ 65 mph).
_SIM_SPEED_MPS = 2 * math.pi * (_SIM_RADIUS_DEG * 111_000.0) / _SIM_PERIOD_S


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
        self._transport, self._host, self._port = _parse_endpoint(cfg.gps_port)
        self._simulated = cfg.simulated or self._transport == "sim"
        self._serial = None
        self._sock = None
        self._tcp_buf = b""

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
        if self._sock is not None:
            self._sock.close()
            self._sock = None
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

    def _simulated_position(self, t: float) -> tuple[float, float, float, float, float]:
        angle = 2 * math.pi * t / _SIM_PERIOD_S
        lat = _SIM_LAT + _SIM_RADIUS_DEG * math.sin(angle)
        lon = _SIM_LON + _SIM_RADIUS_DEG * math.cos(angle)
        alt_m = 15.0
        heading_deg = (math.degrees(angle) + 90) % 360
        return lat, lon, alt_m, heading_deg, _SIM_SPEED_MPS

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
        sock = self._sock
        if sock is None:
            return None
        try:
            data, _addr = sock.recvfrom(4096)
        except (socket.timeout, OSError):
            return None
        return _parse_nmea_burst(data)

    def _read_tcp(self) -> tuple[float, float, float, float, float | None] | None:
        import socket
        sock = self._sock
        if sock is None:
            return None
        try:
            chunk = sock.recv(4096)
        except (socket.timeout, OSError):
            return None
        if not chunk:
            return None
        self._tcp_buf += chunk
        # Cap reassembly buffer: NMEA sentences are <83 bytes, so 64 KiB of
        # data without a newline means the stream is not NMEA — drop it.
        if len(self._tcp_buf) > 65536:
            self._tcp_buf = b""
            return None
        complete, sep, self._tcp_buf = self._tcp_buf.rpartition(b"\n")
        if not sep:
            return None
        return _parse_nmea_burst(complete)

    def disconnect(self) -> None:
        if self._serial is not None:
            self._serial.close()
            self._serial = None
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        self._tcp_buf = b""


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
    Selection rule: the latest sentence that carries speed (RMC) wins;
    if none carries speed, the first parseable fix is kept.
    Returns None if nothing parsed.
    """
    fix = None
    for line in data.decode("ascii", errors="ignore").splitlines():
        parsed = _parse_gga_or_rmc(line.strip())
        if parsed is not None and (fix is None or parsed[4] is not None):
            fix = parsed
    return fix


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
            return lat, lon, 0.0, heading_deg, speed_mps  # RMC carries no altitude field
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
