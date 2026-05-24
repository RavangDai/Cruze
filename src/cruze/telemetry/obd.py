"""
OBD-II reader with simulator fallback.

Real mode: uses python-OBD to read speed from a connected ELM327 dongle.
Simulated mode: generates a plausible driving speed profile (accelerate,
cruise, brake) on a fixed sine-ish cycle so the rest of the pipeline has
something to reason about without hardware.

Published value: speed in m/s (always SI internally; convert for display).
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

# Simulated drive: a 60-second cycle between 0 and ~27 m/s (≈ 60 mph).
_SIM_PERIOD_S = 60.0
_SIM_MAX_SPEED_MPS = 27.0


def _simulated_speed(t: float) -> float:
    """Return a plausible speed in m/s for time t (seconds)."""
    return max(0.0, _SIM_MAX_SPEED_MPS * (0.5 + 0.5 * math.sin(2 * math.pi * t / _SIM_PERIOD_S)))


class OBDReader:
    """
    Reads vehicle speed from OBD-II.

    Parameters
    ----------
    cfg:
        Telemetry config. If cfg.simulated or cfg.obd_port == "sim", runs in
        simulated mode.
    poll_interval_s:
        How often to query OBD (seconds). ELM327 roundtrip is ~100 ms.
    """

    def __init__(self, cfg: "TelemetryConfig", poll_interval_s: float = 0.1) -> None:
        self._cfg = cfg
        self._poll_interval = poll_interval_s
        self._simulated = cfg.simulated or cfg.obd_port == "sim"
        self._connection = None
        self._speed_mps: float | None = None
        self._source = "simulated" if self._simulated else "obd"

    def connect(self) -> bool:
        if self._simulated:
            logger.info("OBD: running in simulated mode")
            return True
        try:
            import obd  # type: ignore
        except ImportError as exc:
            logger.warning("python-obd not installed — OBD will be unavailable: %s", exc)
            self._simulated = True
            self._source = "simulated"
            return True

        port = None if self._cfg.obd_port == "auto" else self._cfg.obd_port
        conn = obd.OBD(portstr=port, fast=True)
        if not conn.is_connected():
            logger.warning("OBD: could not connect on port=%s — falling back to simulation", port)
            self._simulated = True
            self._source = "simulated"
            return False
        self._connection = conn
        logger.info("OBD: connected on %s", conn.port_name())
        return True

    async def run(self, callback) -> None:
        """
        Poll speed continuously and call callback(speed_mps, source) each tick.
        """
        t_start = time.monotonic()
        while True:
            t = time.monotonic() - t_start
            if self._simulated:
                speed = _simulated_speed(t)
                source = "simulated"
            else:
                speed = self._read_real_speed()
                source = "obd"
            callback(speed, source)
            await asyncio.sleep(self._poll_interval)

    def _read_real_speed(self) -> float | None:
        if self._connection is None:
            return None
        import obd  # type: ignore
        response = self._connection.query(obd.commands.SPEED)
        if response.is_null():
            return None
        return float(response.value.to("m/s").magnitude)

    def disconnect(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
