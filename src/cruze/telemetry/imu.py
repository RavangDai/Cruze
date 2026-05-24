"""
IMU reader — accelerometer / gyroscope.

Real mode: reads from an I2C IMU (MPU-6050 or similar via smbus2).
Simulated mode: returns gentle acceleration that tracks the OBD speed changes.

Used for:
  - Longitudinal acceleration (harsh braking detection)
  - Lateral acceleration (sharp turns — Phase 2)
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

_SIM_PERIOD_S = 60.0


class IMUReader:
    def __init__(self, cfg: "TelemetryConfig", poll_interval_s: float = 0.05) -> None:
        self._cfg = cfg
        self._poll_interval = poll_interval_s
        self._simulated = cfg.simulated or cfg.imu_port == "sim"

    def connect(self) -> bool:
        if self._simulated:
            logger.info("IMU: running in simulated mode")
            return True
        logger.info("IMU: real IMU reading not yet implemented — simulating")
        self._simulated = True
        return True

    async def run(self, callback) -> None:
        """Call callback(accel_mps2) at poll_interval_s rate."""
        t_start = time.monotonic()
        while True:
            t = time.monotonic() - t_start
            if self._simulated:
                accel = self._simulated_accel(t)
            else:
                accel = 0.0
            callback(accel)
            await asyncio.sleep(self._poll_interval)

    def _simulated_accel(self, t: float) -> float:
        """Derivative of the simulated OBD speed profile ≈ longitudinal accel."""
        omega = 2 * math.pi / _SIM_PERIOD_S
        max_speed = 27.0
        return max_speed * omega * 0.5 * math.cos(omega * t)
