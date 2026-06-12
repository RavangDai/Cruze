"""
Vehicle state service.

Fuses OBD speed, GPS position, and IMU acceleration into a single
VehicleState snapshot, then publishes it to Channel.TELEMETRY_VEHICLE_STATE.

Speed source priority (see telemetry/fusion.py):
  1. OBD (most accurate, synchronized to wheel speed sensor)
  2. GPS Doppler (RMC speed-over-ground — cable-free, ~0.3 m/s accuracy)
  3. GPS position-derived (haversine Δposition / Δtime — noisiest fallback)

A source is used only while fresh (STALENESS_S); otherwise fusion falls
through to the next. GPS always provides position and heading.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from cruze.core.bus import Channel, EventBus
from cruze.core.types import VehicleState
from cruze.telemetry.fusion import SpeedSample, fuse_speed, position_speed_mps
from cruze.telemetry.gps import GPSReader
from cruze.telemetry.imu import IMUReader
from cruze.telemetry.obd import OBDReader

if TYPE_CHECKING:
    from cruze.core.config import Config

logger = logging.getLogger(__name__)


class VehicleStateService:
    """
    Continuously fuses telemetry sources and publishes VehicleState.

    Parameters
    ----------
    cfg:
        Full application config.
    bus:
        Shared event bus.
    publish_interval_s:
        How often to publish a fused state snapshot.
    """

    def __init__(
        self,
        cfg: "Config",
        bus: EventBus,
        publish_interval_s: float = 0.1,
    ) -> None:
        self._cfg = cfg
        self._bus = bus
        self._publish_interval = publish_interval_s
        self._running = False

        self._obd = OBDReader(cfg.telemetry)
        self._gps = GPSReader(cfg.telemetry)
        self._imu = IMUReader(cfg.telemetry)

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

        # Speed limit is injected by the maps module via set_posted_limit().
        self._posted_speed_limit_mps: float | None = None

    def set_posted_limit(self, limit_mps: float | None) -> None:
        self._posted_speed_limit_mps = limit_mps

    async def run(self) -> None:
        self._running = True

        # Connect hardware (or simulators).
        self._obd.connect()
        self._gps.connect()
        self._imu.connect()

        # Kick off reader coroutines + the publisher concurrently.
        await asyncio.gather(
            self._obd.run(self._on_obd),
            self._gps.run(self._on_gps),
            self._imu.run(self._on_imu),
            self._publish_loop(),
        )

    async def stop(self) -> None:
        self._running = False
        self._obd.disconnect()
        self._gps.disconnect()

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

    def _on_imu(self, accel_mps2: float) -> None:
        self._accel_mps2 = accel_mps2

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
            # Raw Doppler speed exposed alongside the fused value; reuse
            # fuse_speed so the staleness rule lives in one place.
            gps_speed, _ = fuse_speed([(self._gps_doppler_sample, "gps")], now)
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
