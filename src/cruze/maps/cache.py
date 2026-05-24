"""
SQLite-backed speed limit cache.

Spatial index: lat/lon bucketed to a configurable grid cell size (~100 m at
0.001° ≈ 111 m). Cache hit = O(1) lookup; miss triggers an Overpass query.
"""

from __future__ import annotations

import logging
import math
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


class SpeedLimitCache:
    """
    Persistent cache mapping (lat_bucket, lon_bucket) → speed_limit_mps.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.
    grid_deg:
        Cell size in degrees. 0.001° ≈ 111 m along a meridian.
    """

    def __init__(self, db_path: str = "cache/speed_limits.db", grid_deg: float = 0.001) -> None:
        self._grid = grid_deg
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS speed_limits (
                lat_bucket REAL NOT NULL,
                lon_bucket REAL NOT NULL,
                limit_mps  REAL NOT NULL,
                PRIMARY KEY (lat_bucket, lon_bucket)
            )"""
        )
        self._conn.commit()

    def _bucket(self, lat: float, lon: float) -> tuple[float, float]:
        return (
            round(math.floor(lat / self._grid) * self._grid, 6),
            round(math.floor(lon / self._grid) * self._grid, 6),
        )

    def get(self, lat: float, lon: float) -> float | None:
        lb, lob = self._bucket(lat, lon)
        row = self._conn.execute(
            "SELECT limit_mps FROM speed_limits WHERE lat_bucket=? AND lon_bucket=?",
            (lb, lob),
        ).fetchone()
        return float(row[0]) if row else None

    def put(self, lat: float, lon: float, limit_mps: float) -> None:
        lb, lob = self._bucket(lat, lon)
        self._conn.execute(
            "INSERT OR REPLACE INTO speed_limits VALUES (?, ?, ?)",
            (lb, lob, limit_mps),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
