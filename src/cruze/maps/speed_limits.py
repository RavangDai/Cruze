"""
Speed limit lookup via OSM Overpass API with SQLite offline cache.

Usage pattern:
  1. On each GPS update, call lookup(lat, lon).
  2. Cache hit → instant return.
  3. Cache miss → fire async Overpass query, populate cache, return result.
  4. When offline_only=True or network unavailable, return None (caller handles
     gracefully by skipping speeding checks).

Overpass query: find the nearest highway=* with maxspeed=* within 50 m.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

from cruze.maps.cache import SpeedLimitCache

if TYPE_CHECKING:
    from cruze.core.config import MapsConfig

logger = logging.getLogger(__name__)

# Overpass radius in metres to search for speed limit tags.
_QUERY_RADIUS_M = 50

# Common maxspeed tag values that aren't numeric (country defaults).
_SPECIAL_LIMITS: dict[str, float] = {
    "walk": 1.4,      # pedestrian zone ~3 mph
    "living_street": 2.8,
    "urban": 13.9,    # 50 km/h
    "rural": 22.2,    # 80 km/h
    "motorway": 33.3, # 120 km/h
}


def _parse_maxspeed(value: str) -> float | None:
    """Convert an OSM maxspeed tag to m/s, or None on failure."""
    value = value.strip().lower()
    if value in _SPECIAL_LIMITS:
        return _SPECIAL_LIMITS[value]
    # "mph" suffix → convert.
    if value.endswith(" mph"):
        try:
            return float(value[:-4]) * 0.44704
        except ValueError:
            return None
    # Plain numeric (assumed km/h in OSM).
    try:
        return float(value) / 3.6
    except ValueError:
        return None


class SpeedLimitService:
    """
    Async speed limit lookup.

    Parameters
    ----------
    cfg:
        Maps config block.
    """

    def __init__(self, cfg: "MapsConfig") -> None:
        self._cfg = cfg
        self._cache = SpeedLimitCache(cfg.cache_db_path, cfg.cache_grid_deg)
        self._pending: set[tuple[float, float]] = set()

    async def lookup(self, lat: float, lon: float) -> float | None:
        """
        Return speed limit in m/s for (lat, lon), or None if unknown.
        Non-blocking: cache hit returns immediately; miss fires a background query.
        """
        cached = self._cache.get(lat, lon)
        if cached is not None:
            return cached

        if self._cfg.offline_only:
            return None

        # Fire background query without blocking the caller.
        bucket = self._cache._bucket(lat, lon)
        if bucket not in self._pending:
            self._pending.add(bucket)
            asyncio.ensure_future(self._query_and_cache(lat, lon))

        return None  # caller will get the limit on the next GPS tick

    async def _query_and_cache(self, lat: float, lon: float) -> None:
        try:
            limit_mps = await self._overpass_query(lat, lon)
            if limit_mps is not None:
                self._cache.put(lat, lon, limit_mps)
                logger.debug("Cached speed limit %.1f m/s at (%.4f, %.4f)", limit_mps, lat, lon)
        except Exception as exc:
            logger.warning("Overpass query failed for (%.4f, %.4f): %s", lat, lon, exc)
        finally:
            bucket = self._cache._bucket(lat, lon)
            self._pending.discard(bucket)

    async def _overpass_query(self, lat: float, lon: float) -> float | None:
        query = (
            f"[out:json][timeout:5];"
            f"way(around:{_QUERY_RADIUS_M},{lat},{lon})[maxspeed];"
            f"out tags 1;"
        )
        try:
            import urllib.request
            import urllib.parse
            data = urllib.parse.urlencode({"data": query}).encode()
            # Run the blocking HTTP call in a thread so we don't block the loop.
            response_bytes = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: urllib.request.urlopen(
                    self._cfg.overpass_url, data=data, timeout=6
                ).read(),
            )
            result = json.loads(response_bytes)
            for element in result.get("elements", []):
                tags = element.get("tags", {})
                raw = tags.get("maxspeed", "")
                if raw:
                    limit = _parse_maxspeed(raw)
                    if limit is not None:
                        return limit
        except Exception as exc:
            raise RuntimeError(f"Overpass HTTP error: {exc}") from exc
        return None

    def close(self) -> None:
        self._cache.close()
