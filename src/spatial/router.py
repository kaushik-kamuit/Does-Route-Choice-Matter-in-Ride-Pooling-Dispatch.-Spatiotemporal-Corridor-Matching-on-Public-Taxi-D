"""
Route fetching via OSRM (Open Source Routing Machine).

Produces road-following polylines for driver origin->destination pairs.
All responses are cached to disk so repeated runs cost zero API calls.

Self-hosted MLD server: set OSRM_BASE_URL env var to your instance.
MLD natively returns genuine alternative routes -- no waypoint hack needed.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import requests

from .h3_utils import LatLng

OSRM_DEFAULT_URL = "https://router.project-osrm.org"
COORD_PRECISION = 4  # ~11m resolution for cache keys
MIN_REQUEST_INTERVAL_S = 1.1  # respect public server rate limit


@dataclass(frozen=True, slots=True)
class RouteInfo:
    polyline: tuple[LatLng, ...]
    distance_m: float
    duration_s: float

    def to_dict(self) -> dict:
        return {
            "polyline": [list(p) for p in self.polyline],
            "distance_m": self.distance_m,
            "duration_s": self.duration_s,
        }

    @classmethod
    def from_dict(cls, d: dict) -> RouteInfo:
        return cls(
            polyline=tuple(tuple(p) for p in d["polyline"]),
            distance_m=d["distance_m"],
            duration_s=d["duration_s"],
        )


class RouteCache:
    """SQLite-backed key-value cache for OSRM route responses.

    Near-zero memory footprint: only the single entry being read/written is
    in memory at any time.  The previous JSON implementation loaded the
    entire cache (~2 GB) into RAM which caused OOM on most machines.
    """

    COMMIT_EVERY = 20

    def __init__(self, path: Path):
        import sqlite3
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS routes "
            "(cache_key TEXT PRIMARY KEY, routes_json TEXT NOT NULL)"
        )
        self._conn.commit()
        self._writes_since_commit = 0
        n = self.size
        print(f"  Route cache opened: {n:,} entries in {path.name}")

    def flush(self) -> None:
        self._conn.commit()
        self._writes_since_commit = 0

    def get(self, key: str) -> list[RouteInfo] | None:
        row = self._conn.execute(
            "SELECT routes_json FROM routes WHERE cache_key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        return [RouteInfo.from_dict(r) for r in json.loads(row[0])]

    def put(self, key: str, routes: list[RouteInfo]) -> None:
        data = json.dumps([r.to_dict() for r in routes], separators=(",", ":"))
        self._conn.execute(
            "INSERT OR REPLACE INTO routes VALUES (?, ?)", (key, data)
        )
        self._writes_since_commit += 1
        if self._writes_since_commit >= self.COMMIT_EVERY:
            self._conn.commit()
            self._writes_since_commit = 0

    def close(self) -> None:
        self._conn.commit()
        self._conn.close()

    @property
    def size(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM routes").fetchone()[0]


def _make_cache_key(origin: LatLng, dest: LatLng, n_alternatives: int) -> str:
    o = f"{origin[0]:.{COORD_PRECISION}f},{origin[1]:.{COORD_PRECISION}f}"
    d = f"{dest[0]:.{COORD_PRECISION}f},{dest[1]:.{COORD_PRECISION}f}"
    return f"{o}->{d}|alt={n_alternatives}"


def _decode_polyline5(encoded: str) -> list[LatLng]:
    """Decode a Google-encoded polyline (precision 5) into (lat, lng) tuples."""
    points: list[LatLng] = []
    idx, lat, lng = 0, 0, 0
    while idx < len(encoded):
        for coord_ref in range(2):
            shift, result = 0, 0
            while True:
                b = ord(encoded[idx]) - 63
                idx += 1
                result |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:
                    break
            delta = ~(result >> 1) if (result & 1) else (result >> 1)
            if coord_ref == 0:
                lat += delta
            else:
                lng += delta
        points.append((lat / 1e5, lng / 1e5))
    return points


def _parse_osrm_routes(data: dict) -> list[RouteInfo]:
    routes: list[RouteInfo] = []
    for r in data.get("routes", []):
        polyline_pts = _decode_polyline5(r["geometry"])
        routes.append(RouteInfo(
            polyline=tuple(polyline_pts),
            distance_m=r["distance"],
            duration_s=r["duration"],
        ))
    return routes


class OSRMRouter:
    """
    Fetch driving routes from an OSRM server with disk-backed caching.

    Both cold-start and warm-up share a single alt=3 request per O-D pair,
    ensuring a fair paired comparison (same routes[0] baseline).

    Usage:
        router = OSRMRouter()
        route = router.get_default_route(origin, dest)       # cold-start
        routes = router.get_alternative_routes(origin, dest)  # warm-up
        router.flush_cache()
    """

    N_ALTERNATIVES = 3

    def __init__(
        self,
        base_url: str | None = None,
        cache_path: Path | str = "data/route_cache.db",
        rate_limit: bool = True,
        cache_only: bool = False,
    ):
        self._base_url = (
            base_url
            or os.environ.get("OSRM_BASE_URL")
            or OSRM_DEFAULT_URL
        ).rstrip("/")
        self._cache = RouteCache(Path(cache_path))
        self._cache_only = cache_only or bool(os.environ.get("CARPOOL_NO_API"))
        self._rate_limit = rate_limit and (OSRM_DEFAULT_URL in self._base_url)
        self._last_request_time = 0.0
        self._api_calls = 0

    def _throttle(self) -> None:
        if not self._rate_limit:
            return
        elapsed = time.time() - self._last_request_time
        if elapsed < MIN_REQUEST_INTERVAL_S:
            time.sleep(MIN_REQUEST_INTERVAL_S - elapsed)

    def _fetch(
        self, origin: LatLng, dest: LatLng, alternatives: int, _retries: int = 3
    ) -> list[RouteInfo]:
        coords = f"{origin[1]},{origin[0]};{dest[1]},{dest[0]}"
        alt_param = str(alternatives) if alternatives > 1 else "false"
        url = (
            f"{self._base_url}/route/v1/driving/{coords}"
            f"?alternatives={alt_param}"
            f"&geometries=polyline&overview=full"
        )

        last_err: Exception | None = None
        for attempt in range(_retries):
            self._throttle()
            try:
                resp = requests.get(url, timeout=30)
                self._last_request_time = time.time()
                self._api_calls += 1
                resp.raise_for_status()
                data = resp.json()
                if data.get("code") != "Ok":
                    raise RuntimeError(f"OSRM error: {data.get('code')} - {data.get('message')}")
                return _parse_osrm_routes(data)[:alternatives]
            except (requests.RequestException, RuntimeError) as e:
                last_err = e
                wait = MIN_REQUEST_INTERVAL_S * (2 ** attempt)
                time.sleep(wait)

        raise last_err  # type: ignore[misc]

    def get_default_route(
        self, origin: LatLng, dest: LatLng
    ) -> RouteInfo | None:
        """Fetch the default (fastest) route -- cold-start mode.

        Uses the same alt=3 cache key as warm-up so both strategies
        share an identical routes[0] baseline for fair comparison.
        Returns None when no routes are available (cache miss in cache-only mode).
        """
        routes = self._get_routes(origin, dest)
        if not routes:
            return None
        return routes[0]

    def get_alternative_routes(
        self, origin: LatLng, dest: LatLng, max_alternatives: int = 3
    ) -> list[RouteInfo]:
        """Fetch up to max_alternatives routes (warm-up mode)."""
        return self._get_routes(origin, dest)[:max_alternatives]

    def _get_routes(
        self, origin: LatLng, dest: LatLng
    ) -> list[RouteInfo]:
        n = self.N_ALTERNATIVES
        key = _make_cache_key(origin, dest, n)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        if self._cache_only:
            return []

        routes = self._fetch(origin, dest, n)
        self._cache.put(key, routes)
        return routes

    # ------------------------------------------------------------------
    # Future feature: route deduplication by corridor overlap.
    # Implemented but NOT called in the active pipeline.
    # ------------------------------------------------------------------

    def _deduplicate_routes(
        self, routes: list[RouteInfo], threshold: float = 0.80
    ) -> list[RouteInfo]:
        """Filter near-duplicate routes by corridor cell overlap (Jaccard).

        Not used in the current experiment -- MLD alternatives are trusted
        as-is. Can be enabled post-paper for route diversity analysis.
        """
        from .h3_utils import polyline_to_h3_cells
        kept: list[RouteInfo] = []
        kept_cells: list[set[str]] = []
        for r in routes:
            cells = set(polyline_to_h3_cells(r.polyline))
            is_dup = False
            for existing in kept_cells:
                jaccard = len(cells & existing) / max(len(cells | existing), 1)
                if jaccard > threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(r)
                kept_cells.append(cells)
        return kept

    # ------------------------------------------------------------------

    def flush_cache(self) -> None:
        """Write any new cache entries to disk."""
        self._cache.flush()

    @property
    def cache_size(self) -> int:
        return self._cache.size

    @property
    def api_calls(self) -> int:
        return self._api_calls
