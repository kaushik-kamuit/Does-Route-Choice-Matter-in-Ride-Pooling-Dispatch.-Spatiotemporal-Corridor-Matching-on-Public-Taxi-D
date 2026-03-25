"""
Build an interactive HTML map showing 3 alternate routes and their H3 corridors.

Uses a long route from the OSRM route cache (with at least 3 alternatives) so the
corridor geometry and matching intuition are easy to see.
Output: results/plots/corridor_map.html

Usage:
    python visualizations/plot_corridor_map.py           # longest route with 3+ alts (default)
    python visualizations/plot_corridor_map.py --index 1  # 2nd-longest such route
    python visualizations/plot_corridor_map.py --index 4  # 5th-longest such route
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
CACHE_PATH = ROOT / "data" / "route_cache.db"

sys.path.insert(0, str(ROOT / "src"))

import h3
from spatial.corridor import build_corridor
from spatial.router import RouteInfo


def _h3_cell_to_geojson_polygon(cell: str) -> list[list[float]]:
    """Return GeoJSON exterior ring: [[lng, lat], ...], closed."""
    boundary = h3.cell_to_boundary(cell)
    # GeoJSON: [lng, lat]; close the ring
    ring = [[float(p[1]), float(p[0])] for p in boundary]
    ring.append(ring[0])
    return ring


def _find_long_routes_from_cache(
    max_rows: int = 5000,
    route_index: int = 0,
    min_alternatives: int = 3,
) -> tuple[str, list[dict]]:
    """Return (cache_key, list of route dicts) for a long route with multiple alternatives.
    Only considers cache entries with at least min_alternatives routes.
    route_index=0 -> longest such route, 1 -> 2nd longest, etc.
    """
    if not CACHE_PATH.exists():
        raise FileNotFoundError(f"Route cache not found: {CACHE_PATH}")
    conn = sqlite3.connect(str(CACHE_PATH))
    rows = conn.execute(
        "SELECT cache_key, routes_json FROM routes ORDER BY length(routes_json) DESC LIMIT ?",
        (max_rows,),
    ).fetchall()
    conn.close()
    candidates = []
    for key, routes_json in rows:
        routes = json.loads(routes_json)
        if len(routes) < min_alternatives:
            continue
        dist = routes[0].get("distance_m", 0) or 0
        candidates.append((dist, key, routes_json))
    candidates.sort(key=lambda x: -x[0])
    if not candidates:
        # Fallback: try with fewer alternatives so we still produce a map
        for fallback_min in [2, 1]:
            if fallback_min >= min_alternatives:
                continue
            for key, routes_json in rows:
                routes = json.loads(routes_json)
                if len(routes) < fallback_min:
                    continue
                dist = routes[0].get("distance_m", 0) or 0
                candidates.append((dist, key, routes_json))
            if candidates:
                candidates.sort(key=lambda x: -x[0])
                break
    if not candidates:
        raise ValueError(
            "No cache entry has multiple alternative routes. "
            "Populate data/route_cache.db with OSRM routes (alternatives enabled)."
        )
    if route_index >= len(candidates):
        route_index = len(candidates) - 1
    _, best_key, best_routes_json = candidates[route_index]
    return best_key, json.loads(best_routes_json)


MAX_CELLS_PER_CORRIDOR = 600  # cap so HTML stays small and renders quickly


def _build_corridor_geojson(corridor_cells: set[str]) -> dict:
    """GeoJSON FeatureCollection of polygons for each H3 cell."""
    cells = list(corridor_cells)
    if len(cells) > MAX_CELLS_PER_CORRIDOR:
        step = len(cells) / MAX_CELLS_PER_CORRIDOR
        cells = [cells[int(i * step)] for i in range(MAX_CELLS_PER_CORRIDOR)]
    features = []
    for cell in cells:
        ring = _h3_cell_to_geojson_polygon(cell)
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": {},
        })
    return {"type": "FeatureCollection", "features": features}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build corridor map for 3 alternate routes")
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Which long route to use: 0=longest with 3+ alts (default), 1=2nd longest, ...",
    )
    parser.add_argument(
        "--min-alternatives",
        type=int,
        default=3,
        help="Require at least this many route alternatives (default: 3).",
    )
    args = parser.parse_args()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Finding long route with 3+ alternatives in cache...")
    cache_key, routes_dicts = _find_long_routes_from_cache(
        route_index=args.index, min_alternatives=args.min_alternatives
    )
    routes = [RouteInfo.from_dict(d) for d in routes_dicts[:3]]
    if not routes:
        raise ValueError("No routes to display")
    distance_km = routes[0].distance_m / 1000
    rank = args.index + 1
    print(f"  Using #{rank} long route: {cache_key[:60]}... (route 0: {distance_km:.1f} km)")

    print("Building corridors for 3 alternatives...")
    corridors = [build_corridor(r.polyline) for r in routes]
    origin = routes[0].polyline[0]
    dest = routes[0].polyline[-1]

    # GeoJSON for each corridor (corridor_cells = matchable zone)
    corridor_geojsons = [_build_corridor_geojson(set(c.corridor_cells)) for c in corridors]
    # Polylines: list of [lat, lng] for Leaflet
    polylines = [[list(p) for p in r.polyline] for r in routes]

    # Colors: route 1 (default), 2, 3 — distinct and colorblind-friendly
    colors = ["#4C72B0", "#DD8452", "#55A868"]  # blue, orange, green
    names = ["Route 1 (default)", "Route 2 (alt)", "Route 3 (alt)"]

    html_content = _make_html(
        corridor_geojsons=corridor_geojsons,
        polylines=polylines,
        colors=colors,
        names=names,
        origin=origin,
        dest=dest,
        distance_km=distance_km,
    )
    out_path = PLOTS_DIR / "corridor_map.html"
    out_path.write_text(html_content, encoding="utf-8")
    print(f"  Saved: {out_path}")


def _make_html(
    corridor_geojsons: list[dict],
    polylines: list[list[list[float]]],
    colors: list[str],
    names: list[str],
    origin: tuple[float, float],
    dest: tuple[float, float],
    distance_km: float,
) -> str:
    """Self-contained HTML with Leaflet (CDN)."""
    origin_js = list(origin)
    dest_js = list(dest)
    geojsons_js = json.dumps(corridor_geojsons)
    polylines_js = json.dumps(polylines)
    colors_js = json.dumps(colors)
    names_js = json.dumps(names)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Corridors for 3 alternate routes (long route, {distance_km:.1f} km)</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    body {{ margin: 0; font-family: system-ui, sans-serif; }}
    #map {{ height: 100vh; }}
    .legend {{ position: absolute; bottom: 24px; left: 12px; z-index: 1000; background: white; padding: 10px 14px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.2); font-size: 13px; }}
    .legend h4 {{ margin: 0 0 6px 0; }}
    .legend span {{ display: inline-block; width: 14px; height: 14px; margin-right: 6px; vertical-align: middle; border: 1px solid #333; }}
    .legend p {{ margin: 4px 0 0 0; color: #555; }}
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="legend">
    <h4>3 alternate routes &amp; H3 corridors</h4>
    <p><span style="background:#4C72B0;"></span> Route 1 (default) — corridor = matchable zone</p>
    <p><span style="background:#DD8452;"></span> Route 2 (alt)</p>
    <p><span style="background:#55A868;"></span> Route 3 (alt)</p>
    <p>Hexes = H3 resolution 9 (~174 m edge). Matching: riders with pickup and dropoff inside a corridor can be matched.</p>
  </div>
  <script>
    const corridorGeojsons = {geojsons_js};
    const polylines = {polylines_js};
    const colors = {colors_js};
    const names = {names_js};
    const origin = {origin_js};
    const dest = {dest_js};

    const map = L.map("map").setView([(origin[0] + dest[0]) / 2, (origin[1] + dest[1]) / 2], 11);
    L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
      attribution: "&copy; OpenStreetMap"
    }}).addTo(map);

    corridorGeojsons.forEach((geojson, i) => {{
      const layer = L.geoJSON(geojson, {{
        style: {{ fillColor: colors[i], color: colors[i], weight: 1.5, fillOpacity: 0.35 }}
      }});
      layer.addTo(map);
    }});

    polylines.forEach((pts, i) => {{
      L.polyline(pts, {{ color: colors[i], weight: 4, opacity: 0.9 }}).addTo(map);
    }});

    L.marker(origin).addTo(map).bindPopup("Origin");
    L.marker(dest).addTo(map).bindPopup("Destination");
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
