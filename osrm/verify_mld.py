"""Verify the self-hosted OSRM MLD server returns native alternative routes."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from spatial.router import OSRMRouter
from spatial.corridor import build_corridor

CACHE = ROOT / "data" / "route_cache_mld_test.db"

ORIGIN = (40.758, -73.985)  # Times Square
DEST = (40.678, -73.944)    # Prospect Park, Brooklyn


def main() -> None:
    router = OSRMRouter(cache_path=CACHE)
    print(f"OSRM base URL: {router._base_url}")
    print(f"Rate limiting: {router._rate_limit}")
    print()

    print("--- Requesting 3 alternatives (native MLD) ---")
    routes = router.get_alternative_routes(ORIGIN, DEST, max_alternatives=3)
    print(f"Routes returned: {len(routes)}")
    print()

    for i, r in enumerate(routes):
        corridor = build_corridor(r.polyline)
        print(f"Route {i}:")
        print(f"  distance   : {r.distance_m:.0f} m ({r.distance_m / 1609.34:.1f} mi)")
        print(f"  duration   : {r.duration_s:.0f} s ({r.duration_s / 60:.1f} min)")
        print(f"  polyline   : {len(r.polyline)} points")
        print(f"  corridor   : {corridor.n_route_cells} route / {corridor.n_corridor_cells} total cells")
        print()

    if len(routes) >= 2:
        print("SUCCESS: MLD server returns multiple native alternatives.")
        print("No waypoint fallback needed.")
    else:
        print("WARNING: Only 1 route returned. MLD may not find alternatives")
        print("for this specific O-D pair. Try a different pair or check server logs.")

    print(f"\nAPI calls: {router.api_calls}")

    router.flush_cache()
    try:
        if CACHE.exists():
            CACHE.unlink()
    except OSError:
        pass


if __name__ == "__main__":
    main()
