"""
Rebuild the ML training dataset with enhanced features.

Removes: matched_rider_count, feasible_rider_count (matching-output features)
Adds:
  - Spatial: corridor demand stats from H3 pre-aggregation, origin/dest cell stats
  - Temporal: 15-min bin, day_of_month, sin/cos hour encoding
  - Geometric: route sinuosity, bearing, speed ratio
  - Landmark: distance to JFK, LGA, Penn Station, Times Square, Grand Central
  - Metadata: service_month and service_date for temporal holdout selection
"""
import sys
import time
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import h3
import numpy as np
import pandas as pd
from tqdm import tqdm

from data_prep.domain_config import get_domain_config
from spatial.router import OSRMRouter
from spatial.corridor import build_corridor
from spatial.h3_utils import haversine_m
from matching.rider_index import RiderIndex
from matching.matcher import COST_PER_MILE, METERS_PER_MILE, PLATFORM_SHARE, match_riders

SAMPLE_SEED = 42
DEFAULT_SAMPLE = 100_000
CHECKPOINT_EVERY = 5_000

LANDMARKS = {
    "jfk":          (40.6413, -73.7781),
    "lga":          (40.7769, -73.8740),
    "penn_station":  (40.7505, -73.9935),
    "times_square":  (40.7580, -73.9855),
    "grand_central": (40.7527, -73.9772),
    "world_trade":   (40.7127, -74.0134),
}


def bearing_deg(lat1, lng1, lat2, lng2):
    """Initial bearing from point 1 to point 2 in degrees [0, 360)."""
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlng = math.radians(lng2 - lng1)
    x = math.sin(dlng) * math.cos(rlat2)
    y = math.cos(rlat1) * math.sin(rlat2) - math.sin(rlat1) * math.cos(rlat2) * math.cos(dlng)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def min_landmark_dist(lat, lng):
    dists = {}
    for name, (llat, llng) in LANDMARKS.items():
        dists[name] = haversine_m((lat, lng), (llat, llng)) / 1000.0
    nearest = min(dists, key=dists.get)
    return dists[nearest], nearest, dists


def corridor_demand_from_stats(corridor_cells, h3_stats_df, h3_stats_dict, hour):
    """Aggregate pre-computed H3 cell stats for the corridor cells."""
    total_pickups = 0
    total_dropoffs = 0
    fare_sum = 0.0
    fare_values = []
    hour_col = f"h{hour}"

    for cell in corridor_cells:
        row = h3_stats_dict.get(cell)
        if row is None:
            continue
        total_pickups += row["pickup_count"]
        total_dropoffs += row["dropoff_count"]
        fare_sum += row["mean_fare"] * row["pickup_count"]
        if row["pickup_count"] > 0:
            fare_values.append(row["mean_fare"])

    n_cells = len(corridor_cells)
    mean_cell_fare = np.mean(fare_values) if fare_values else 0.0
    pickup_density = total_pickups / max(n_cells, 1)
    dropoff_density = total_dropoffs / max(n_cells, 1)

    return {
        "corridor_hist_pickups": total_pickups,
        "corridor_hist_dropoffs": total_dropoffs,
        "corridor_hist_pickup_density": pickup_density,
        "corridor_hist_dropoff_density": dropoff_density,
        "corridor_hist_mean_fare": mean_cell_fare,
        "corridor_hist_fare_density": fare_sum / max(n_cells, 1),
    }


def extract_features(
    driver_id, route_idx, route, corridor, rider_index,
    query_datetime, minute_of_day, hour, day_of_week, is_weekend, day_of_month,
    origin, destination, h3_stats_dict,
    candidate_window_bins: int = 1,
    max_request_offset_min: int | None = None,
    max_detour_min: float = 4.0,
    platform_share: float = PLATFORM_SHARE,
    cost_per_mile: float = COST_PER_MILE,
    urban_speed_kmh: float = 40.0,
    seats: int = 3,
):
    candidates = rider_index.find_in_corridor(
        corridor.corridor_cells,
        minute_of_day,
        window_bins=candidate_window_bins,
        max_request_offset_min=max_request_offset_min,
        query_datetime=query_datetime,
    )
    n_riders = len(candidates)
    n_cells = corridor.n_corridor_cells
    fares = candidates["fare_amount"].values if n_riders > 0 else np.array([])
    fare_sum = float(fares.sum()) if n_riders > 0 else 0.0

    matched, feasible = match_riders(
        corridor, route.polyline, rider_index, minute_of_day,
        query_datetime=query_datetime,
        seats=seats, max_detour_min=max_detour_min,
        candidate_window_bins=candidate_window_bins,
        max_request_offset_min=max_request_offset_min,
        platform_share=platform_share,
        urban_speed_kmh=urban_speed_kmh,
        seed=SAMPLE_SEED,
        candidates=candidates,
    )

    total_revenue = sum(m["fare_share"] for m in matched)
    driver_cost = route.distance_m / METERS_PER_MILE * cost_per_mile
    expected_profit = total_revenue - driver_cost

    straight_dist = haversine_m(origin, destination)
    sinuosity = route.distance_m / max(straight_dist, 1.0)
    avg_speed = route.distance_m / max(route.duration_s, 1.0)

    brg = bearing_deg(origin[0], origin[1], destination[0], destination[1])
    bearing_sin = math.sin(math.radians(brg))
    bearing_cos = math.cos(math.radians(brg))

    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)

    qh = hour * 4 + (minute_of_day % 60) // 15

    origin_dist, origin_nearest, origin_dists = min_landmark_dist(*origin)
    dest_dist, dest_nearest, dest_dists = min_landmark_dist(*destination)

    corridor_hist = corridor_demand_from_stats(
        corridor.corridor_cells, None, h3_stats_dict, hour
    )

    o_stats = h3_stats_dict.get(h3.latlng_to_cell(origin[0], origin[1], 9), {})
    d_stats = h3_stats_dict.get(h3.latlng_to_cell(destination[0], destination[1], 9), {})

    row = {
        "driver_id": driver_id,
        "route_idx": route_idx,
        "service_month": int(query_datetime.month),
        "service_date": pd.Timestamp(query_datetime).normalize(),
        "route_distance_m": route.distance_m,
        "route_duration_s": route.duration_s,
        "corridor_cell_count": n_cells,
        "hour_of_day": hour,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "corridor_rider_count": n_riders,
        "corridor_demand_density": n_riders / max(n_cells, 1),
        "mean_rider_fare": float(fares.mean()) if n_riders > 0 else 0.0,
        "corridor_fare_density": fare_sum / max(n_cells, 1),

        "day_of_month": day_of_month,
        "time_bin_15min": qh,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,

        "route_sinuosity": sinuosity,
        "route_avg_speed_ms": avg_speed,
        "bearing_sin": bearing_sin,
        "bearing_cos": bearing_cos,
        "straight_line_dist_m": straight_dist,

        "origin_landmark_dist_km": origin_dist,
        "dest_landmark_dist_km": dest_dist,
        "origin_jfk_km": origin_dists["jfk"],
        "origin_lga_km": origin_dists["lga"],
        "origin_penn_km": origin_dists["penn_station"],
        "origin_times_sq_km": origin_dists["times_square"],
        "dest_jfk_km": dest_dists["jfk"],
        "dest_lga_km": dest_dists["lga"],
        "dest_penn_km": dest_dists["penn_station"],
        "dest_times_sq_km": dest_dists["times_square"],

        "corridor_hist_pickups": corridor_hist["corridor_hist_pickups"],
        "corridor_hist_dropoffs": corridor_hist["corridor_hist_dropoffs"],
        "corridor_hist_pickup_density": corridor_hist["corridor_hist_pickup_density"],
        "corridor_hist_dropoff_density": corridor_hist["corridor_hist_dropoff_density"],
        "corridor_hist_mean_fare": corridor_hist["corridor_hist_mean_fare"],
        "corridor_hist_fare_density": corridor_hist["corridor_hist_fare_density"],

        "origin_cell_pickups": o_stats.get("pickup_count", 0),
        "origin_cell_mean_fare": o_stats.get("mean_fare", 0.0),
        "dest_cell_dropoffs": d_stats.get("dropoff_count", 0),

        "expected_revenue": total_revenue,
        "driver_cost": driver_cost,
        "expected_profit": expected_profit,
    }
    return row


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="yellow", choices=["yellow", "green"])
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--index-bin-minutes", type=int, default=15)
    parser.add_argument("--candidate-window-bins", type=int, default=1)
    parser.add_argument("--max-request-offset-min", type=int, default=None)
    parser.add_argument("--max-detour-min", type=float, default=4.0)
    parser.add_argument("--platform-share", type=float, default=PLATFORM_SHARE)
    parser.add_argument("--cost-per-mile", type=float, default=COST_PER_MILE)
    parser.add_argument("--urban-speed-kmh", type=float, default=40.0)
    parser.add_argument("--h3-resolution", type=int, default=9)
    parser.add_argument("--corridor-k-ring", type=int, default=1)
    parser.add_argument("--corridor-densify-step-m", type=float, default=80.0)
    parser.add_argument("--seats", type=int, default=3)
    parser.add_argument("--max-riders", type=int, default=None)
    parser.add_argument("--fetch-routes", action="store_true", help="Fetch missing OSRM routes instead of cache-only mode")
    args = parser.parse_args()

    domain_config = get_domain_config(args.domain)
    drivers_path = domain_config.drivers_path()
    riders_path = domain_config.riders_path()
    cache_path = domain_config.route_cache_path
    h3_stats_path = domain_config.h3_stats_path()
    base_output_path = domain_config.training_dataset_path()
    suffix = f"_{args.tag}" if args.tag else ""
    output_path = base_output_path.with_name(f"{base_output_path.stem}{suffix}{base_output_path.suffix}")

    print(f"=== Build Enhanced Training Dataset (v2) [{domain_config.display_name}] ===")

    drivers = pd.read_parquet(drivers_path)
    train = drivers[drivers["split"] == "train"].reset_index(drop=True)
    del drivers
    print(f"  Train drivers: {len(train):,}")

    if not args.all and args.sample < len(train):
        train = train.sample(n=args.sample, random_state=SAMPLE_SEED).reset_index(drop=True)
        print(f"  Sampled to: {len(train):,}")

    riders = pd.read_parquet(riders_path)
    train_riders = riders[riders["split"] == "train"].reset_index(drop=True)
    del riders
    if args.max_riders is not None and len(train_riders) > args.max_riders:
        train_riders = train_riders.sample(n=args.max_riders, random_state=SAMPLE_SEED).reset_index(drop=True)
        print(f"  Riders subsampled: {len(train_riders):,} (user cap)")
    else:
        print(f"  Riders: {len(train_riders):,}")

    rider_index = RiderIndex(train_riders, index_bin_minutes=args.index_bin_minutes)

    print("  Loading H3 cell stats...")
    h3_stats = pd.read_parquet(h3_stats_path)
    h3_stats_dict = {}
    for _, r in h3_stats.iterrows():
        h3_stats_dict[r["h3_cell"]] = r.to_dict()
    print(f"  H3 stats loaded: {len(h3_stats_dict):,} cells")

    router = OSRMRouter(cache_path=cache_path, cache_only=not args.fetch_routes)
    print(f"  Route cache: {router.cache_size:,} entries")

    pickup_dt = train["pickup_datetime"]
    minutes_of_day = (pickup_dt.dt.hour * 60 + pickup_dt.dt.minute).values
    hours = train["hour_of_day"].values
    dows = train["day_of_week"].values
    weekends = train["is_weekend"].values
    day_of_months = pickup_dt.dt.day.values
    service_window_pos = train["service_window_pos"].values if "service_window_pos" in train.columns else np.ones(len(train), dtype=np.int8)

    origin_lats = train["origin_lat"].values
    origin_lngs = train["origin_lng"].values
    dest_lats = train["dest_lat"].values
    dest_lngs = train["dest_lng"].values

    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path.with_suffix(".partial.parquet")

    rows = []
    skipped = 0
    t_start = time.time()

    pbar = tqdm(range(len(train)), desc="  Building v2", unit="driver", ncols=100)
    for i in pbar:
        origin = (float(origin_lats[i]), float(origin_lngs[i]))
        dest = (float(dest_lats[i]), float(dest_lngs[i]))

        try:
            routes = router.get_alternative_routes(origin, dest, 3)
        except Exception:
            skipped += 1
            continue
        if not routes:
            skipped += 1
            continue

        for ri, route in enumerate(routes):
            corridor = build_corridor(
                route.polyline,
                resolution=args.h3_resolution,
                buffer_rings=args.corridor_k_ring,
                densify_step_m=args.corridor_densify_step_m,
            )
            row = extract_features(
                i, ri, route, corridor, rider_index,
                pickup_dt.iloc[i],
                int(minutes_of_day[i]), int(hours[i]),
                int(dows[i]), int(weekends[i]), int(day_of_months[i]),
                origin, dest, h3_stats_dict,
                candidate_window_bins=args.candidate_window_bins,
                max_request_offset_min=args.max_request_offset_min,
                max_detour_min=args.max_detour_min,
                platform_share=args.platform_share,
                cost_per_mile=args.cost_per_mile,
                urban_speed_kmh=args.urban_speed_kmh,
                seats=args.seats,
            )
            row["service_window_pos"] = int(service_window_pos[i])
            rows.append(row)

        if (i + 1) % 500 == 0:
            pbar.set_postfix({"rows": f"{len(rows):,}", "skip": skipped})

        if (i + 1) % CHECKPOINT_EVERY == 0 and rows:
            pd.DataFrame(rows).to_parquet(checkpoint_path, compression="snappy", index=False)

    pbar.close()
    elapsed = time.time() - t_start

    df = pd.DataFrame(rows)
    df.to_parquet(output_path, compression="snappy", index=False)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    router.flush_cache()

    print(f"\n=== Enhanced Dataset Built ===")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Skipped: {skipped:,}")
    print(f"  Time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"  Output: {output_path}")
    print(f"  Size: {output_path.stat().st_size / (1024**2):.1f} MB")
    if df.empty:
        raise RuntimeError(
            "Enhanced dataset build produced zero rows. "
            "This usually means route fetching was disabled on an empty cache or route generation failed."
        )
    print(f"\n  Feature columns ({len(df.columns) - 4} features + 4 meta):")
    for c in sorted(df.columns):
        if c not in ("driver_id", "route_idx", "expected_revenue", "driver_cost", "expected_profit"):
            print(f"    {c}")


if __name__ == "__main__":
    main()
