"""
Batch-route drivers through the OSRM MLD server.

Populates the route cache so that build_dataset.py and the simulation
runner can load polylines instantly from disk.

Usage:
    python src/models/batch_route.py                 # route 100K sampled train drivers
    python src/models/batch_route.py --all            # route ALL train + test drivers
    python src/models/batch_route.py --sample 50000  # custom sample size

Restart-safe: already-cached O-D pairs are skipped automatically.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import pandas as pd
from tqdm import tqdm
from spatial.router import OSRMRouter

DRIVERS_PATH = ROOT / "data" / "processed" / "drivers.parquet"
CACHE_PATH = ROOT / "data" / "route_cache.db"
DEFAULT_SAMPLE = 100_000
SAMPLE_SEED = 42
FLUSH_EVERY = 5000
MAX_ALTERNATIVES = 3


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-route drivers via OSRM")
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE,
                        help=f"Number of drivers to sample (default {DEFAULT_SAMPLE:,})")
    parser.add_argument("--all", action="store_true",
                        help="Route ALL train + test drivers (overrides --sample)")
    parser.add_argument("--test", action="store_true",
                        help="Route test drivers instead of train (for simulation)")
    args = parser.parse_args()

    print("=== Batch Route: Drivers via MLD ===")
    print(f"  Cache path: {CACHE_PATH}")
    print(f"  Alternatives: {MAX_ALTERNATIVES}")
    print()

    drivers = pd.read_parquet(
        DRIVERS_PATH,
        columns=["origin_lat", "origin_lng", "dest_lat", "dest_lng", "split"],
    )

    if args.all:
        subset = drivers.reset_index(drop=True)
        print(f"  Routing ALL {len(subset):,} drivers (train + test)")
    elif args.test:
        test = drivers.loc[drivers["split"] == "test"].reset_index(drop=True)
        if args.sample < len(test):
            subset = test.sample(n=args.sample, random_state=SAMPLE_SEED).reset_index(drop=True)
            print(f"  Sampled {len(subset):,} test drivers (seed={SAMPLE_SEED})")
        else:
            subset = test
            print(f"  Routing all {len(subset):,} test drivers")
    else:
        train = drivers.loc[drivers["split"] == "train"].reset_index(drop=True)
        if args.sample < len(train):
            subset = train.sample(n=args.sample, random_state=SAMPLE_SEED).reset_index(drop=True)
            print(f"  Sampled {len(subset):,} train drivers (seed={SAMPLE_SEED})")
        else:
            subset = train
            print(f"  Routing all {len(subset):,} train drivers")
    del drivers

    total = len(subset)
    print(f"  Drivers to route: {total:,}")

    router = OSRMRouter(cache_path=CACHE_PATH)
    initial_cache = router.cache_size
    print(f"  Cache entries at start: {initial_cache:,}")
    print()

    origin_lats = subset["origin_lat"].values
    origin_lngs = subset["origin_lng"].values
    dest_lats = subset["dest_lat"].values
    dest_lngs = subset["dest_lng"].values

    t_start = time.time()
    errors = 0

    pbar = tqdm(range(total), desc="  Routing", unit="driver", ncols=100)
    for i in pbar:
        origin = (float(origin_lats[i]), float(origin_lngs[i]))
        dest = (float(dest_lats[i]), float(dest_lngs[i]))

        try:
            router.get_alternative_routes(origin, dest, MAX_ALTERNATIVES)
        except Exception as e:
            errors += 1
            if errors <= 10:
                tqdm.write(f"  ERROR at row {i}: {e}")
            continue

        if (i + 1) % FLUSH_EVERY == 0:
            router.flush_cache()

        if (i + 1) % 10000 == 0:
            new_entries = router.cache_size - initial_cache
            pbar.set_postfix({
                'cache': f'{router.cache_size:,}',
                'new': f'{new_entries:,}',
                'api': f'{router.api_calls:,}',
                'err': errors,
            })

    pbar.close()
    router.flush_cache()
    elapsed = time.time() - t_start

    print()
    print("=== Batch Routing Complete ===")
    print(f"  Drivers processed: {total:,}")
    print(f"  Final cache size: {router.cache_size:,}")
    print(f"  New entries added: {router.cache_size - initial_cache:,}")
    print(f"  Total API calls: {router.api_calls:,}")
    print(f"  Errors: {errors}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Cache file: {CACHE_PATH}")

    cache_mb = CACHE_PATH.stat().st_size / (1024 ** 2)
    print(f"  Cache file size: {cache_mb:.1f} MB")


if __name__ == "__main__":
    main()
