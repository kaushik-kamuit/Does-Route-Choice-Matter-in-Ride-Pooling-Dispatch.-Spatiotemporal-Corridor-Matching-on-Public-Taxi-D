"""
One-time migration: route_cache.json -> route_cache.db (SQLite).

Streams the JSON file with ijson so peak memory stays at O(1 entry),
not O(all entries). Safe for 2 GB+ files on machines with limited RAM.

Handles truncated JSON gracefully (from a previous crash) by recovering
every entry before the corruption point.

Usage:
    pip install ijson
    python scripts/migrate_cache_to_sqlite.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from decimal import Decimal
from pathlib import Path

try:
    import ijson
except ImportError:
    print("ERROR: ijson is required.  pip install ijson")
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "data" / "route_cache.json"
DB_PATH = ROOT / "data" / "route_cache.db"

BATCH_SIZE = 1_000
LOG_EVERY = 5_000


def main() -> None:
    if not JSON_PATH.exists():
        print(f"ERROR: {JSON_PATH} not found.")
        sys.exit(1)

    if DB_PATH.exists():
        conn_check = sqlite3.connect(str(DB_PATH))
        existing = conn_check.execute("SELECT COUNT(*) FROM routes").fetchone()[0]
        conn_check.close()
        if existing > 0:
            print(f"WARNING: {DB_PATH} already exists with {existing:,} entries.")
            resp = input("  Overwrite? (y/N): ").strip().lower()
            if resp != "y":
                print("Aborted.")
                sys.exit(0)
            DB_PATH.unlink()

    size_mb = JSON_PATH.stat().st_size / (1024 ** 2)
    print(f"=== Migrate Route Cache: JSON -> SQLite ===")
    print(f"  Source: {JSON_PATH}  ({size_mb:.0f} MB)")
    print(f"  Target: {DB_PATH}")
    print()

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS routes "
        "(cache_key TEXT PRIMARY KEY, routes_json TEXT NOT NULL)"
    )
    conn.commit()

    count = 0
    t0 = time.time()

    with open(JSON_PATH, "rb") as f:
        try:
            conn.execute("BEGIN")
            for key, value in ijson.kvitems(f, ""):
                compact = json.dumps(
                    value, separators=(",", ":"),
                    default=lambda o: float(o) if isinstance(o, Decimal) else str(o),
                )
                conn.execute(
                    "INSERT OR IGNORE INTO routes VALUES (?, ?)",
                    (key, compact),
                )
                count += 1

                if count % BATCH_SIZE == 0:
                    conn.commit()
                    conn.execute("BEGIN")

                if count % LOG_EVERY == 0:
                    elapsed = time.time() - t0
                    rate = count / elapsed if elapsed > 0 else 0
                    print(f"  {count:,} entries migrated  ({rate:.0f} entries/s)")

            conn.commit()

        except Exception as e:
            conn.commit()
            print(f"\n  JSON parse stopped: {type(e).__name__}: {e}")
            print(f"  Recovered {count:,} entries before the error.")

    elapsed = time.time() - t0

    final_count = conn.execute("SELECT COUNT(*) FROM routes").fetchone()[0]
    db_mb = DB_PATH.stat().st_size / (1024 ** 2)
    conn.close()

    print()
    print(f"=== Migration Complete ===")
    print(f"  Entries migrated: {final_count:,}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  SQLite file size: {db_mb:.1f} MB")
    print(f"  JSON file size:   {size_mb:.1f} MB")
    print()
    print(f"  Verify with:")
    print(f'    python -c "import sqlite3; c=sqlite3.connect(\'{DB_PATH}\'); '
          f"print(c.execute('SELECT COUNT(*) FROM routes').fetchone())\"")
    print()
    print(f"  After verification, you can delete the JSON file:")
    print(f"    del \"{JSON_PATH}\"")


if __name__ == "__main__":
    main()
