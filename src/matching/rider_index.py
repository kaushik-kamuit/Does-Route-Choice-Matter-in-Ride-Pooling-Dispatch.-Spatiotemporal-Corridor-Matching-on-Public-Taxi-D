"""
Spatial-temporal index over rider trips for fast corridor lookups.

Instead of scanning all ~10M riders per corridor query, this index
maps H3 cells to rider row indices. A corridor query does O(corridor_cells)
    dict lookups and returns only the riders whose pickup falls within the
    corridor, optionally requiring the dropoff to do the same, and whose
    time is compatible.

The index uses coarse temporal bins for speed, but callers can also apply
an exact post-retrieval request-offset filter in minutes. This keeps the
lookup fast without conflating bin width with the true matching window.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

import numpy as np
import pandas as pd

MINUTES_PER_DAY = 24 * 60


@dataclass(frozen=True)
class CorridorLookupStats:
    pickup_bucket_hits: int
    dropoff_bucket_hits: int
    corridor_joint_candidates: int
    exact_time_eligible: int


def circular_minute_diff(minute_a: int | np.ndarray, minute_b: int) -> np.ndarray:
    """Return the shortest absolute minute distance on a 24-hour clock."""
    diff = (np.asarray(minute_a, dtype=np.int32) - int(minute_b)) % MINUTES_PER_DAY
    return np.minimum(diff, MINUTES_PER_DAY - diff)


class RiderIndex:
    """
    In-memory index: (H3 cell, date-aware 15-min bucket) -> rider row indices.

    Build once from riders.parquet, then query repeatedly per corridor.
    """

    def __init__(self, riders: pd.DataFrame, index_bin_minutes: int = 15):
        self._riders = riders
        self._n = len(riders)
        self._index_bin_minutes = index_bin_minutes
        self._bins_per_day = MINUTES_PER_DAY // index_bin_minutes
        self._bucket_freq = f"{index_bin_minutes}min"
        self._service_dates = pd.Index([], dtype="datetime64[ns]")

        self._pickup_idx: dict[tuple[str, pd.Timestamp], np.ndarray] = {}
        self._dropoff_idx: dict[tuple[str, pd.Timestamp], np.ndarray] = {}

        self._build()

    def _build(self) -> None:
        t0 = time.time()
        print(f"  Building RiderIndex for {self._n:,} riders...")

        if "pickup_minute_of_day" not in self._riders.columns:
            dt = self._riders["pickup_datetime"]
            self._riders = self._riders.copy()
            self._riders["pickup_minute_of_day"] = (
                dt.dt.hour * 60 + dt.dt.minute
            ).astype(np.int16)
            self._service_dates = dt.dt.normalize().drop_duplicates().sort_values()
        else:
            self._service_dates = (
                self._riders["pickup_datetime"].dt.normalize().drop_duplicates().sort_values()
            )

        bin_col = self._bucket_column
        if bin_col not in self._riders.columns:
            self._riders = self._riders.copy()
            self._riders[bin_col] = (
                self._riders["pickup_datetime"].dt.floor(self._bucket_freq)
            )

        for (cell, bucket), group in self._riders.groupby(["pickup_h3", bin_col]).groups.items():
            self._pickup_idx[(cell, pd.Timestamp(bucket))] = group.to_numpy(dtype=np.int32)

        for (cell, bucket), group in self._riders.groupby(["dropoff_h3", bin_col]).groups.items():
            self._dropoff_idx[(cell, pd.Timestamp(bucket))] = group.to_numpy(dtype=np.int32)

        n_pickup_cells = len({k[0] for k in self._pickup_idx})
        n_dropoff_cells = len({k[0] for k in self._dropoff_idx})

        elapsed = time.time() - t0
        print(f"    Pickup cells indexed: {n_pickup_cells:,} ({len(self._pickup_idx):,} cell-time buckets)")
        print(f"    Dropoff cells indexed: {n_dropoff_cells:,} ({len(self._dropoff_idx):,} cell-time buckets)")
        print(
            f"    Temporal bins: {self._index_bin_minutes}-min "
            f"({self._bins_per_day} per day)"
        )
        print(f"    Build time: {elapsed:.1f}s")

    def _gather_indices_np(
        self,
        cells: frozenset[str] | set[str],
        index: dict[tuple[str, pd.Timestamp], np.ndarray],
        buckets: list[pd.Timestamp],
    ) -> np.ndarray:
        """Collect unique rider indices from matching (cell, time-bucket) keys."""
        arrays: list[np.ndarray] = []
        for cell in cells:
            for bucket in buckets:
                arr = index.get((cell, bucket))
                if arr is not None:
                    arrays.append(arr)
        if not arrays:
            return np.empty(0, dtype=np.int32)
        return np.unique(np.concatenate(arrays))

    def find_in_corridor(
        self,
        corridor_cells: frozenset[str] | set[str],
        minute_of_day: int,
        window_bins: int = 1,
        max_request_offset_min: int | None = None,
        query_datetime: datetime | pd.Timestamp | None = None,
        require_dropoff_in_corridor: bool = True,
    ) -> pd.DataFrame:
        subset, _ = self.find_in_corridor_with_stats(
            corridor_cells,
            minute_of_day,
            window_bins=window_bins,
            max_request_offset_min=max_request_offset_min,
            query_datetime=query_datetime,
            require_dropoff_in_corridor=require_dropoff_in_corridor,
        )
        return subset

    def find_in_corridor_with_stats(
        self,
        corridor_cells: frozenset[str] | set[str],
        minute_of_day: int,
        window_bins: int = 1,
        max_request_offset_min: int | None = None,
        query_datetime: datetime | pd.Timestamp | None = None,
        require_dropoff_in_corridor: bool = True,
    ) -> tuple[pd.DataFrame, CorridorLookupStats]:
        """
        Find riders whose pickup is inside the corridor and whose index bin is
        within [bin - window, bin + window]. Optionally require the dropoff to
        be inside the corridor as well.

        Args:
            minute_of_day: Driver's departure minute (0-1439). Converted
                           internally to an index bin. Deprecated when
                           query_datetime is available.
            window_bins:   Number of index bins to extend in each direction.
            max_request_offset_min:
                           Exact minute filter applied after retrieval.
                           If None, bin retrieval alone defines compatibility.
            query_datetime:
                           Driver departure timestamp. When provided, the
                           lookup is date-aware and the exact request filter
                           uses true timedeltas instead of only clock time.

        Returns a DataFrame (subset of the original riders) plus lookup-stage stats.
        """
        if query_datetime is not None:
            query_ts = pd.Timestamp(query_datetime)
            center_bucket = query_ts.floor(self._bucket_freq)
            buckets = [
                center_bucket + pd.Timedelta(minutes=self._index_bin_minutes * d)
                for d in range(-window_bins, window_bins + 1)
            ]
        else:
            center_minute = (
                minute_of_day // self._index_bin_minutes
            ) * self._index_bin_minutes
            buckets = []
            for service_date in self._service_dates:
                base_bucket = pd.Timestamp(service_date) + pd.Timedelta(minutes=int(center_minute))
                for delta_bins in range(-window_bins, window_bins + 1):
                    buckets.append(
                        base_bucket + pd.Timedelta(minutes=self._index_bin_minutes * delta_bins)
                    )
            query_ts = None

        pickup_arr = self._gather_indices_np(corridor_cells, self._pickup_idx, buckets)
        if pickup_arr.size == 0:
            return self._riders.iloc[0:0], CorridorLookupStats(0, 0, 0, 0)

        if require_dropoff_in_corridor:
            dropoff_arr = self._gather_indices_np(corridor_cells, self._dropoff_idx, buckets)
            if dropoff_arr.size == 0:
                return self._riders.iloc[0:0], CorridorLookupStats(int(pickup_arr.size), 0, 0, 0)

            candidate_arr = np.intersect1d(pickup_arr, dropoff_arr)
            if candidate_arr.size == 0:
                return self._riders.iloc[0:0], CorridorLookupStats(int(pickup_arr.size), int(dropoff_arr.size), 0, 0)

            subset = self._riders.iloc[candidate_arr]
            base_stats = CorridorLookupStats(
                pickup_bucket_hits=int(pickup_arr.size),
                dropoff_bucket_hits=int(dropoff_arr.size),
                corridor_joint_candidates=int(candidate_arr.size),
                exact_time_eligible=int(candidate_arr.size),
            )
        else:
            subset = self._riders.iloc[pickup_arr]
            base_stats = CorridorLookupStats(
                pickup_bucket_hits=int(pickup_arr.size),
                dropoff_bucket_hits=0,
                corridor_joint_candidates=int(pickup_arr.size),
                exact_time_eligible=int(pickup_arr.size),
            )
        if max_request_offset_min is None:
            return subset, base_stats

        if query_datetime is None:
            delta = circular_minute_diff(
                subset["pickup_minute_of_day"].to_numpy(),
                minute_of_day,
            )
            keep = delta <= int(max_request_offset_min)
        elif query_ts is not None:
            delta_s = (
                subset["pickup_datetime"] - query_ts
            ).dt.total_seconds().abs()
            keep = delta_s <= int(max_request_offset_min) * 60
        else:
            delta = circular_minute_diff(
                subset["pickup_minute_of_day"].to_numpy(),
                minute_of_day,
            )
            keep = delta <= int(max_request_offset_min)
        if not np.any(keep):
            return self._riders.iloc[0:0], CorridorLookupStats(
                pickup_bucket_hits=base_stats.pickup_bucket_hits,
                dropoff_bucket_hits=base_stats.dropoff_bucket_hits,
                corridor_joint_candidates=base_stats.corridor_joint_candidates,
                exact_time_eligible=0,
            )
        filtered = subset.iloc[np.flatnonzero(keep)]
        return filtered, CorridorLookupStats(
            pickup_bucket_hits=base_stats.pickup_bucket_hits,
            dropoff_bucket_hits=base_stats.dropoff_bucket_hits,
            corridor_joint_candidates=base_stats.corridor_joint_candidates,
            exact_time_eligible=int(filtered.shape[0]),
        )

    @property
    def _bucket_column(self) -> str:
        return f"pickup_bucket_{self._index_bin_minutes}m"

    @property
    def n_riders(self) -> int:
        return self._n

    @property
    def n_pickup_cells(self) -> int:
        return len(self._pickup_idx)

    @property
    def n_dropoff_cells(self) -> int:
        return len(self._dropoff_idx)
