"""
Spatial-temporal index over rider trips for fast corridor lookups.

Instead of scanning all ~10M riders per corridor query, this index
maps H3 cells to rider row indices. A corridor query does O(corridor_cells)
dict lookups and returns only the riders whose pickup AND dropoff fall
within the corridor and whose time is compatible.

Temporal resolution: 15-minute bins (0-95 per day).  A query with
+/-1 bin gives a 45-minute window, much tighter than the old +/-1 hour
(3-hour window) while still handling bin-boundary crossings.
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import pandas as pd

BINS_PER_DAY = 96  # 24 * 4


class RiderIndex:
    """
    In-memory index: (H3 cell, 15-min bin) -> rider row indices.

    Build once from riders.parquet, then query repeatedly per corridor.
    """

    def __init__(self, riders: pd.DataFrame):
        self._riders = riders
        self._n = len(riders)

        self._pickup_idx: dict[tuple[str, int], np.ndarray] = {}
        self._dropoff_idx: dict[tuple[str, int], np.ndarray] = {}

        self._build()

    def _build(self) -> None:
        t0 = time.time()
        print(f"  Building RiderIndex for {self._n:,} riders...")

        if "pickup_qh" not in self._riders.columns:
            dt = self._riders["pickup_datetime"]
            self._riders = self._riders.copy()
            self._riders["pickup_qh"] = (dt.dt.hour * 4 + dt.dt.minute // 15).astype(np.int8)

        for (cell, qh), group in self._riders.groupby(["pickup_h3", "pickup_qh"]).groups.items():
            self._pickup_idx[(cell, int(qh))] = group.to_numpy(dtype=np.int32)

        for (cell, qh), group in self._riders.groupby(["dropoff_h3", "pickup_qh"]).groups.items():
            self._dropoff_idx[(cell, int(qh))] = group.to_numpy(dtype=np.int32)

        n_pickup_cells = len({k[0] for k in self._pickup_idx})
        n_dropoff_cells = len({k[0] for k in self._dropoff_idx})

        elapsed = time.time() - t0
        print(f"    Pickup cells indexed: {n_pickup_cells:,} ({len(self._pickup_idx):,} cell-qh buckets)")
        print(f"    Dropoff cells indexed: {n_dropoff_cells:,} ({len(self._dropoff_idx):,} cell-qh buckets)")
        print(f"    Temporal bins: 15-min ({BINS_PER_DAY} per day)")
        print(f"    Build time: {elapsed:.1f}s")

    def _gather_indices_np(
        self,
        cells: frozenset[str] | set[str],
        index: dict[tuple[str, int], np.ndarray],
        qh_bins: list[int],
    ) -> np.ndarray:
        """Collect unique rider indices from matching (cell, qh_bin) buckets."""
        arrays: list[np.ndarray] = []
        for cell in cells:
            for qh in qh_bins:
                arr = index.get((cell, qh))
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
    ) -> pd.DataFrame:
        """
        Find riders whose pickup AND dropoff are both inside the corridor
        and whose 15-min time bin is within [bin - window, bin + window].

        Args:
            minute_of_day: Driver's departure minute (0-1439). Converted
                           internally to a 15-min bin (0-95).
            window_bins:   Number of 15-min bins to extend in each direction.
                           Default 1 → 3 bins = 45-minute window.

        Returns a DataFrame (subset of the original riders).
        """
        center_bin = minute_of_day // 15
        qh_bins = [(center_bin + d) % BINS_PER_DAY
                    for d in range(-window_bins, window_bins + 1)]

        pickup_arr = self._gather_indices_np(corridor_cells, self._pickup_idx, qh_bins)
        if pickup_arr.size == 0:
            return self._riders.iloc[0:0]

        dropoff_arr = self._gather_indices_np(corridor_cells, self._dropoff_idx, qh_bins)
        if dropoff_arr.size == 0:
            return self._riders.iloc[0:0]

        both = np.intersect1d(pickup_arr, dropoff_arr)
        if both.size == 0:
            return self._riders.iloc[0:0]

        return self._riders.iloc[both]

    @property
    def n_riders(self) -> int:
        return self._n

    @property
    def n_pickup_cells(self) -> int:
        return len(self._pickup_idx)

    @property
    def n_dropoff_cells(self) -> int:
        return len(self._dropoff_idx)
