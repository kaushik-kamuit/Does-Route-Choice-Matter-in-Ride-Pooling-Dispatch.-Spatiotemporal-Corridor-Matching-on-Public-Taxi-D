from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


@dataclass(frozen=True)
class EvalSplit:
    split_name: str
    train_idx: np.ndarray
    val_idx: np.ndarray
    train_label: str
    val_label: str


def build_eval_split(
    df: pd.DataFrame,
    *,
    prefer_temporal: bool = True,
    group_test_size: float = 0.20,
    split_seed: int = 42,
) -> EvalSplit:
    """Create a validation split for model evaluation.

    Preference order:
      1. Relative temporal split via service_window_pos (1-2 train, 3 validation).
      2. Temporal month split (Jan-Feb train, Mar validation) when service_month exists.
      3. GroupShuffleSplit by driver_id as a fallback.
    """
    if prefer_temporal and "service_window_pos" in df.columns:
        positions = set(df["service_window_pos"].dropna().astype(int).unique().tolist())
        if {1, 2, 3}.issubset(positions):
            train_mask = df["service_window_pos"].isin([1, 2]).to_numpy()
            val_mask = df["service_window_pos"].eq(3).to_numpy()
            train_idx = np.flatnonzero(train_mask)
            val_idx = np.flatnonzero(val_mask)
            if train_idx.size > 0 and val_idx.size > 0:
                return EvalSplit(
                    split_name="temporal_windowpos_1_2_to_3",
                    train_idx=train_idx,
                    val_idx=val_idx,
                    train_label="Window months 1-2",
                    val_label="Window month 3",
                )

    if prefer_temporal and "service_month" in df.columns:
        months = set(df["service_month"].dropna().astype(int).unique().tolist())
        if {1, 2, 3}.issubset(months):
            train_mask = df["service_month"].isin([1, 2]).to_numpy()
            val_mask = df["service_month"].eq(3).to_numpy()
            train_idx = np.flatnonzero(train_mask)
            val_idx = np.flatnonzero(val_mask)
            if train_idx.size > 0 and val_idx.size > 0:
                return EvalSplit(
                    split_name="temporal_jan_feb_to_mar",
                    train_idx=train_idx,
                    val_idx=val_idx,
                    train_label="Jan-Feb 2015",
                    val_label="Mar 2015",
                )

    groups = df["driver_id"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=group_test_size, random_state=split_seed)
    dummy = np.zeros(len(df), dtype=np.float32)
    train_idx, val_idx = next(gss.split(dummy, dummy, groups=groups))
    return EvalSplit(
        split_name="group_shuffle_by_driver",
        train_idx=train_idx,
        val_idx=val_idx,
        train_label="Grouped random train",
        val_label="Grouped random val",
    )
