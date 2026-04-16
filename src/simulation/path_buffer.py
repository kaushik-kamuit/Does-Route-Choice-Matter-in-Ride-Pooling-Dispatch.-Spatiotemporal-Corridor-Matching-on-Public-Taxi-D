from __future__ import annotations

from datetime import datetime
from typing import Sequence

import numpy as np
import pandas as pd
import shapely
from shapely import LineString

from matching.matcher import COS_LAT, DEG_TO_M
from matching.rider_index import RiderIndex, circular_minute_diff
from spatial.h3_utils import LatLng

DEFAULT_PATH_BUFFER_M = 1000.0


def _make_route_line(polyline: Sequence[LatLng]) -> LineString:
    return LineString([(lng * COS_LAT, lat) for lat, lng in polyline])


def _time_filter_candidates(
    rider_index: RiderIndex,
    minute_of_day: int,
    *,
    max_request_offset_min: int | None,
    query_datetime: datetime | pd.Timestamp | None,
    available_rider_ids: set[int] | None,
) -> pd.DataFrame:
    riders = rider_index.riders
    if riders.empty:
        return riders.iloc[0:0]

    keep = np.ones(len(riders), dtype=bool)
    if available_rider_ids is not None:
        keep &= riders.index.isin(available_rider_ids)

    if query_datetime is not None:
        query_ts = pd.Timestamp(query_datetime)
        keep &= riders["pickup_datetime"].dt.normalize().eq(query_ts.normalize()).to_numpy()
        if max_request_offset_min is not None:
            delta_s = (riders["pickup_datetime"] - query_ts).dt.total_seconds().abs().to_numpy()
            keep &= delta_s <= int(max_request_offset_min) * 60
    elif max_request_offset_min is not None:
        delta_min = circular_minute_diff(riders["pickup_minute_of_day"].to_numpy(), minute_of_day)
        keep &= delta_min <= int(max_request_offset_min)

    if not np.any(keep):
        return riders.iloc[0:0]
    return riders.loc[keep]


def path_buffer_candidates(
    rider_index: RiderIndex,
    polyline: Sequence[LatLng],
    minute_of_day: int,
    *,
    max_request_offset_min: int | None,
    query_datetime: datetime | pd.Timestamp | None,
    available_rider_ids: set[int] | None = None,
    buffer_m: float = DEFAULT_PATH_BUFFER_M,
) -> pd.DataFrame:
    """Simple geometric retrieval baseline using a fixed path buffer."""
    candidates = _time_filter_candidates(
        rider_index,
        minute_of_day,
        max_request_offset_min=max_request_offset_min,
        query_datetime=query_datetime,
        available_rider_ids=available_rider_ids,
    )
    if candidates.empty:
        return candidates

    route_line = _make_route_line(polyline)
    if route_line.length < 1e-9:
        return candidates.iloc[0:0]

    pu_pts = shapely.points(
        candidates["pickup_lng"].to_numpy() * COS_LAT,
        candidates["pickup_lat"].to_numpy(),
    )
    do_pts = shapely.points(
        candidates["dropoff_lng"].to_numpy() * COS_LAT,
        candidates["dropoff_lat"].to_numpy(),
    )
    pu_dist_m = shapely.distance(route_line, pu_pts) * DEG_TO_M
    do_dist_m = shapely.distance(route_line, do_pts) * DEG_TO_M
    keep = (pu_dist_m <= buffer_m) & (do_dist_m <= buffer_m)
    if not np.any(keep):
        return candidates.iloc[0:0]
    return candidates.iloc[np.flatnonzero(keep)]


def path_buffer_candidate_count(
    rider_index: RiderIndex,
    polyline: Sequence[LatLng],
    minute_of_day: int,
    *,
    max_request_offset_min: int | None,
    query_datetime: datetime | pd.Timestamp | None,
    available_rider_ids: set[int] | None = None,
    buffer_m: float = DEFAULT_PATH_BUFFER_M,
) -> int:
    return int(
        path_buffer_candidates(
            rider_index,
            polyline,
            minute_of_day,
            max_request_offset_min=max_request_offset_min,
            query_datetime=query_datetime,
            available_rider_ids=available_rider_ids,
            buffer_m=buffer_m,
        ).shape[0]
    )
