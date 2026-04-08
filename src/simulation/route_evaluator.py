from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from matching.matcher import METERS_PER_MILE, match_riders
from matching.rider_index import RiderIndex
from models.predict import ProfitPredictor
from simulation.baselines import _proxy_profit_score
from simulation.data_types import DriverOutcome, DriverTrip
from simulation.path_buffer import path_buffer_candidate_count
from simulation.warmup import _route_features
from spatial.corridor import Corridor, build_corridor
from spatial.router import RouteInfo


def _deterministic_max(route_evals: list["RouteEvaluation"], metric_getter) -> int:
    """Pick the highest-metric route without using realized profit as a tie-breaker."""
    best = max(route_evals, key=lambda row: (metric_getter(row), -row.route_idx))
    return best.route_idx


@dataclass(frozen=True)
class RouteEvaluation:
    route_idx: int
    route: RouteInfo
    corridor: Corridor
    candidates: pd.DataFrame
    matched: list[dict]
    feasible: list[dict]
    actual_profit: float
    total_revenue: float
    driving_cost: float
    retrieved_candidate_count: int
    exact_time_candidate_count: int
    candidate_count: int
    feasible_count: int
    path_buffer_count: int
    fare_density: float
    proxy_profit: float
    predicted_profit: float


@dataclass(frozen=True)
class StrategyPlan:
    strategy: str
    route_idx: int
    score: float
    outcome: DriverOutcome
    matched: list[dict]
    retrieved_candidates_considered: int
    exact_time_candidates_considered: int
    candidates_considered: int
    feasible_count: int


@dataclass(frozen=True)
class DriverPolicyEvaluation:
    driver_id: int
    route_evaluations: tuple[RouteEvaluation, ...]
    plans: dict[str, StrategyPlan]


def _filter_available_candidates(
    candidates: pd.DataFrame,
    available_rider_ids: set[int] | None,
) -> pd.DataFrame:
    if available_rider_ids is None or candidates.empty:
        return candidates
    keep = candidates.index.isin(available_rider_ids)
    if not np.any(keep):
        return candidates.iloc[0:0]
    return candidates.loc[keep]


def _build_outcome(
    driver: DriverTrip,
    strategy: str,
    route_eval: RouteEvaluation,
    *,
    route_rank_chosen: int,
    predicted_profit: float,
    compute_time_s: float,
    seed: int,
) -> DriverOutcome:
    distance_miles = route_eval.route.distance_m / METERS_PER_MILE
    return DriverOutcome(
        driver_id=driver.driver_id,
        strategy=strategy,
        route_distance_miles=distance_miles,
        matched_riders=len(route_eval.matched),
        total_revenue=route_eval.total_revenue,
        driving_cost=route_eval.driving_cost,
        profit=route_eval.actual_profit,
        route_rank_chosen=route_rank_chosen,
        predicted_profit=predicted_profit,
        compute_time_s=compute_time_s,
        route_length_category=driver.route_length_category,
        seed=seed,
        hour=driver.hour,
    )


def evaluate_driver_policies(
    driver: DriverTrip,
    rider_index: RiderIndex,
    predictor: ProfitPredictor,
    h3_stats_dict: dict[str, dict],
    *,
    day_of_week: int,
    is_weekend: int,
    day_of_month: int,
    seed: int,
    candidate_window_bins: int,
    max_request_offset_min: int | None,
    h3_resolution: int,
    corridor_k_ring: int,
    corridor_densify_step_m: float,
    available_rider_ids: set[int] | None = None,
    routes: list[RouteInfo],
) -> DriverPolicyEvaluation:
    corridors = [
        build_corridor(
            route.polyline,
            resolution=h3_resolution,
            buffer_rings=corridor_k_ring,
            densify_step_m=corridor_densify_step_m,
        )
        for route in routes
    ]

    feature_rows: list[dict[str, float]] = []
    route_evals: list[RouteEvaluation] = []
    for idx, (route, corridor) in enumerate(zip(routes, corridors)):
        candidates, lookup_stats = rider_index.find_in_corridor_with_stats(
            corridor.corridor_cells,
            driver.minute_of_day,
            window_bins=candidate_window_bins,
            max_request_offset_min=max_request_offset_min,
            query_datetime=driver.departure_time,
        )
        candidates = _filter_available_candidates(candidates, available_rider_ids)
        matched, feasible = match_riders(
            corridor,
            route.polyline,
            rider_index,
            minute_of_day=driver.minute_of_day,
            query_datetime=driver.departure_time,
            seats=driver.seats,
            max_detour_min=driver.max_detour_minutes,
            candidate_window_bins=candidate_window_bins,
            max_request_offset_min=max_request_offset_min,
            platform_share=driver.platform_share,
            urban_speed_kmh=driver.urban_speed_kmh,
            seed=seed,
            candidates=candidates,
        )
        total_revenue = sum(match["fare_share"] for match in matched)
        driving_cost = (route.distance_m / METERS_PER_MILE) * driver.cost_per_mile
        actual_profit = total_revenue - driving_cost
        fare_density = (
            float(candidates["fare_amount"].sum()) / max(corridor.n_corridor_cells, 1)
            if not candidates.empty
            else 0.0
        )
        proxy_profit = _proxy_profit_score(driver, route, candidates, corridor)
        feature_rows.append(
            _route_features(
                route,
                corridor,
                rider_index,
                route.polyline,
                driver.minute_of_day,
                driver.hour,
                day_of_week,
                is_weekend,
                day_of_month,
                driver.origin,
                driver.destination,
                h3_stats_dict,
                seats=driver.seats,
                max_detour_min=driver.max_detour_minutes,
                candidate_window_bins=candidate_window_bins,
                max_request_offset_min=max_request_offset_min,
                query_datetime=driver.departure_time,
                candidates=candidates,
            )
        )
        route_evals.append(
            RouteEvaluation(
                route_idx=idx,
                route=route,
                corridor=corridor,
                candidates=candidates,
                matched=matched,
                feasible=feasible,
                actual_profit=actual_profit,
                total_revenue=total_revenue,
                driving_cost=driving_cost,
                candidate_count=len(candidates),
                retrieved_candidate_count=lookup_stats.corridor_joint_candidates,
                exact_time_candidate_count=lookup_stats.exact_time_eligible,
                feasible_count=len(feasible),
                path_buffer_count=path_buffer_candidate_count(
                    rider_index,
                    route.polyline,
                    driver.minute_of_day,
                    max_request_offset_min=max_request_offset_min,
                    query_datetime=driver.departure_time,
                    available_rider_ids=available_rider_ids,
                ),
                fare_density=fare_density,
                proxy_profit=proxy_profit,
                predicted_profit=0.0,
            )
        )

    ranking = predictor.rank_routes(feature_rows) if route_evals else []
    predicted_map = {idx: pred for idx, pred in ranking}
    route_evals = [
        RouteEvaluation(
            **{**route_eval.__dict__, "predicted_profit": float(predicted_map.get(route_eval.route_idx, 0.0))}
        )
        for route_eval in route_evals
    ]
    route_eval_map = {route_eval.route_idx: route_eval for route_eval in route_evals}

    def _plan(
        strategy: str,
        route_idx: int,
        score: float,
        *,
        predicted_profit: float = 0.0,
    ) -> StrategyPlan:
        route_eval = route_eval_map[route_idx]
        outcome = _build_outcome(
            driver,
            strategy,
            route_eval,
            route_rank_chosen=route_idx + 1,
            predicted_profit=predicted_profit,
            compute_time_s=0.0,
            seed=seed,
        )
        return StrategyPlan(
            strategy=strategy,
            route_idx=route_idx,
            score=score,
            outcome=outcome,
            matched=route_eval.matched,
            retrieved_candidates_considered=route_eval.retrieved_candidate_count,
            exact_time_candidates_considered=route_eval.exact_time_candidate_count,
            candidates_considered=route_eval.candidate_count,
            feasible_count=route_eval.feasible_count,
        )

    if not route_evals:
        return DriverPolicyEvaluation(driver_id=driver.driver_id, route_evaluations=tuple(), plans={})

    rng = np.random.default_rng(seed + driver.driver_id)
    random_idx = int(rng.integers(0, len(route_evals)))
    oracle_idx = max(route_evals, key=lambda row: row.actual_profit).route_idx
    heuristic_count_idx = _deterministic_max(route_evals, lambda row: row.candidate_count)
    heuristic_path_buffer_idx = _deterministic_max(route_evals, lambda row: row.path_buffer_count)
    heuristic_fare_idx = _deterministic_max(route_evals, lambda row: row.fare_density)
    heuristic_feasible_idx = _deterministic_max(route_evals, lambda row: row.feasible_count)
    heuristic_proxy_idx = _deterministic_max(route_evals, lambda row: row.proxy_profit)
    warmup_idx, warmup_pred = ranking[0] if ranking else (0, route_evals[0].predicted_profit)

    plans = {
        "coldstart": _plan("coldstart", 0, route_eval_map[0].actual_profit),
        "random": _plan("random", random_idx, route_eval_map[random_idx].actual_profit),
        "heuristic_count": _plan(
            "heuristic_count",
            heuristic_count_idx,
            float(route_eval_map[heuristic_count_idx].candidate_count),
        ),
        "heuristic_path_buffer": _plan(
            "heuristic_path_buffer",
            heuristic_path_buffer_idx,
            float(route_eval_map[heuristic_path_buffer_idx].path_buffer_count),
        ),
        "heuristic_fare_density": _plan(
            "heuristic_fare_density",
            heuristic_fare_idx,
            route_eval_map[heuristic_fare_idx].fare_density,
        ),
        "heuristic_feasible_count": _plan(
            "heuristic_feasible_count",
            heuristic_feasible_idx,
            float(route_eval_map[heuristic_feasible_idx].feasible_count),
        ),
        "heuristic_profit_proxy": _plan(
            "heuristic_profit_proxy",
            heuristic_proxy_idx,
            route_eval_map[heuristic_proxy_idx].proxy_profit,
        ),
        "warmup": _plan(
            "warmup",
            warmup_idx,
            float(warmup_pred),
            predicted_profit=float(warmup_pred),
        ),
        "oracle": _plan("oracle", oracle_idx, route_eval_map[oracle_idx].actual_profit),
    }
    return DriverPolicyEvaluation(
        driver_id=driver.driver_id,
        route_evaluations=tuple(route_evals),
        plans=plans,
    )
