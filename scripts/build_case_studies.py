from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import h3
import folium
import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from shapely import wkt
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rendezvous.analysis import select_case_studies
from rendezvous.run_registry import backfill_legacy_runs, has_registered_runs, registered_file_paths
from rendezvous.selectors import DeterministicMeetingPointSelector

RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "paper_rendezvous" / "figures"
RAW_CONTEXT_DIR = ROOT / "data" / "urban_context" / "raw"
INTERACTIVE_MAP_DIR = ROOT / "results" / "interactive_maps"

BUILDINGS_PATH = RAW_CONTEXT_DIR / "building_footprints.csv"
SIDEWALKS_PATH = RAW_CONTEXT_DIR / "sidewalk_centerline.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build curated case-study overlays for rendezvous observability")
    parser.add_argument("--pairs", type=str, default=str(RESULTS_DIR / "rendezvous_observability_matched_pairs.csv"))
    parser.add_argument("--routes-pattern", type=str, default="rendezvous_route_evaluations*.csv")
    parser.add_argument("--opportunities-pattern", type=str, default="rendezvous_route_opportunities*.csv")
    parser.add_argument("--total-cases", type=int, default=8)
    args = parser.parse_args()

    backfill_legacy_runs(RESULTS_DIR)
    matched_pairs = _load_csv(Path(args.pairs))
    route_df = _load_artifact_frames(role="route_evaluations", pattern=args.routes_pattern, run_kind="driver")
    opportunity_df = _load_artifact_frames(role="route_opportunities", pattern=args.opportunities_pattern, run_kind="driver")
    if matched_pairs.empty or route_df.empty or opportunity_df.empty:
        raise SystemExit("Matched pairs, route evaluations, and route opportunities are all required.")

    selected = select_case_studies(matched_pairs, total_cases=args.total_cases)
    if selected.empty:
        raise SystemExit("No case studies matched the requested filters.")

    route_df["area_slice"] = route_df.get("area_slice", "all").fillna("all")
    opportunity_df["area_slice"] = opportunity_df.get("area_slice", "all").fillna("all")

    selected_cases = _materialize_cases(selected, route_df, opportunity_df)
    if not selected_cases:
        raise SystemExit("Selected case studies could not be materialized from the available artifacts.")

    geometry_by_case = _load_geometry_for_cases(selected_cases)
    rows: list[dict[str, object]] = []
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    INTERACTIVE_MAP_DIR.mkdir(parents=True, exist_ok=True)
    for case_index, case in enumerate(selected_cases, start=1):
        case_geometry = geometry_by_case.get(case["case_id"], {})
        row_records = _score_case(case_index, case, case_geometry)
        rows.extend(row_records)
        _save_case_panel(case_index, case, case_geometry, include_legend=False)
        _save_case_interactive_map(case_index, case, case_geometry)

    case_df = pd.DataFrame(rows)
    case_df.to_csv(RESULTS_DIR / "rendezvous_case_studies.csv", index=False)
    _write_agreement_summary(case_df)
    mechanism_case = _choose_mechanism_case(case_df)
    mechanism_lookup = {
        idx: (case, geometry_by_case.get(case["case_id"], {}))
        for idx, case in enumerate(selected_cases, start=1)
    }
    if mechanism_case in mechanism_lookup:
        mechanism_case_data, mechanism_geometry = mechanism_lookup[mechanism_case]
        _save_case_panel(
            mechanism_case,
            mechanism_case_data,
            mechanism_geometry,
            include_legend=True,
            output_name="rendezvous_fig2_matched_pair_mechanism",
            title_override=(
                f"Illustrative matched pair: driver {mechanism_case_data['driver_id']}, "
                f"routes {mechanism_case_data['high_route_idx']} vs {mechanism_case_data['low_route_idx']} "
                f"({str(mechanism_case_data['time_slice']).replace('_', ' ')})"
            ),
        )
    _build_manuscript_figure(case_df)
    _build_appendix_figure(case_df)


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_glob(pattern: str) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in sorted(RESULTS_DIR.glob(pattern))]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_artifact_frames(*, role: str, pattern: str, run_kind: str | None = None) -> pd.DataFrame:
    if has_registered_runs(RESULTS_DIR):
        frames = [pd.read_csv(path) for path in registered_file_paths(RESULTS_DIR, role=role, run_kind=run_kind)]
        if frames:
            return pd.concat(frames, ignore_index=True)
    return _load_glob(pattern)


def _materialize_cases(
    selected: pd.DataFrame,
    route_df: pd.DataFrame,
    opportunity_df: pd.DataFrame,
) -> list[dict[str, object]]:
    selector = DeterministicMeetingPointSelector(use_observability=True)
    cases: list[dict[str, object]] = []
    key_cols = ["domain", "scenario_name", "time_slice", "area_slice", "driver_id"]
    for idx, row in selected.reset_index(drop=True).iterrows():
        key_map = {column: row[column] for column in key_cols}
        route_rows = route_df.copy()
        for column, value in key_map.items():
            route_rows = route_rows[route_rows[column] == value]
        high_route = route_rows[route_rows["route_idx"] == row["high_route_idx"]]
        low_route = route_rows[route_rows["route_idx"] == row["low_route_idx"]]
        if high_route.empty or low_route.empty:
            continue
        high_route = high_route.iloc[0]
        low_route = low_route.iloc[0]

        high_opp = _subset_opportunities(opportunity_df, key_map, int(row["high_route_idx"]))
        low_opp = _subset_opportunities(opportunity_df, key_map, int(row["low_route_idx"]))
        if high_opp.empty or low_opp.empty:
            continue
        high_selected = _selected_anchor_rows(high_opp, selector)
        low_selected = _selected_anchor_rows(low_opp, selector)

        case_id = f"case_{idx + 1:02d}"
        cases.append(
            {
                "case_id": case_id,
                "domain": row["domain"],
                "scenario_name": row["scenario_name"],
                "time_slice": row["time_slice"],
                "area_slice": row.get("area_slice", "all"),
                "driver_id": int(row["driver_id"]),
                "high_route_idx": int(row["high_route_idx"]),
                "low_route_idx": int(row["low_route_idx"]),
                "pair_row": row,
                "high_route": high_route,
                "low_route": low_route,
                "high_opportunities": high_opp,
                "low_opportunities": low_opp,
                "high_selected": high_selected,
                "low_selected": low_selected,
            }
        )
    return cases


def _subset_opportunities(
    opportunity_df: pd.DataFrame,
    key_map: dict[str, object],
    route_idx: int,
) -> pd.DataFrame:
    subset = opportunity_df.copy()
    for column, value in key_map.items():
        subset = subset[subset[column] == value]
    return subset[subset["route_idx"] == route_idx].reset_index(drop=True)


def _selected_anchor_rows(opportunity_df: pd.DataFrame, selector: DeterministicMeetingPointSelector) -> pd.DataFrame:
    from rendezvous.data_types import RendezvousOpportunity

    opportunities = tuple(
        RendezvousOpportunity(
            rider_id=int(row.rider_id),
            anchor_cell=str(row.anchor_cell),
            anchor_idx=int(row.anchor_idx),
            pickup_h3=str(row.pickup_h3),
            dropoff_h3=str(row.dropoff_h3),
            fare_share=float(row.fare_share),
            passenger_count=int(row.passenger_count),
            walk_m=float(row.walk_m),
            walk_min=float(row.walk_min),
            anchor_progress=float(row.anchor_progress),
            travel_fraction=float(row.travel_fraction),
            ambiguity_count=int(row.ambiguity_count),
            local_straightness=float(row.local_straightness),
            turn_severity=float(row.turn_severity),
            anchor_clutter=float(row.anchor_clutter),
            urban_clutter_index=float(row.urban_clutter_index),
            sidewalk_access_score=float(row.sidewalk_access_score),
            building_height_proxy=float(row.building_height_proxy),
            context_is_imputed=_as_bool(row.context_is_imputed),
            observability_score=float(row.observability_score),
            success_probability=float(row.success_probability),
        )
        for row in opportunity_df.itertuples(index=False)
    )
    selected = selector.select(opportunities, seats=3)
    selected_keys = {(item.rider_id, item.anchor_idx, item.anchor_cell) for item in selected}
    return opportunity_df[
        opportunity_df.apply(
            lambda row: (int(row["rider_id"]), int(row["anchor_idx"]), str(row["anchor_cell"])) in selected_keys,
            axis=1,
        )
    ].reset_index(drop=True)


def _load_geometry_for_cases(cases: list[dict[str, object]]) -> dict[str, dict[str, list[object]]]:
    bboxes = {case["case_id"]: _case_bbox(case) for case in cases}
    building_shapes = {case_id: [] for case_id in bboxes}
    sidewalk_shapes = {case_id: [] for case_id in bboxes}

    _scan_geometry_csv(BUILDINGS_PATH, bboxes, building_shapes, max_per_case=600)
    _scan_geometry_csv(SIDEWALKS_PATH, bboxes, sidewalk_shapes, max_per_case=400)
    return {
        case_id: {"buildings": building_shapes[case_id], "sidewalks": sidewalk_shapes[case_id]}
        for case_id in bboxes
    }


def _case_bbox(case: dict[str, object]):
    candidate_cells = []
    for frame_key in ["high_opportunities", "low_opportunities", "high_selected", "low_selected"]:
        df = case.get(frame_key)
        if df is None or df.empty:
            continue
        candidate_cells.extend(df["anchor_cell"].astype(str).tolist())
        if "pickup_h3" in df.columns:
            candidate_cells.extend(df["pickup_h3"].astype(str).tolist())
    if candidate_cells:
        points = [h3.cell_to_latlng(cell) for cell in candidate_cells]
        lats = [lat for lat, _lng in points]
        lngs = [lng for _lat, lng in points]
        padding = 0.0026
        return box(min(lngs) - padding, min(lats) - padding, max(lngs) + padding, max(lats) + padding)

    high_polyline = json.loads(case["high_route"]["polyline_json"])
    low_polyline = json.loads(case["low_route"]["polyline_json"])
    lats = [point[0] for point in high_polyline + low_polyline]
    lngs = [point[1] for point in high_polyline + low_polyline]
    padding = 0.003
    return box(min(lngs) - padding, min(lats) - padding, max(lngs) + padding, max(lats) + padding)


def _scan_geometry_csv(
    path: Path,
    bboxes: dict[str, object],
    target: dict[str, list[object]],
    *,
    max_per_case: int,
) -> None:
    if not path.exists():
        return
    active_cases = set(bboxes)
    for chunk in pd.read_csv(path, usecols=["the_geom"], chunksize=20000):
        if not active_cases:
            break
        for geom_text in chunk["the_geom"].dropna():
            if not active_cases:
                break
            try:
                geom = wkt.loads(str(geom_text))
            except Exception:
                continue
            bounds = box(*geom.bounds)
            for case_id in list(active_cases):
                if len(target[case_id]) >= max_per_case:
                    active_cases.discard(case_id)
                    continue
                if bounds.intersects(bboxes[case_id]):
                    target[case_id].append(geom)


def _score_case(case_index: int, case: dict[str, object], geometry: dict[str, list[object]]) -> list[dict[str, object]]:
    pair_row = case["pair_row"]
    rows = []
    for role, route_key in [("higher_observability", "high"), ("lower_observability", "low")]:
        route = case[f"{route_key}_route"]
        opportunities = case[f"{route_key}_opportunities"]
        selected = case[f"{route_key}_selected"]
        rubric = _rubric_scores(route, opportunities, selected, geometry)
        rows.append(
            {
                "case_id": case["case_id"],
                "case_rank": case_index,
                "domain": case["domain"],
                "scenario_name": case["scenario_name"],
                "time_slice": case["time_slice"],
                "area_slice": case["area_slice"],
                "driver_id": case["driver_id"],
                "route_role": role,
                "route_idx": int(route["route_idx"]),
                "mean_route_observability": float(route["mean_route_observability"]),
                "mean_route_walk_min": float(route["mean_route_walk_min"]),
                "route_distance_miles": float(route["route_distance_miles"]),
                "route_cost": float(route["route_cost"]),
                "observability_gap": float(pair_row["mean_observability_gap"]),
                "mean_profit_delta": float(pair_row["mean_profit_delta"]),
                **rubric,
                "panel_path": str(FIG_DIR / f"rendezvous_case_study_{case_index:02d}.png"),
            }
        )
    return rows


def _rubric_scores(
    route_row: pd.Series,
    opportunities: pd.DataFrame,
    selected: pd.DataFrame,
    geometry: dict[str, list[object]],
) -> dict[str, object]:
    source = selected if not selected.empty else opportunities
    if source.empty:
        return {
            "approach_legibility": 1,
            "anchor_ambiguity": 1,
            "sidewalk_continuity": 1,
            "local_openness": 1,
            "rubric_total": 4,
        }

    anchor_gdf = _anchor_points_gdf(source).to_crs(epsg=3857)
    anchor_geom = unary_union(anchor_gdf.geometry.tolist()) if not anchor_gdf.empty else None
    anchor_point = anchor_geom.centroid if anchor_geom is not None and not anchor_geom.is_empty else None

    pickup_gdf = _pickup_points_gdf(source).to_crs(epsg=3857)
    pickup_geom = unary_union(pickup_gdf.geometry.tolist()) if not pickup_gdf.empty else None
    pickup_point = pickup_geom.centroid if pickup_geom is not None and not pickup_geom.is_empty else None

    walk_link = None
    if pickup_point is not None and anchor_point is not None:
        walk_link = LineString([pickup_point, anchor_point])

    route_web = _route_line_gdf(route_row).to_crs(epsg=3857).geometry.iloc[0]
    turn = _line_turn_near_point(route_web, anchor_point) if anchor_point is not None else 0.5
    walk_m = float(walk_link.length) if walk_link is not None else 0.0
    approach = 0.65 * max(0.0, 1.0 - turn) + 0.35 * max(0.0, 1.0 - min(walk_m / 300.0, 1.0))

    selected_riders = set(source["rider_id"].astype(int).tolist()) if "rider_id" in source.columns else set()
    candidate_subset = opportunities[opportunities["rider_id"].isin(selected_riders)] if selected_riders else opportunities
    ambiguity_count = max(int(candidate_subset["anchor_cell"].astype(str).nunique()), 1)
    ambiguity_score = 1.0 / min(float(ambiguity_count), 4.0)

    sidewalk_score = _sidewalk_continuity_score(geometry.get("sidewalks", []), walk_link, anchor_point)
    openness_score = _local_openness_score(geometry.get("buildings", []), anchor_point)

    return {
        "approach_legibility": _bucket(approach),
        "anchor_ambiguity": _bucket(ambiguity_score),
        "sidewalk_continuity": _bucket(sidewalk_score),
        "local_openness": _bucket(openness_score),
        "rubric_total": int(
            _bucket(approach)
            + _bucket(ambiguity_score)
            + _bucket(sidewalk_score)
            + _bucket(openness_score)
        ),
    }


def _bucket(value: float) -> int:
    if value >= 0.67:
        return 3
    if value >= 0.34:
        return 2
    return 1


def _line_turn_near_point(route_line: LineString, anchor_point: Point | None) -> float:
    if anchor_point is None or route_line.is_empty or route_line.length <= 1e-9:
        return 0.5
    center = route_line.project(anchor_point)
    delta = min(40.0, route_line.length / 6.0)
    before = route_line.interpolate(max(0.0, center - delta))
    current = route_line.interpolate(center)
    after = route_line.interpolate(min(route_line.length, center + delta))
    if before.equals(current) or current.equals(after):
        return 0.0
    incoming = math.atan2(current.y - before.y, current.x - before.x)
    outgoing = math.atan2(after.y - current.y, after.x - current.x)
    diff = abs(outgoing - incoming)
    while diff > math.pi:
        diff -= 2 * math.pi
    return min(abs(diff) / math.pi, 1.0)


def _sidewalk_continuity_score(sidewalks: list[object], walk_link: LineString | None, anchor_point: Point | None) -> float:
    if not sidewalks:
        return 0.5
    geoms = [geom for geom in sidewalks if geom is not None and not geom.is_empty]
    if not geoms:
        return 0.5
    sidewalk_union = unary_union(geoms)
    if walk_link is not None and walk_link.length > 1e-9:
        support = walk_link.buffer(18.0, cap_style=2)
        overlap = sidewalk_union.intersection(support).length
        return max(0.0, min(1.0, overlap / max(walk_link.length, 1.0)))
    if anchor_point is None:
        return 0.5
    support = anchor_point.buffer(25.0)
    overlap = sidewalk_union.intersection(support).length
    return max(0.0, min(1.0, overlap / 120.0))


def _local_openness_score(buildings: list[object], anchor_point: Point | None) -> float:
    if not buildings or anchor_point is None:
        return 0.5
    geoms = [geom for geom in buildings if geom is not None and not geom.is_empty]
    if not geoms:
        return 0.5
    building_union = unary_union(geoms)
    support = anchor_point.buffer(65.0)
    coverage = building_union.intersection(support).area / max(support.area, 1.0)
    return max(0.0, min(1.0, 1.0 - coverage))


def _case_legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color="#143d59", lw=2.3, label="Route"),
        Line2D([0], [0], color="#bc6c25", lw=6.0, alpha=0.28, label="Corridor"),
        Line2D([0], [0], color="#6c757d", lw=1.2, linestyle="--", label="Walk-to-anchor link"),
        Line2D([0], [0], marker="o", linestyle="", markersize=5.8, markerfacecolor="#6f1d9b", markeredgecolor="white", label="Selected rider pickup"),
        Line2D([0], [0], marker="o", linestyle="", markersize=5.5, markerfacecolor="#2a9d8f", markeredgecolor="white", label="Candidate anchor"),
        Line2D([0], [0], marker="*", linestyle="", markersize=9, markerfacecolor="#d62828", markeredgecolor="white", label="Selected anchor"),
    ]


def _save_case_panel(
    case_index: int,
    case: dict[str, object],
    geometry: dict[str, list[object]],
    *,
    include_legend: bool,
    output_name: str | None = None,
    title_override: str | None = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.8))
    high_bounds = _compute_panel_bounds(case["high_route"], case["high_opportunities"], case["high_selected"], geometry)
    low_bounds = _compute_panel_bounds(case["low_route"], case["low_opportunities"], case["low_selected"], geometry)
    shared_bounds = (
        min(high_bounds[0], low_bounds[0]),
        min(high_bounds[1], low_bounds[1]),
        max(high_bounds[2], low_bounds[2]),
        max(high_bounds[3], low_bounds[3]),
    )
    for ax, route_key, title in [
        (axes[0], "high", "Higher observability"),
        (axes[1], "low", "Lower observability"),
    ]:
        route_row = case[f"{route_key}_route"]
        opportunities_df = case[f"{route_key}_opportunities"]
        selected_df = case[f"{route_key}_selected"]
        _plot_route_panel(ax, route_row, opportunities_df, selected_df, geometry, fixed_bounds=shared_bounds)
        route_row = case[f"{route_key}_route"]
        ax.set_title(title, fontsize=12.0, pad=7)
        metrics = _panel_metrics(route_row, opportunities_df, selected_df)
        box_x, box_y, box_ha, box_va = _panel_metrics_box_position(
            case_index=case_index,
            route_key=route_key,
            output_name=output_name,
        )
        ax.text(
            box_x,
            box_y,
            (
                f"Route obs. {metrics['route_obs']:.2f}\n"
                f"Anchor obs. {metrics['anchor_obs']:.2f}\n"
                f"p_succ {metrics['anchor_psucc']:.2f}\n"
                f"Walk {metrics['walk_min']:.2f} min\n"
                f"Route {metrics['route_mi']:.2f} mi"
            ),
            transform=ax.transAxes,
            ha=box_ha,
            va=box_va,
            fontsize=11.0,
            color="#1f1f1f",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#bcbcbc", alpha=0.9),
        )
    panel_title = title_override or (
        f"Case {case_index}: driver {case['driver_id']}, routes {case['high_route_idx']} vs {case['low_route_idx']} "
        f"({case['time_slice'].replace('_', ' ')})"
    )
    fig.suptitle(panel_title, fontsize=16.5, fontweight="semibold", y=1.03)
    if include_legend:
        fig.legend(
            handles=_case_legend_handles(),
            loc="lower center",
            ncol=3,
            frameon=False,
            fontsize=11.0,
            bbox_to_anchor=(0.5, -0.005),
        )
        fig.subplots_adjust(bottom=0.17, top=0.84, wspace=0.04)
    else:
        fig.subplots_adjust(bottom=0.04, top=0.84, wspace=0.04)
    stem = output_name or f"rendezvous_case_study_{case_index:02d}"
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=260, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _panel_metrics(route_row: pd.Series, opportunities_df: pd.DataFrame, selected_df: pd.DataFrame) -> dict[str, float]:
    source_df = selected_df if not selected_df.empty else opportunities_df
    if source_df.empty:
        anchor_obs = float(route_row["mean_route_observability"])
        anchor_psucc = float(route_row.get("mean_route_observability", 0.0))
    else:
        anchor_obs = float(source_df["observability_score"].mean()) if "observability_score" in source_df.columns else float(route_row["mean_route_observability"])
        anchor_psucc = float(source_df["success_probability"].mean()) if "success_probability" in source_df.columns else anchor_obs
    return {
        "route_obs": float(route_row["mean_route_observability"]),
        "anchor_obs": anchor_obs,
        "anchor_psucc": anchor_psucc,
        "walk_min": float(route_row["mean_route_walk_min"]),
        "route_mi": float(route_row["route_distance_miles"]),
    }


def _panel_metrics_box_position(*, case_index: int, route_key: str, output_name: str | None) -> tuple[float, float, str, str]:
    if output_name == "rendezvous_fig2_matched_pair_mechanism":
        if route_key == "low":
            return 0.98, 0.02, "right", "bottom"
        return 0.02, 0.98, "left", "top"
    if case_index == 1:
        return 0.02, 0.02, "left", "bottom"
    if case_index == 2:
        return 0.98, 0.02, "right", "bottom"
    return 0.02, 0.98, "left", "top"


def _compute_panel_bounds(
    route_row: pd.Series,
    opportunity_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    geometry: dict[str, list[object]],
) -> tuple[float, float, float, float]:
    route_gdf = _route_line_gdf(route_row)
    corridor_gdf = _corridor_hexes_gdf(route_row)
    anchor_gdf = _anchor_points_gdf(opportunity_df)
    selected_gdf = _anchor_points_gdf(selected_df)
    pickup_gdf = _pickup_points_gdf(selected_df)
    walklink_gdf = _walk_link_gdf(selected_df)
    building_gdf = _geometry_gdf(geometry.get("buildings", []))
    sidewalk_gdf = _geometry_gdf(geometry.get("sidewalks", []))

    layers = [item for item in [route_gdf, corridor_gdf, anchor_gdf, selected_gdf, pickup_gdf, walklink_gdf, building_gdf, sidewalk_gdf] if item is not None and not item.empty]
    if not layers:
        return (0.0, 0.0, 1.0, 1.0)

    total = gpd.GeoDataFrame(pd.concat(layers, ignore_index=True), geometry="geometry", crs="EPSG:4326").to_crs(epsg=3857)
    focus_layers = []
    if not selected_gdf.empty:
        focus_layers.append(selected_gdf)
    if not pickup_gdf.empty:
        focus_layers.append(pickup_gdf)
    if not walklink_gdf.empty:
        focus_layers.append(walklink_gdf)
    if not anchor_gdf.empty and len(anchor_gdf) <= 8:
        focus_layers.append(anchor_gdf)
    focus = (
        gpd.GeoDataFrame(pd.concat(focus_layers, ignore_index=True), geometry="geometry", crs="EPSG:4326").to_crs(epsg=3857)
        if focus_layers
        else total
    )
    bounds = focus.total_bounds
    route_bounds = route_gdf.to_crs(epsg=3857).total_bounds
    pad_x = max((bounds[2] - bounds[0]) * 0.22, 65)
    pad_y = max((bounds[3] - bounds[1]) * 0.22, 65)
    left = max(bounds[0] - pad_x, route_bounds[0] - 120)
    bottom = max(bounds[1] - pad_y, route_bounds[1] - 120)
    right = min(bounds[2] + pad_x, route_bounds[2] + 120)
    top = min(bounds[3] + pad_y, route_bounds[3] + 120)
    return (left, bottom, right, top)


def _plot_route_panel(
    ax,
    route_row: pd.Series,
    opportunity_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    geometry: dict[str, list[object]],
    *,
    fixed_bounds: tuple[float, float, float, float] | None = None,
) -> None:
    route_gdf = _route_line_gdf(route_row)
    corridor_gdf = _corridor_hexes_gdf(route_row)
    anchor_gdf = _anchor_points_gdf(opportunity_df)
    selected_gdf = _anchor_points_gdf(selected_df)
    pickup_gdf = _pickup_points_gdf(selected_df)
    walklink_gdf = _walk_link_gdf(selected_df)
    building_gdf = _geometry_gdf(geometry.get("buildings", []))
    sidewalk_gdf = _geometry_gdf(geometry.get("sidewalks", []))

    layers = [item for item in [route_gdf, corridor_gdf, anchor_gdf, selected_gdf, pickup_gdf, walklink_gdf, building_gdf, sidewalk_gdf] if item is not None and not item.empty]
    if not layers:
        ax.set_axis_off()
        return

    if not building_gdf.empty:
        building_gdf.to_crs(epsg=3857).plot(ax=ax, color="#d9d9d9", edgecolor="#b9b9b9", linewidth=0.15, alpha=0.55, zorder=2)
    if not sidewalk_gdf.empty:
        sidewalk_gdf.to_crs(epsg=3857).plot(ax=ax, color="#9ecae1", linewidth=0.8, alpha=0.6, zorder=3)
    if not corridor_gdf.empty:
        corridor_gdf.to_crs(epsg=3857).plot(ax=ax, color="#f4a261", edgecolor="#bc6c25", linewidth=0.18, alpha=0.18, zorder=4)
    route_web = route_gdf.to_crs(epsg=3857)
    route_web.plot(ax=ax, color="#143d59", linewidth=2.3, alpha=0.98, zorder=6)
    if not walklink_gdf.empty:
        walklink_gdf.to_crs(epsg=3857).plot(ax=ax, color="#6c757d", linewidth=1.1, alpha=0.9, linestyle="--", zorder=6.4)
    if not pickup_gdf.empty:
        pickup_gdf.to_crs(epsg=3857).plot(ax=ax, color="#6f1d9b", markersize=34, alpha=0.92, edgecolor="white", linewidth=0.45, zorder=6.8)
    if not anchor_gdf.empty:
        anchor_gdf.to_crs(epsg=3857).plot(ax=ax, color="#2a9d8f", markersize=18, alpha=0.78, edgecolor="white", linewidth=0.35, zorder=7)
    if not selected_gdf.empty:
        selected_gdf.to_crs(epsg=3857).plot(ax=ax, color="#d62828", markersize=115, marker="*", edgecolor="white", linewidth=0.5, zorder=8)

    bounds = fixed_bounds if fixed_bounds is not None else _compute_panel_bounds(route_row, opportunity_df, selected_df, geometry)
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("auto")
    for spine in ax.spines.values():
        spine.set_visible(False)


def _route_line_gdf(route_row: pd.Series) -> gpd.GeoDataFrame:
    polyline = json.loads(route_row["polyline_json"])
    line = LineString([(point[1], point[0]) for point in polyline])
    return gpd.GeoDataFrame({"geometry": [line]}, geometry="geometry", crs="EPSG:4326")


def _corridor_hexes_gdf(route_row: pd.Series) -> gpd.GeoDataFrame:
    corridor_cells = [cell for cell in str(route_row.get("corridor_cells", "")).split(";") if cell]
    polygons: list[Polygon] = []
    for cell in corridor_cells[:500]:
        boundary = h3.cell_to_boundary(cell)
        polygons.append(Polygon([(lng, lat) for lat, lng in boundary]))
    if not polygons:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")
    return gpd.GeoDataFrame({"geometry": polygons}, geometry="geometry", crs="EPSG:4326")


def _anchor_points_gdf(df: pd.DataFrame) -> gpd.GeoDataFrame:
    if df.empty:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")
    points = []
    for cell in df["anchor_cell"].astype(str):
        lat, lng = h3.cell_to_latlng(cell)
        points.append(Point(lng, lat))
    return gpd.GeoDataFrame({"geometry": points}, geometry="geometry", crs="EPSG:4326")


def _pickup_points_gdf(df: pd.DataFrame) -> gpd.GeoDataFrame:
    if df.empty:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")
    dedup = df[["rider_id", "pickup_h3"]].drop_duplicates()
    points = []
    for cell in dedup["pickup_h3"].astype(str):
        lat, lng = h3.cell_to_latlng(cell)
        points.append(Point(lng, lat))
    return gpd.GeoDataFrame({"geometry": points}, geometry="geometry", crs="EPSG:4326")


def _walk_link_gdf(df: pd.DataFrame) -> gpd.GeoDataFrame:
    if df.empty:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")
    dedup = df[["rider_id", "pickup_h3", "anchor_cell"]].drop_duplicates()
    lines = []
    for row in dedup.itertuples(index=False):
        plat, plng = h3.cell_to_latlng(str(row.pickup_h3))
        alat, alng = h3.cell_to_latlng(str(row.anchor_cell))
        lines.append(LineString([(plng, plat), (alng, alat)]))
    return gpd.GeoDataFrame({"geometry": lines}, geometry="geometry", crs="EPSG:4326")


def _geometry_gdf(geoms: list[object]) -> gpd.GeoDataFrame:
    if not geoms:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")
    flat = []
    for geom in geoms:
        flat.extend(getattr(geom, "geoms", [geom]))
    return gpd.GeoDataFrame({"geometry": flat}, geometry="geometry", crs="EPSG:4326")


def _save_case_interactive_map(case_index: int, case: dict[str, object], geometry: dict[str, list[object]]) -> None:
    high_route = case["high_route"]
    low_route = case["low_route"]
    bbox = _case_bbox(case)
    center = [(bbox.bounds[1] + bbox.bounds[3]) / 2.0, (bbox.bounds[0] + bbox.bounds[2]) / 2.0]
    fmap = folium.Map(location=center, zoom_start=16, tiles="CartoDB positron", control_scale=True)
    for label, route_row, opportunity_df, selected_df, color in [
        ("Higher observability", high_route, case["high_opportunities"], case["high_selected"], "#0b6e4f"),
        ("Lower observability", low_route, case["low_opportunities"], case["low_selected"], "#b02e0c"),
    ]:
        polyline = json.loads(route_row["polyline_json"])
        folium.PolyLine(polyline, color=color, weight=5, opacity=0.9, tooltip=label).add_to(fmap)
        corridor_cells = [cell for cell in str(route_row.get("corridor_cells", "")).split(";") if cell]
        for cell in corridor_cells[:180]:
            boundary = [(lat, lng) for lat, lng in h3.cell_to_boundary(cell)]
            folium.Polygon(boundary, color=color, weight=0.5, fill=True, fill_opacity=0.08).add_to(fmap)
        for row in opportunity_df.itertuples(index=False):
            lat, lng = h3.cell_to_latlng(str(row.anchor_cell))
            folium.CircleMarker(
                location=(lat, lng),
                radius=3.2,
                color="#1d3557",
                weight=1,
                fill=True,
                fill_color="#8ecae6",
                fill_opacity=0.8,
                tooltip=f"{label} anchor {int(row.anchor_idx)}",
            ).add_to(fmap)
        for row in selected_df.itertuples(index=False):
            lat, lng = h3.cell_to_latlng(str(row.anchor_cell))
            folium.Marker(
                location=(lat, lng),
                icon=folium.Icon(color="red", icon="star"),
                tooltip=f"{label} selected anchor",
            ).add_to(fmap)
            plat, plng = h3.cell_to_latlng(str(row.pickup_h3))
            folium.CircleMarker(
                location=(plat, plng),
                radius=4.0,
                color="#6f1d9b",
                weight=1,
                fill=True,
                fill_color="#6f1d9b",
                fill_opacity=0.88,
                tooltip=f"{label} rider pickup",
            ).add_to(fmap)
            folium.PolyLine(
                locations=[(plat, plng), (lat, lng)],
                color="#6c757d",
                weight=2,
                opacity=0.9,
                dash_array="6,6",
                tooltip=f"{label} walk-to-anchor",
            ).add_to(fmap)
    fmap.fit_bounds([[bbox.bounds[1], bbox.bounds[0]], [bbox.bounds[3], bbox.bounds[2]]])
    fmap.save(str(INTERACTIVE_MAP_DIR / f"rendezvous_case_study_{case_index:02d}.html"))


def _write_agreement_summary(case_df: pd.DataFrame) -> None:
    if case_df.empty:
        return
    pivot = case_df.pivot_table(
        index=["case_id", "case_rank", "domain", "scenario_name", "time_slice", "driver_id"],
        columns="route_role",
        values="rubric_total",
    ).reset_index()
    if {"higher_observability", "lower_observability"}.issubset(pivot.columns):
        pivot["agreement_with_model_preference"] = (
            pivot["higher_observability"] >= pivot["lower_observability"]
        ).astype(int)
    pivot.to_csv(RESULTS_DIR / "rendezvous_case_study_agreement.csv", index=False)


def _build_manuscript_figure(case_df: pd.DataFrame) -> None:
    if case_df.empty:
        return
    mechanism_case = _choose_mechanism_case(case_df)
    case_scores = _case_pair_summary(case_df)
    top_cases = [rank for rank in case_scores["case_rank"].tolist() if int(rank) != int(mechanism_case)][:2]
    if len(top_cases) < 2:
        top_cases = case_scores["case_rank"].tolist()[:2]
    top_cases = sorted(top_cases)
    images = [FIG_DIR / f"rendezvous_case_study_{rank:02d}.png" for rank in top_cases]
    _compose_case_figure(images, FIG_DIR / "rendezvous_fig9_case_studies.png", ncols=1)


def _build_appendix_figure(case_df: pd.DataFrame) -> None:
    if case_df.empty:
        return
    case_ranks = case_df["case_rank"].drop_duplicates().sort_values().tolist()
    images = [FIG_DIR / f"rendezvous_case_study_{rank:02d}.png" for rank in case_ranks]
    _compose_case_figure(images, FIG_DIR / "rendezvous_appendix_case_studies.png", ncols=2)


def _choose_mechanism_case(case_df: pd.DataFrame) -> int:
    pair = _case_pair_summary(case_df)
    required = {"obs_gap", "mean_profit_delta", "walk_gap", "dist_gap"}
    if not required.issubset(set(pair.columns)):
        return int(case_df["case_rank"].min())
    focus = pair[
        (pair["domain"] == "yellow")
        & (pair["scenario_name"] == "sparse_high_occlusion")
        & (pair["mean_profit_delta"] > 0.0)
    ].copy()
    if focus.empty:
        focus = pair[pair["mean_profit_delta"] > 0.0].copy()
    if focus.empty:
        return int(case_df["case_rank"].min())
    focus["visual_score"] = 6.0 * focus["obs_gap"] + 1.5 * focus["walk_gap"] + 0.5 * focus["dist_gap"]
    best = focus.sort_values(["visual_score", "mean_profit_delta", "obs_gap"], ascending=[False, False, False]).iloc[0]
    return int(best["case_rank"])


def _case_pair_summary(case_df: pd.DataFrame) -> pd.DataFrame:
    pair = case_df.pivot_table(
        index=["case_rank", "domain", "scenario_name", "time_slice", "driver_id"],
        columns="route_role",
        values=[
            "mean_route_observability",
            "mean_route_walk_min",
            "route_distance_miles",
            "mean_profit_delta",
            "rubric_total",
        ],
        aggfunc="first",
    ).reset_index()
    pair.columns = ["_".join([str(c) for c in col if c != ""]).strip("_") for col in pair.columns]
    required = {
        "mean_route_observability_higher_observability",
        "mean_route_observability_lower_observability",
        "mean_route_walk_min_higher_observability",
        "mean_route_walk_min_lower_observability",
        "route_distance_miles_higher_observability",
        "route_distance_miles_lower_observability",
        "mean_profit_delta_higher_observability",
    }
    if not required.issubset(set(pair.columns)):
        return pd.DataFrame()
    pair["obs_gap"] = (
        pair["mean_route_observability_higher_observability"]
        - pair["mean_route_observability_lower_observability"]
    )
    pair["walk_gap"] = (
        pair["mean_route_walk_min_higher_observability"]
        - pair["mean_route_walk_min_lower_observability"]
    ).abs()
    pair["dist_gap"] = (
        pair["route_distance_miles_higher_observability"]
        - pair["route_distance_miles_lower_observability"]
    ).abs()
    pair["mean_profit_delta"] = pair["mean_profit_delta_higher_observability"]
    pair["visual_score"] = 6.0 * pair["obs_gap"] + 1.5 * pair["walk_gap"] + 0.5 * pair["dist_gap"]
    return pair.sort_values(
        ["visual_score", "mean_profit_delta", "obs_gap"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _stitch_images(
    image_paths: list[Path],
    output_path: Path,
    *,
    ncols: int,
    crop_bottom_ratio: float = 0.0,
) -> None:
    from PIL import Image

    existing = [Image.open(path) for path in image_paths if path.exists()]
    if not existing:
        return
    cropped: list[Image.Image] = []
    for image in existing:
        crop_px = int(image.height * max(crop_bottom_ratio, 0.0))
        if crop_px > 0 and crop_px < image.height - 4:
            cropped.append(image.crop((0, 0, image.width, image.height - crop_px)))
        else:
            cropped.append(image.copy())
        image.close()
    width = max(image.width for image in cropped)
    height = max(image.height for image in cropped)
    ncols = max(ncols, 1)
    nrows = math.ceil(len(cropped) / ncols)
    canvas = Image.new("RGB", (width * ncols, height * nrows), color="white")
    for idx, image in enumerate(cropped):
        row = idx // ncols
        col = idx % ncols
        canvas.paste(image, (col * width, row * height))
        image.close()
    canvas.save(output_path)
    canvas.save(output_path.with_suffix(".pdf"), "PDF", resolution=220.0)


def _compose_case_figure(image_paths: list[Path], output_path: Path, *, ncols: int) -> None:
    from PIL import Image

    existing = [path for path in image_paths if path.exists()]
    if not existing:
        return
    images = [Image.open(path).convert("RGBA") for path in existing]
    arrays = [np.asarray(image) for image in images]
    ncols = max(ncols, 1)
    nrows = math.ceil(len(arrays) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.2, 5.8 * nrows))
    axes_arr = np.atleast_1d(axes).ravel()
    for ax, arr in zip(axes_arr, arrays):
        ax.imshow(arr)
        ax.axis("off")
    for ax in axes_arr[len(arrays):]:
        ax.axis("off")
    fig.legend(
        handles=_case_legend_handles(),
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=11.0,
        bbox_to_anchor=(0.5, 0.015),
    )
    fig.subplots_adjust(bottom=0.10, top=0.98, hspace=0.08, wspace=0.04)
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    for image in images:
        image.close()


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


if __name__ == "__main__":
    main()
