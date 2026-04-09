from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import h3
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Polygon as MplPolygon
from shapely import wkt
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rendezvous.analysis import select_case_studies
from rendezvous.selectors import DeterministicMeetingPointSelector

RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "paper_rendezvous" / "figures"
RAW_CONTEXT_DIR = ROOT / "data" / "urban_context" / "raw"

BUILDINGS_PATH = RAW_CONTEXT_DIR / "building_footprints.csv"
SIDEWALKS_PATH = RAW_CONTEXT_DIR / "sidewalk_centerline.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build curated case-study overlays for rendezvous observability")
    parser.add_argument("--pairs", type=str, default=str(RESULTS_DIR / "rendezvous_observability_matched_pairs.csv"))
    parser.add_argument("--routes-pattern", type=str, default="rendezvous_route_evaluations*.csv")
    parser.add_argument("--opportunities-pattern", type=str, default="rendezvous_route_opportunities*.csv")
    parser.add_argument("--total-cases", type=int, default=8)
    args = parser.parse_args()

    matched_pairs = _load_csv(Path(args.pairs))
    route_df = _load_glob(args.routes_pattern)
    opportunity_df = _load_glob(args.opportunities_pattern)
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
    for case_index, case in enumerate(selected_cases, start=1):
        row_records = _score_case(case_index, case)
        rows.extend(row_records)
        _save_case_panel(case_index, case, geometry_by_case.get(case["case_id"], {}))

    case_df = pd.DataFrame(rows)
    case_df.to_csv(RESULTS_DIR / "rendezvous_case_studies.csv", index=False)
    _write_agreement_summary(case_df)
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


def _score_case(case_index: int, case: dict[str, object]) -> list[dict[str, object]]:
    pair_row = case["pair_row"]
    rows = []
    for role, route_key in [("higher_observability", "high"), ("lower_observability", "low")]:
        route = case[f"{route_key}_route"]
        opportunities = case[f"{route_key}_opportunities"]
        selected = case[f"{route_key}_selected"]
        rubric = _rubric_scores(opportunities, selected)
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


def _rubric_scores(opportunities: pd.DataFrame, selected: pd.DataFrame) -> dict[str, object]:
    source = selected if not selected.empty else opportunities
    straightness = float(source["local_straightness"].mean()) if "local_straightness" in source else 0.5
    turn = float(source["turn_severity"].mean()) if "turn_severity" in source else 0.5
    ambiguity = float(source["ambiguity_count"].mean()) if "ambiguity_count" in source else 2.0
    sidewalk = float(source["sidewalk_access_score"].mean()) if "sidewalk_access_score" in source else 0.5
    openness = 1.0 - min(
        1.0,
        float(source["urban_clutter_index"].mean()) + 0.5 * float(source["building_height_proxy"].mean()),
    )
    return {
        "approach_legibility": _bucket(0.6 * straightness + 0.4 * (1.0 - turn)),
        "anchor_ambiguity": _bucket(1.0 / max(ambiguity, 1.0)),
        "sidewalk_continuity": _bucket(sidewalk),
        "local_openness": _bucket(openness),
        "rubric_total": int(
            _bucket(0.6 * straightness + 0.4 * (1.0 - turn))
            + _bucket(1.0 / max(ambiguity, 1.0))
            + _bucket(sidewalk)
            + _bucket(openness)
        ),
    }


def _bucket(value: float) -> int:
    if value >= 0.67:
        return 3
    if value >= 0.34:
        return 2
    return 1


def _save_case_panel(case_index: int, case: dict[str, object], geometry: dict[str, list[object]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.0))
    for ax, route_key, title in [
        (axes[0], "high", "Higher observability"),
        (axes[1], "low", "Lower observability"),
    ]:
        _plot_route_panel(ax, case[f"{route_key}_route"], case[f"{route_key}_opportunities"], case[f"{route_key}_selected"], geometry)
        ax.set_title(title, fontsize=9)
    fig.suptitle(
        f"Case {case_index}: {case['domain'].title()} {case['scenario_name'].replace('_', ' ')} "
        f"({case['time_slice'].replace('_', ' ')})",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"rendezvous_case_study_{case_index:02d}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_route_panel(ax, route_row: pd.Series, opportunity_df: pd.DataFrame, selected_df: pd.DataFrame, geometry: dict[str, list[object]]) -> None:
    for geom in geometry.get("buildings", []):
        for poly in getattr(geom, "geoms", [geom]):
            x, y = poly.exterior.xy
            ax.add_patch(MplPolygon(list(zip(x, y)), closed=True, facecolor="#d9d9d9", edgecolor="none", alpha=0.55))
    for geom in geometry.get("sidewalks", []):
        for line in getattr(geom, "geoms", [geom]):
            x, y = line.xy
            ax.plot(x, y, color="#8ecae6", linewidth=0.6, alpha=0.7)

    polyline = json.loads(route_row["polyline_json"])
    route_x = [point[1] for point in polyline]
    route_y = [point[0] for point in polyline]
    ax.plot(route_x, route_y, color="#023047", linewidth=2.0)

    corridor_cells = [cell for cell in str(route_row.get("corridor_cells", "")).split(";") if cell]
    if corridor_cells:
        corridor_points = [h3.cell_to_latlng(cell) for cell in corridor_cells[:400]]
        ax.scatter(
            [point[1] for point in corridor_points],
            [point[0] for point in corridor_points],
            s=6,
            color="#ffb703",
            alpha=0.15,
            zorder=1,
        )
    if not opportunity_df.empty:
        anchor_points = [h3.cell_to_latlng(cell) for cell in opportunity_df["anchor_cell"].astype(str)]
        ax.scatter(
            [point[1] for point in anchor_points],
            [point[0] for point in anchor_points],
            s=12,
            color="#219ebc",
            alpha=0.55,
            zorder=3,
        )
    if not selected_df.empty:
        selected_points = [h3.cell_to_latlng(cell) for cell in selected_df["anchor_cell"].astype(str)]
        ax.scatter(
            [point[1] for point in selected_points],
            [point[0] for point in selected_points],
            s=48,
            color="#d62828",
            marker="*",
            zorder=4,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    ax.set_frame_on(False)


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
    top_cases = case_df["case_rank"].drop_duplicates().sort_values().tolist()[:3]
    images = [FIG_DIR / f"rendezvous_case_study_{rank:02d}.png" for rank in top_cases]
    _stitch_images(images, FIG_DIR / "rendezvous_fig9_case_studies.png", ncols=1)
    if images:
        (FIG_DIR / "rendezvous_fig2_matched_pair_mechanism.png").write_bytes(images[0].read_bytes())


def _build_appendix_figure(case_df: pd.DataFrame) -> None:
    if case_df.empty:
        return
    case_ranks = case_df["case_rank"].drop_duplicates().sort_values().tolist()
    images = [FIG_DIR / f"rendezvous_case_study_{rank:02d}.png" for rank in case_ranks]
    _stitch_images(images, FIG_DIR / "rendezvous_appendix_case_studies.png", ncols=2)


def _stitch_images(image_paths: list[Path], output_path: Path, *, ncols: int) -> None:
    from PIL import Image

    existing = [Image.open(path) for path in image_paths if path.exists()]
    if not existing:
        return
    width = max(image.width for image in existing)
    height = max(image.height for image in existing)
    ncols = max(ncols, 1)
    nrows = math.ceil(len(existing) / ncols)
    canvas = Image.new("RGB", (width * ncols, height * nrows), color="white")
    for idx, image in enumerate(existing):
        row = idx // ncols
        col = idx % ncols
        canvas.paste(image, (col * width, row * height))
        image.close()
    canvas.save(output_path)


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


if __name__ == "__main__":
    main()
