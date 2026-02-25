---

overview: A phased prompt for building the warm-up vs cold-start carpooling profit comparison system from scratch.
todos:
  - id: phase1-data
    content: "Phase 1: Data foundation -- download NYC TLC data WITH fare columns, preprocess, build H3 corridor engine, integrate OSRM for polylines"
    status: pending
  - id: phase2-ml
    content: "Phase 2: Profit prediction ML -- build labeled dataset (route features -> actual profit), train LightGBM regressor, validate with temporal split"
    status: pending
  - id: phase3-sim
    content: "Phase 3: Simulation -- implement cold-start (1 route) and warm-up (3 routes + ML ranking) pipelines, shared matching logic, multi-seed experiment runner"
    status: pending
  - id: phase4-viz
    content: "Phase 4: Visualization -- profit comparison plots (bar, box, cumulative), match rate, compute time, model insight plots, statistical summary"
    status: pending
  - id: phase5-polish
    content: "Phase 5: Polish -- README, run_all.py, --no-api fallback, ensure offline-capable via route cache"
    status: pending
isProject: false
---

# Detailed Prompt: Warm-Up vs Cold-Start Carpooling Profit Comparison

## Project Goal

Build an experimental system that **proves warm-up route selection is more profitable than cold-start** for real-time carpooling. A driver provides origin and destination. In **warm-up mode**, the system fetches up to 3 alternative routes, builds H3 corridors along each, uses an ML model to predict expected profit per route, ranks them, and lets the driver pick the best. In **cold-start mode**, the system fetches 1 route, builds its corridor, and the driver drives it hoping to match riders. The final output is comparative graphs showing profit difference.

---

## Phase 1: Data Foundation and Spatial Core

**Goal:** Get clean trip data with revenue, build the H3 corridor engine, integrate OSRM for road-following polylines.

### 1A: Dataset Selection and Preparation

**Use the NYC TLC yellow taxi dataset** (same source, but use it correctly this time).

Why NYC TLC works:

- Millions of trips with `pickup_datetime`, `dropoff_datetime`, `pickup_lat/lng`, `dropoff_lat/lng` (or zone IDs + centroids)
- **Crucially: contains `fare_amount`, `tip_amount`, `total_amount**` -- real revenue data the current codebase ignores
- Free, public, well-documented
- Dense enough to simulate realistic rider demand patterns along corridors

What to extract per trip:

- `pickup_lat`, `pickup_lng`, `dropoff_lat`, `dropoff_lng`
- `pickup_datetime` (for temporal features)
- `fare_amount` (this IS the rider's payment -- the revenue signal)
- `trip_distance_miles`
- `passenger_count`

Preprocessing:

- Filter to trips > 5 miles (to approximate longer-distance/intercity-like trips, not 3-block taxi hops)
- Filter to reasonable fares ($5-$200)
- Convert pickup/dropoff to H3 cells at resolution 9 (hex edge ~174 m, tight corridor for dense urban areas like NYC)
- Split into **drivers** (longer trips, > 10 miles) and **riders** (shorter trips that could be served by a detour off a driver's route)

### 1B: H3 Spatial Corridor Engine

Build a clean spatial module with:

- `geo_to_h3(lat, lng, resolution) -> str`
- `polyline_to_h3_cells(polyline: list[LatLng], resolution: int, step_m=200) -> list[str]` -- densify polyline, convert each point to H3
- `expand_corridor(cells: list[str], k: int) -> set[str]` -- k-ring expansion
- `build_corridor(polyline, resolution, buffer_meters) -> Corridor` where `Corridor` has `route_cells`, `corridor_cells`, `envelope_cells`

One module, no duplication. All matching files import from here.

### 1C: OSRM Route Integration

Uses [OSRM](https://project-osrm.org/) (Open Source Routing Machine) -- free, open-source, OpenStreetMap-based routing.
Public demo server for development; self-host for production throughput.

- `OSRMRouter.get_single_route(origin, dest) -> RouteInfo` -- returns polyline points, distance_m, duration_s
- `OSRMRouter.get_alternative_routes(origin, dest, max_alternatives=3) -> list[RouteInfo]` -- returns up to 3 routes
  - Requests native OSRM alternatives first; if the server returns fewer, generates additional routes by routing through waypoints offset perpendicular to the O-D axis (each is still a real road-network polyline)
- JSON file cache keyed by `(origin_rounded, dest_rounded)` to avoid repeat API calls
- Load cache into memory at startup, flush on exit (not read/write full file per call)
- Base URL configurable via `OSRM_BASE_URL` env var (defaults to public server)
- Auto rate-limiting (1 req/s) when using public server; disabled for self-hosted

**Deliverable:** Given any origin/dest pair, the system can produce 1 or 3 polylines and build H3 corridors along each.

---

## Phase 2: Profit Prediction ML Model

**Goal:** Train a model that, given a route and a time, predicts the expected profit from carpooling riders along that route.

### 2A: Define Profit Clearly

```
profit = total_rider_revenue - driver_cost

total_rider_revenue = sum(fare_share for each matched rider)
fare_share = rider_fare * platform_share_factor  (e.g., 0.5)

driver_cost = route_distance_miles * cost_per_mile  (e.g., IRS rate $0.67/mile)
```

This means profit depends on:

1. How many riders the route can match (demand along corridor)
2. How much those riders pay (fare distribution along corridor)
3. How far the driver drives (cost)

### 2B: Build the Training Dataset

For each historical driver trip:

1. Get 3 alternative routes from OSRM (use cache)
2. For each route, build H3 corridor
3. Count how many historical rider trips have **both** pickup AND dropoff inside or near the corridor, within a reasonable time window
4. Sum their `fare_amount * share_factor` as `expected_revenue`
5. Compute `driver_cost = route_distance * cost_per_mile`
6. Label: `expected_profit = expected_revenue - driver_cost`

Features per route:

- `route_distance_m`, `route_duration_s`
- `corridor_cell_count` (spatial coverage)
- `hour_of_day`, `day_of_week`, `is_weekend`
- `corridor_demand_density` -- mean historical pickup count per corridor cell for that time window
- `corridor_fare_density` -- mean historical fare per corridor cell
- `corridor_rider_count` -- count of historical riders matchable along this corridor
- `mean_rider_fare` -- average fare of matchable riders

**Critical: Use temporal splitting** -- train on months 1-9, test on months 10-12. Never random split for time-series data.

### 2C: Train the Model

**Recommended: LightGBM regressor** (or XGBoost).

Why tree-based:

- The research papers referenced in the project cite efficiency as key
- LightGBM handles mixed feature types (numeric + categorical) natively
- Fast inference (~microseconds per prediction) -- important for real-time route ranking
- Interpretable via feature importance (good for paper)
- Handles the tabular feature set perfectly

Target: `expected_profit` (continuous, in dollars)
Evaluation: RMSE, MAE, and R-squared on the temporal test set

**Alternative if tree models underperform:** A small feedforward neural network (2-3 hidden layers) with H3 cell embeddings. But start with LightGBM -- it will likely be sufficient and is much simpler.

### 2D: Model Validation

- Feature importance plot (which features drive profit prediction?)
- Predicted vs actual scatter plot
- Profit distribution by route rank (does rank 1 consistently have highest actual profit?)

**Deliverable:** A trained model that takes route features + time and outputs expected profit in dollars. Saved as a single file (pickle or joblib).

---

## Phase 3: Warm-Up vs Cold-Start Simulation

**Goal:** Run both strategies on the same test data and record profit for every driver.

### 3A: Data Structures

```python
@dataclass
class DriverTrip:
    driver_id: int
    origin: LatLng
    destination: LatLng
    departure_time: datetime
    seats: int
    max_detour_minutes: float

@dataclass
class RiderRequest:
    rider_id: int
    origin: LatLng
    destination: LatLng
    request_time: datetime
    fare: float  # KEEP THE FARE -- this is the revenue signal

@dataclass
class MatchResult:
    driver_id: int
    rider_id: int
    fare_share: float  # rider fare * platform share
    detour_minutes: float
    pickup_time: datetime

@dataclass
class DriverOutcome:
    driver_id: int
    strategy: str  # "warmup" or "coldstart"
    route_distance_miles: float
    route_duration_minutes: float
    matched_riders: int
    total_revenue: float
    driving_cost: float
    profit: float  # revenue - cost
    route_rank_chosen: int  # 1-3 for warmup, always 1 for coldstart
    compute_time_seconds: float
```

### 3B: Cold-Start Pipeline

For each driver:

1. Fetch **1 route** from OSRM
2. Build H3 corridor
3. Find all riders whose pickup AND dropoff fall within/near the corridor, within the driver's time window
4. Apply matching filters (capacity, time compatibility, max detour)
5. Match greedily (best detour first)
6. Record: `profit = sum(matched_rider_fares * share_factor) - (distance * cost_per_mile)`

### 3C: Warm-Up Pipeline

For each driver:

1. Fetch **3 routes** from OSRM
2. Build H3 corridor for each route
3. Run profit prediction model on each route
4. Rank routes by predicted profit (highest first)
5. Select top route (or let simulation "choose" top-1)
6. Match riders along chosen corridor (same matching logic as cold-start)
7. Record: actual profit (same formula), plus predicted profit and rank

### 3D: Matching Logic (Shared)

Implement a clean two-stage filter (simplify from three-door):

**Stage 1 -- Spatial filter:**

- Rider pickup H3 cell must be in driver corridor
- Rider dropoff H3 cell must be in driver corridor (or within k-ring of corridor)

**Stage 2 -- Feasibility filter:**

- Driver has enough seats
- Rider's request time falls within driver's trip time window (CHECK ACTUAL TIME, not just date)
- Estimated detour does not exceed driver's max
- Pickup point comes before dropoff point along the route direction

**Detour calculation (fix the unit bug):**

```
detour_distance_m = distance(rider_pickup, nearest_route_point) + distance(rider_dropoff, nearest_route_point)
detour_time_minutes = detour_distance_m / avg_speed_mps * (1/60)
```

Use a reasonable average speed (e.g., 40 km/h urban, 80 km/h highway).

### 3E: Experiment Runner

- Run both strategies on the **same set of drivers and riders** (test set from Phase 2)
- Run **5 random seeds** (shuffle rider arrival order) for statistical robustness
- Record per-driver outcomes in a DataFrame
- Save to CSV: `results/coldstart_outcomes.csv`, `results/warmup_outcomes.csv`

**Deliverable:** Two CSV files with per-driver profit and metrics for each strategy.

---

## Phase 4: Results Visualization and Analysis

**Goal:** Produce publication-quality comparison graphs.

### 4A: Core Comparison Plots

1. **Profit comparison (bar chart with error bars):** Mean profit per driver, cold-start vs warm-up, with 95% CI from the 5 seeds
2. **Profit distribution (box plot):** Full distribution of per-driver profit for each strategy
3. **Cumulative profit (line chart):** Running total profit over N drivers, both strategies overlaid
4. **Match rate comparison (bar chart):** Percentage of drivers who matched at least 1 rider
5. **Revenue vs cost breakdown (stacked bar):** Show revenue and cost components side by side
6. **Compute time comparison (bar chart):** Average time to process one driver (warm-up includes ML inference + multi-route fetch; cold-start includes single route + corridor build)

### 4B: ML Model Insight Plots

1. **Route rank accuracy:** When warm-up picks rank-1, how often is it actually the most profitable? (Predicted rank vs actual rank confusion matrix)
2. **Feature importance:** Which features drive profit prediction?
3. **Predicted vs actual profit scatter**

### 4C: Statistical Summary

- Print a text table: mean, median, std, min, max profit for each strategy
- Paired t-test or Wilcoxon signed-rank test for statistical significance
- Save to `results/summary.txt`

**Deliverable:** A `results/` folder with all plots as PNG files and a text summary.

---

## Phase 5: Polish and Paper-Readiness

- Clean README with setup instructions, how to run each phase
- `requirements.txt` with pinned versions
- Single `run_all.py` script that executes phases 1-4 in sequence
- Ensure all OSRM calls are cached so the system can run offline after first pass
- Add a `--no-api` flag that uses straight-line polylines (for testing without OSRM server)

---

## File Structure

```
project/
  src/
    spatial/
      h3_utils.py          # H3 primitives (ONE file, no duplication)
      corridor.py           # Corridor building (single + multi-route)
      router.py             # OSRM routing with disk-backed cache
    matching/
      matcher.py            # Shared matching logic (spatial filter + feasibility)
    models/
      build_dataset.py      # Build profit-labeled training data
      train_profit_model.py # Train LightGBM profit predictor
      predict.py            # Load model + predict profit for a route
    simulation/
      coldstart.py          # Cold-start pipeline
      warmup.py             # Warm-up pipeline
      runner.py             # Experiment runner (multi-seed)
    data_prep/
      download.py           # Download NYC TLC data
      preprocess.py         # Clean + filter + extract revenue
  visualizations/
    plot_comparison.py      # All comparison plots
    plot_model.py           # ML model insight plots
  data/                     # Raw + processed data (gitignored)
  models/                   # Saved ML models (gitignored)
  results/                  # Output CSVs + plots + summary
  run_all.py
  requirements.txt
  README.md
  .env                      # OSRM_BASE_URL (optional, gitignored)
```

---

## Key Decisions Summary


| Decision           | Choice                                  | Rationale                                                |
| ------------------ | --------------------------------------- | -------------------------------------------------------- |
| Dataset            | NYC TLC yellow taxi (WITH fare columns) | Free, public, has revenue data, millions of trips        |
| Trip filter        | > 5 miles, $5-$200 fare                 | Approximate longer-distance trips, remove outliers       |
| H3 resolution      | 9 (edge ~174 m)                         | Tight corridor (~520 m wide) suited for dense NYC grid   |
| ML model           | LightGBM regressor                      | Fast inference, handles tabular data well, interpretable |
| ML target          | Expected profit in dollars              | Directly answers "which route is most profitable?"       |
| Train/test split   | Temporal (not random)                   | Avoids data leakage in time-series demand patterns       |
| Matching           | 2-stage (spatial + feasibility)         | Simpler than 3-door, same effectiveness                  |
| Experiment seeds   | 5 runs per strategy                     | Statistical robustness for error bars                    |
| Route alternatives | 3 for warm-up, 1 for cold-start         | Core experimental variable                               |


