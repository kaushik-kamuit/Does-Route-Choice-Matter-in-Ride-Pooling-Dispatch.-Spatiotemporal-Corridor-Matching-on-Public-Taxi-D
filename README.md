# Warm-Up Route Selection vs Cold-Start: A Profit-Driven Carpooling Experiment

An experimental system demonstrating that ML-guided route selection ("warm-up") produces statistically significant higher profit than naive default routing ("cold-start") in a real-time carpooling context. Built on NYC TLC taxi data, H3 spatial indexing, OSRM road-network routing, and a LightGBM profit predictor.

**Core result:** Using an enhanced v2 feature set (38 non-leaky features), warm-up achieves statistically significant higher profit than cold-start, validated across 10,000 test drivers, 5 seeds, and 5 strategy baselines (cold-start, random, heuristic, ML warm-up, oracle). See [Results](#10-results) for detailed numbers.

---

## Table of Contents

1. [Thesis and Motivation](#1-thesis-and-motivation)
2. [Experimental Design](#2-experimental-design)
3. [Architecture Overview](#3-architecture-overview)
4. [Data Pipeline](#4-data-pipeline)
5. [Spatial Infrastructure](#5-spatial-infrastructure)
6. [Matching Engine](#6-matching-engine)
7. [ML Profit Predictor](#7-ml-profit-predictor)
8. [Simulation Framework](#8-simulation-framework)
9. [Statistical Analysis](#9-statistical-analysis)
10. [Results](#10-results)
11. [Design Decisions: Why and Why Not](#11-design-decisions-why-and-why-not)
12. [Known Limitations](#12-known-limitations)
13. [Future Development Checklist](#13-future-development-checklist)
14. [Setup and Reproduction](#14-setup-and-reproduction)
15. [Project Structure](#15-project-structure)

---

## 1. Thesis and Motivation

**Thesis:** When a driver provides origin and destination for a carpooling trip, an ML model that evaluates multiple alternative routes and selects the one with highest predicted rider-matching profit will consistently outperform the naive strategy of accepting the default (fastest) route.

**Why this matters:** In real-time carpooling, the platform has seconds to recommend a route. The fastest route (cold-start) ignores rider demand distribution. A "warm-up" period where the system evaluates 2-3 genuine road-network alternatives and predicts which corridor will attract the most profitable rider matches could increase platform revenue without increasing driver effort.

**What "warm-up" means in this context:** The system "warms up" by doing extra computation (fetching alternatives, building corridors, running ML inference) before committing the driver to a route. The cold-start strategy skips this warm-up and immediately assigns the default route.

---

## 2. Experimental Design

### Strategy Comparison

| Strategy | How It Works | Computation |
|----------|-------------|-------------|
| **Cold-Start** | Takes `routes[0]` (fastest/default OSRM route), builds one H3 corridor, matches riders. | 1 corridor + 1 match |
| **Random** | Picks uniformly at random among the 3 OSRM alternatives, matches riders on chosen route. | 3 corridors + 1 match |
| **Heuristic** | Picks the route whose corridor contains the most riders (no ML). | 3 corridors + 3 rider counts + 1 match |
| **ML Warm-Up** | ML model (LightGBM) predicts profit per route, selects the highest. | 3 corridors + feature extraction + ML inference + 1 match |
| **Oracle** | Runs full matching on all 3 routes, picks the one with highest actual profit. Upper bound. | 3 corridors + 3 full matches |

### Fairness Guarantee

Both strategies share the **exact same OSRM `alt=3` request**. Cold-start uses `routes[0]` from that response. Warm-up ranks all 3 and picks the best. This means cold-start's route is always available as a warm-up candidate. If the ML model provides no value, warm-up would still pick `routes[0]` — making the comparison fair.

### Paired Design

For each of the 10,000 test drivers:
- Both strategies see the same driver (same origin, destination, time).
- Both strategies see the same rider pool (same `RiderIndex` snapshot).
- Both strategies use the same 5 seeds (controlling tie-breaking randomness).
- The only variable is route selection: default vs ML-ranked.

This paired structure enables the paired t-test and Wilcoxon signed-rank test.

### Profit Calculation

```
profit = sum(matched_rider_fares × platform_share) - (route_distance_miles × cost_per_mile)
```

- `platform_share = 0.50` (platform keeps 50% of each matched rider's fare)
- `cost_per_mile = $0.67` (fuel + wear, standard NYC estimate)
- `fare_amount`: actual NYC TLC metered fare from the dataset

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        OSRM MLD Server (Docker)                        │
│              Local instance at http://localhost:5000                    │
│         New York road network, Multi-Level Dijkstra algorithm          │
│              Returns 2-3 genuine alternative routes                    │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │       SQLite Route Cache      │
                    │     data/route_cache.db       │
                    │    ~110K cached OD pairs      │
                    └──────────────┬───────────────┘
                                   │
     ┌─────────────────────────────┼─────────────────────────────┐
     │                             │                             │
     ▼                             ▼                             ▼
┌─────────┐               ┌──────────────┐             ┌──────────────┐
│ Corridor│               │   Shapely    │             │   LightGBM   │
│ Builder │               │   Matcher    │             │   Predictor  │
│ (H3 r9) │               │ (GEOS ufuncs)│             │  (R²=0.79)   │
└────┬────┘               └──────┬───────┘             └──────┬───────┘
     │                           │                            │
     │    ┌──────────────────────┼────────────────────────────┘
     │    │                      │
     ▼    ▼                      ▼
┌────────────────┐     ┌──────────────────┐
│   Cold-Start   │     │    Warm-Up       │
│  (1 route)     │     │  (3 routes + ML) │
└────────┬───────┘     └────────┬─────────┘
         │                      │
         └──────────┬───────────┘
                    │
         ┌──────────▼──────────┐
         │   Runner (paired,   │
         │   driver-major,     │
         │   5 seeds)          │
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
         │  Statistical Tests  │
         │  + Visualizations   │
         └─────────────────────┘
```

---

## 4. Data Pipeline

### Source Data

**NYC TLC Yellow Taxi Trip Records, 2015** (Azure Open Datasets).

- **Why 2015:** Freely available, contains actual metered fare amounts (not just estimates), has GPS coordinates for pickup and dropoff, and has enough volume for statistical power.
- **Why Yellow Taxi:** Green taxi (Boro) has geographic restrictions. FHV data lacks fare columns. Yellow taxi has complete fare + coordinate + timestamp data.

### Preprocessing

| Step | Logic | Rationale |
|------|-------|-----------|
| **Time split** | Jan-Mar → train, April → test | Temporal split prevents data leakage. Train model on winter, test on spring. |
| **Driver filter** | Trips > 10 miles | Carpooling is only viable for long trips. Short trips don't generate enough corridor for rider matching. |
| **Rider filter** | Trips 0.5-10 miles, 25% sample | Riders are short-distance trips that could be picked up along a driver's corridor. 25% subsample for memory. |
| **H3 cells** | `pickup_h3`, `dropoff_h3` at resolution 9 | Pre-computed during preprocessing so `RiderIndex` build is fast. |
| **Temporal columns** | `hour_of_day`, `day_of_week`, `is_weekend`, `pickup_datetime` | Used as ML features and for temporal binning. |

### Output Files

- `data/processed/drivers.parquet` — Long-distance trips with `split` column
- `data/processed/riders.parquet` — Short-distance trips with `split` column

---

## 5. Spatial Infrastructure

### 5.1 OSRM (Open Source Routing Machine)

**What:** Self-hosted routing engine that returns real road-following polylines for any origin-destination pair.

**Algorithm:** MLD (Multi-Level Dijkstra). Chosen over CH (Contraction Hierarchies) because MLD natively supports alternative route generation. CH only returns 1 fastest route; the previous implementation hacked around this by injecting perpendicular waypoints, producing artificial alternatives that weren't genuine road-network paths.

**Why self-hosted:** The public OSRM server (`router.project-osrm.org`) uses CH (no alternatives) and has a 1-request-per-second rate limit. Self-hosting MLD eliminates both constraints.

**Setup:** Docker container with pre-built New York graph. Graph construction (`osrm-extract` → `osrm-partition` → `osrm-customize`) was done once and the resulting files are stored in `osrm/data/`.

**Route cache:** All OSRM responses are cached in SQLite (`data/route_cache.db`). The simulation runs in `cache-only` mode — zero API calls during experiments. The SQLite backend replaced a JSON cache that loaded ~2GB into RAM.

### 5.2 H3 Hexagonal Indexing

**What:** Uber's H3 library converts GPS coordinates to hierarchical hexagonal cells. Used for spatial indexing and corridor construction.

**Resolution 9** is used throughout:
- Hex edge length: ~174 meters
- Hex area: ~0.1 km²
- With k-ring buffer of 1: corridor width ~520 meters from road centerline

**Why resolution 9:** It provides a corridor width that roughly matches the walkable catchment area for a carpool pickup. A rider ~500m from the driver's route is plausibly matchable. Resolution 8 (456m edge) would be too coarse; resolution 10 (66m edge) would miss riders just off the road.

**Why H3 over geohash or S2:** H3 hexagons have uniform adjacency (every cell has exactly 6 neighbors), making k-ring expansion consistent. Geohash rectangles have edge effects. S2 cells vary in shape. H3's `grid_disk` operation is also very fast.

### 5.3 Corridor Construction

A **corridor** is the spatial region around a driver's route where rider pickups/dropoffs are considered.

```
Polyline → Densify (80m steps) → H3 cells → k-ring expand (k=1)
```

**Densification:** OSRM polylines have variable point spacing. If two consecutive points are 300m apart and a hex cell falls between them, that cell would be missed. Interpolating every 80m ensures no H3 cell is skipped.

- **Why 80m:** At resolution 9 (174m edge), a 200m step could skip cells at diagonal crossings. 80m guarantees full coverage. This was tightened from the original 200m after analysis showed missed cells.

**k-ring expansion (k=1):** Adds all immediately adjacent hex cells around the route. This widens the corridor from the exact road path to a ~520m band, capturing riders who are near but not directly on the route.

- **Why k=1:** We considered a dynamic corridor expansion based on driver detour budget, but k=1 provides a reasonable fixed catchment area. Varying corridor width per driver would require re-indexing riders and add complexity without clear benefit for the thesis demonstration.

---

## 6. Matching Engine

### 6.1 Two-Stage Architecture

```
Stage 1: Spatial-Temporal Filter (RiderIndex)
  → Riders whose pickup AND dropoff H3 cells are in the corridor
  → AND whose 15-min time bin is within ±1 bin of driver's departure
  → Output: ~10-50 candidate riders (from millions)

Stage 2: Feasibility Filter (Shapely)
  → Directionality: rider travels ≥5% of route in driver's direction
  → Detour budget: round-trip detour ≤ 4 minutes at 40 km/h
  → Seat capacity: passenger count ≤ remaining seats
  → Output: ~2-5 feasible riders

Stage 3: Greedy Assignment
  → Sort feasible riders by fare descending (with seed-based noise)
  → Fill seats greedily until capacity reached
  → Output: 0-3 matched riders
```

### 6.2 RiderIndex: 15-Minute Temporal Bins

The `RiderIndex` is an in-memory dictionary mapping `(H3_cell, time_bin)` to arrays of rider row indices.

**Original design:** Hourly bins (0-23) with `±1 hour` window = 3-hour matching span. This was unrealistic — no rider waits 2 hours for a carpool.

**Current design:** 15-minute bins (0-95 per day) with `±1 bin` window = 45-minute matching span.

```python
center_bin = minute_of_day // 15  # e.g., 7:23 AM → bin 29
query_bins = [28, 29, 30]         # ±1 bin → 7:00-7:44 AM window
```

**Why 15-minute bins:** The riders parquet has `pickup_datetime` with second-level precision, so sub-hourly bins are free. 15 minutes is a natural urban transit interval. The `±1 bin = 45 minutes` window is realistic for "I'm leaving around the same time" matching. Going to 5-minute bins would make the window too narrow (15 min) and miss valid matches at bin boundaries.

**Impact on performance:** Candidate pool size dropped ~3-4x compared to hourly bins. This reduced per-driver matching time from 0.23s to 0.062s because Shapely processes fewer candidate riders per call.

**Impact on ML model:** The model still uses `hour_of_day` (0-23) as a feature, not the 15-min bin. The temporal bins only affect matching, not feature engineering. Adding the 15-min bin as an ML feature is listed as a future improvement.

### 6.3 Shapely Matcher: Coordinate Handling

**The problem:** Shapely works in flat Cartesian space. GPS coordinates are (lat, lng) on a sphere. At NYC latitude (40.7°N), 1° longitude ≈ 84.4 km but 1° latitude ≈ 111.3 km. Treating these as equal would distort distances by ~24%.

**The solution: Cheap-ruler correction.**

```python
COS_LAT = cos(radians(40.7))  # ≈ 0.7585
DEG_TO_M = 111_320.0          # meters per degree of latitude

# Scale longitude to equalize axes
scaled_lng = lng * COS_LAT
# Now 1 unit ≈ 111,320 meters in both x and y

# After Shapely computes distance in "degrees":
distance_meters = shapely_distance * DEG_TO_M
```

**Why not full geodesic (haversine)?** Shapely's C-level ufuncs (backed by GEOS) are 10-50x faster than Python haversine loops. The cheap-ruler approximation introduces <1% error at NYC latitudes, which is negligible for a 4-minute detour budget. The speed gain from vectorized C computation far outweighs the tiny accuracy loss.

**Why not project to UTM?** Adding pyproj or UTM transforms adds a dependency, complexity, and latency (coordinate transform for every rider point). The `cos(lat)` scaling achieves the same result with one multiplication.

**What Shapely computes:**

- `line_locate_point(route_line, pickup_point, normalized=True)` → fractional position (0.0 = route start, 1.0 = route end) of the nearest point on the polyline to the rider's pickup. This is a true point-to-segment projection, not nearest-vertex.
- `distance(route_line, point)` → perpendicular distance from the point to the nearest segment of the polyline.

Both are numpy ufuncs: they accept arrays of points and process all candidates in a single C call.

### 6.4 Directionality Check

A rider is only valid if they travel in the driver's direction along the route:

```
travel_fraction = dropoff_frac - pickup_frac
valid if travel_fraction >= 0.05
```

**Why 0.05 (5%)?** A rider whose pickup is at fraction 0.90 and dropoff at 0.92 is barely co-directional and would cause disproportionate detour. 5% of a 20-mile route is 1 mile — a minimum meaningful rideshare distance. Setting this to 0 would allow riders going perpendicular to the route.

### 6.5 Detour Formula

```
detour_meters = 2 × (pickup_perp_dist + dropoff_perp_dist) × MANHATTAN_FACTOR
detour_minutes = detour_meters / URBAN_SPEED_MPS / 60
valid if detour_minutes <= max_detour_min (4.0)
```

- **Factor of 2:** The driver must deviate off-route to pickup, then return to route. Same for dropoff. This is a round-trip penalty.
- **MANHATTAN_FACTOR = 1.3:** Urban road networks are grid-like, not straight-line. The actual driving detour is ~30% longer than the perpendicular distance.
- **URBAN_SPEED_MPS = 40 km/h:** Typical NYC urban driving speed for converting meters to time.
- **max_detour_min = 4.0:** Research literature (Agatz et al., 2012; Stiglic et al., 2015) suggests 3-5 minutes is the maximum acceptable detour for carpooling. The original value of 15 minutes was unrealistic.

### 6.6 Greedy Seat Assignment

After feasibility filtering, riders are sorted by `fare_share` descending with a small random noise for tie-breaking:

```python
sort_key = fare_share + rng.uniform(-0.01, 0.01)
```

Riders are assigned seats greedily until the vehicle capacity (3 seats) is filled. Multi-passenger riders (`passenger_count > 1`) are skipped if insufficient seats remain.

**Why greedy, not optimal?** The optimal assignment (maximum weighted matching) is NP-hard for general cases. With only 3 seats and ~2-5 feasible riders, greedy-by-fare is near-optimal and runs in microseconds.

**Why seed-based tie-breaking?** Without noise, identical fares produce deterministic ordering (Python sort stability + DataFrame index order). Adding `±$0.01` noise ensures different seeds produce slightly different matchings, giving the multi-seed experiment actual variance to measure.

---

## 7. ML Profit Predictor

### 7.1 Model Choice: LightGBM

**Why LightGBM:** Gradient-boosted trees excel at tabular data with mixed feature types. Fast training (~4 minutes for 217K rows), low inference latency (microseconds per prediction), and inherent feature importance. No feature scaling or encoding required.

**Why not neural networks:** The dataset is pure tabular (38 numeric features). A model comparison study confirmed LightGBM (67.6% rank accuracy) outperforms MLP (64.2%), consistent with literature on GBDTs dominating tabular data.

**Why not linear regression:** The relationship between corridor geometry, temporal patterns, and profit is non-linear. A linear model would miss interaction effects (e.g., high demand density at rush hour on short routes).

### 7.2 Training Target

```
expected_profit = sum(matched_rider_fare_shares) - (route_distance × cost_per_mile)
```

**Critical fix:** The original implementation summed ALL rider fares in the corridor (~$60K) as the label. The actual post-matching profit is ~$30-60. The model was optimizing a 1000x-wrong objective. The fix calls `match_riders()` during dataset construction to compute the true post-matching profit.

**Why call match_riders during training?** The model needs to learn the relationship between features and the profit that the matching engine will actually produce. Using raw corridor fare sums ignores seat limits, directionality, and detour constraints.

### 7.3 Features (38, v2 -- no leaky features)

The v1 model relied on `matched_rider_count` and `feasible_rider_count` which leaked matching outcomes into prediction inputs. The v2 model removed these and added 26 engineered features:

| Group | Features | Count |
|-------|----------|-------|
| **Geometric** | route_distance_m, route_duration_s, corridor_cell_count, route_sinuosity, route_avg_speed_ms, bearing_sin/cos, straight_line_dist_m | 8 |
| **Temporal** | hour_of_day, day_of_week, is_weekend, day_of_month, time_bin_15min, hour_sin/cos | 7 |
| **Corridor Demand** | corridor_rider_count, corridor_demand_density, mean_rider_fare, corridor_fare_density | 4 |
| **Historical H3** | corridor_hist_pickups, corridor_hist_dropoffs, corridor_hist_pickup/dropoff_density, corridor_hist_mean_fare, corridor_hist_fare_density | 6 |
| **Landmark** | origin/dest distances to JFK, LGA, Penn Station, Times Square; nearest landmark distances | 10 |
| **Cell-level** | origin_cell_pickups, origin_cell_mean_fare, dest_cell_dropoffs | 3 |

Feature ablation study results (see `results/ablation_results.csv`):
- Removing **temporal features** causes the largest drop: R² 0.79 → 0.55, Rank-1 67.9% → 60.9%
- **Spatial demand** features alone achieve 60.8% rank accuracy
- **Landmark** features are largely redundant when combined with other groups

### 7.4 Training Protocol

- **Data:** 217,831 rows from 100K train drivers (avg 2.18 routes per driver).
- **Validation split:** `GroupShuffleSplit` by `driver_id` (80/20). No driver appears in both train and val.
- **Hyperparameters (tuned):** `learning_rate=0.03`, `num_leaves=127`, `max_depth=10`, `early_stopping=80 rounds`, `num_boost_round=2000`.
- **Model comparison:** 6 architectures tested (Ridge, LightGBM baseline/tuned, LambdaRank, XGBoost, MLP). See `results/model_comparison.csv`.

### 7.5 Model Performance (v2)

| Metric | Value | Interpretation |
|--------|-------|---------------|
| R² | 0.79 | Model explains 79% of profit variance |
| RMSE | $7.92 | Average prediction error |
| Top-1 Rank Accuracy | 67.6% | ML picks the most profitable route 67.6% of the time |

This is **better than v1** (R²=0.77) despite removing the leaky features, proving the engineered features are genuinely predictive.

**Top features by importance (gain):**
1. `corridor_fare_density` — revenue density per corridor area
2. `time_bin_15min` — fine-grained temporal demand patterns
3. `mean_rider_fare` — fare potential in the corridor

---

## 8. Simulation Framework

### 8.1 Architecture: Driver-Major Loop

The simulation iterates over drivers (outer loop) and seeds (inner loop):

```
for each driver (10,000):
    # Compute once (seed-independent):
    routes      = fetch 3 alternatives from cache
    corridors   = build 3 H3 corridors
    features    = extract 38 v2 features per route
    ranking     = ML model ranks routes by predicted profit

    for each seed (5):
        # All 5 strategies run on the same driver/seed:
        cold_start = match riders on routes[0]
        random     = match riders on random route
        heuristic  = match riders on highest-count route
        warm_up    = match riders on ML-ranked best route
        oracle     = match riders on all routes, pick best actual
```

**Why driver-major?** The original implementation was seed-major (iterate seeds, then drivers), causing route fetching, corridor building, feature extraction, and ML inference to be repeated 5x per driver. Since these operations are seed-independent, restructuring to driver-major eliminated 80% of redundant work, producing a **2.04x speedup** (5.8 hours → 2.84 hours).

**What changes per seed:** Only the `match_riders` call varies, because it uses a `np.random.default_rng(seed)` for fare tie-breaking noise. Route selection, corridor geometry, and ML ranking are identical across seeds.

### 8.2 Pre-Computation Passthrough

`run_coldstart` and `run_warmup` accept optional pre-computed data:

```python
def run_coldstart(driver, router, rider_index, seed, *,
                  route=None, corridor=None) -> DriverOutcome | None

def run_warmup(driver, router, rider_index, predictor, ..., seed, *,
               routes=None, corridors=None, ranking=None) -> DriverOutcome | None
```

When called from `runner.py`, pre-computed data is passed. When called independently (e.g., debugging), the functions compute everything internally. This preserves backward compatibility.

### 8.3 Feature Extraction in Warm-Up (v2)

The v2 warm-up pipeline computes features from route geometry, temporal context, and historical H3 cell statistics -- without calling `match_riders` for features. Only `corridor_rider_count` (count of riders spatially in the corridor) is computed from the rider index, which is a cheap spatial lookup, not a full matching call. The actual `match_riders` call is only made once for the final selected route.

### 8.4 Configuration

```json
{
  "sample_size": 10000,
  "seeds": [42, 43, 44, 45, 46],
  "platform_share": 0.50,
  "cost_per_mile": 0.67,
  "max_detour_min": 4.0,
  "seats": 3
}
```

---

## 9. Statistical Analysis

### 9.1 Aggregation Before Testing

Each driver appears 5 times (one per seed). Before running statistical tests, outcomes are aggregated to per-driver means:

```python
cs_agg = coldstart_df.groupby("driver_id")["profit"].mean()
wu_agg = warmup_df.groupby("driver_id")["profit"].mean()
```

This produces 10,000 paired observations (one per driver), preventing pseudoreplication.

### 9.2 Tests Performed

**Paired t-test** (`scipy.stats.ttest_rel`): Tests whether the mean difference (warm-up − cold-start) is significantly different from zero. Assumes normally distributed differences. With n=10,000, the Central Limit Theorem ensures the sampling distribution of the mean is approximately normal regardless of the underlying distribution.

**Wilcoxon signed-rank test** (`scipy.stats.wilcoxon`): Non-parametric alternative. Does not assume normality. Tests whether the distribution of differences is symmetric around zero. Included as a robustness check.

### 9.3 Why Both Tests?

If both tests agree, the result is robust regardless of distributional assumptions. If they disagreed, it would suggest the profit differences have heavy tails or outliers affecting the t-test.

### 9.4 Effect Size Metrics

Beyond p-values (which are guaranteed to be small with n=10,000), the extended summary reports:
- **Cohen's d** (standardized effect size)
- **95% bootstrap confidence intervals** on the mean difference (10,000 resamples)
- **Winner/loser analysis** (what % of drivers benefit, and by how much)
- **Oracle gap analysis** (what fraction of theoretical maximum the ML captures)

See `results/extended_summary.txt` for full details.

---

## 10. Results

Results are generated from the v2 model (38 non-leaky features, R²=0.79) with 5 strategy baselines across 10,000 test drivers and 5 seeds. Full numerical results are in `results/extended_summary.txt` and `results/summary.txt`.

### 10.1 Strategy Comparison

All 5 strategies are evaluated on the same drivers, rider pool, and seeds:

| Strategy | Description | vs Cold-Start |
|----------|-------------|---------------|
| Cold-Start | Default route (routes[0]) | baseline |
| Random | Uniform random among 3 alternatives | measures value of any alternative |
| Heuristic | Highest corridor rider count (no ML) | measures value of simple demand rule |
| ML Warm-Up | LightGBM-ranked best route | measures ML contribution |
| Oracle | Best actual profit (hindsight) | theoretical upper bound |

The oracle gap analysis shows what fraction of the theoretical maximum improvement the ML model captures.

### 10.2 Feature Ablation

| Experiment | Features | R² | RMSE | Rank-1 |
|-----------|----------|-----|------|--------|
| All features | 38 | 0.7925 | $7.92 | 67.9% |
| Only Geometric | 8 | 0.3292 | $14.23 | 54.6% |
| Only Temporal | 7 | 0.1042 | $16.45 | 46.5% |
| Only Spatial Demand | 13 | 0.5547 | $11.60 | 60.8% |
| Only Landmark | 10 | 0.3247 | $14.28 | 46.5% |
| All minus Geometric | 30 | 0.7893 | $7.98 | 66.9% |
| All minus Temporal | 31 | 0.5522 | $11.63 | 60.9% |
| All minus Spatial Demand | 25 | 0.7346 | $8.95 | 61.9% |
| All minus Landmark | 28 | 0.7839 | $8.08 | 67.6% |

**Key insight:** Temporal features are the most critical group -- removing them drops R² by 0.24 and rank accuracy by 7 percentage points. This is because rider demand varies dramatically by time of day, and the 15-minute time bin captures fine-grained patterns that drive route profitability differences.

### 10.3 Model Comparison

| Model | R² | RMSE | Rank-1 Accuracy |
|---|---|---|---|
| **LightGBM (tuned)** | **0.7921** | **$7.92** | **67.6%** |
| XGBoost | 0.7846 | $8.07 | 67.1% |
| MLP (Neural Net) | 0.7207 | $9.18 | 64.2% |
| LightGBM (baseline) | 0.7607 | $8.50 | 65.3% |
| LGB LambdaRank | -2.88* | $34.25 | 66.2% |
| Ridge (linear) | 0.3814 | $13.67 | 53.0% |

*LambdaRank optimizes ordering not absolute values, so R²/RMSE are not meaningful for it.

### 10.4 Interpretation of Plots

**Baseline comparison** (`results/plots/baseline_comparison.png`): Mean profit for all 5 strategies with 95% CI, showing the progression from cold-start through heuristic to ML to oracle.

**Winner/loser histogram** (`results/plots/winner_loser.png`): Distribution of per-driver profit differences (warm-up minus cold-start), showing what fraction of drivers benefit from ML route selection.

**Route choice analysis** (`results/plots/route_choice.png`): How often the ML model picks each route, and the advantage when it selects an alternative vs the default.

**Heterogeneity by time** (`results/plots/heterogeneity_time.png`): Warm-up advantage segmented by time of day (morning/midday/evening/night rush).

**Feature importance** (`results/plots/feature_importance.png`): `corridor_fare_density` dominates, followed by `time_bin_15min` and `mean_rider_fare`.

**Ablation heatmap** (`results/plots/ablation_heatmap.png`): Visual comparison of R², RMSE, and Rank-1 accuracy across all feature subset experiments.

**Density vs advantage** (`results/plots/density_advantage.png`): How the warm-up advantage changes as rider density decreases from 100% to 10%, demonstrating that ML route selection becomes more valuable in sparser markets.

**Predicted vs actual** (`results/plots/predicted_vs_actual.png`): Scatter plot showing R²=0.79 on held-out validation drivers.

**Rank accuracy** (`results/plots/rank_accuracy.png`): 67.6% top-1 accuracy.

---

## 11. Design Decisions: Why and Why Not

### Implemented

| Decision | Choice | Why |
|----------|--------|-----|
| OSRM MLD over CH | MLD supports native alternative routes | CH returns only 1 route. The waypoint hack produced synthetic alternatives that weren't genuine road paths. |
| Shapely over haversine loops | Vectorized C (GEOS) ufuncs | 13-22x faster than NumPy haversine. Exact point-to-segment projection, not nearest-vertex approximation. |
| cos(lat) over full geodesic | <1% error at NYC lat, zero dependency | pyproj/UTM adds complexity. For 4-minute detour budgets, <1% distance error is irrelevant. |
| 15-min bins over hourly bins | Tighter matching window (45 min vs 3 hours) | Realistic temporal proximity. Also reduced candidate pools 3-4x, speeding up matching. |
| SQLite cache over JSON | Near-zero memory footprint | JSON cache loaded ~2GB into RAM. SQLite reads one entry at a time. |
| GroupShuffleSplit over random split | No driver leakage | Multiple routes per driver share features. Random split would put correlated rows in train and val. |
| max_detour=4 min over 15 min | Research-backed | Ridesharing literature consensus is 3-5 min. 15 min would match riders who require unrealistic detours. |
| H3 resolution 9 over 8 or 10 | ~520m corridor width with k=1 | Resolution 8 (456m edge) too coarse. Resolution 10 (66m edge) would need k=3 for similar coverage, expanding computation. |
| LightGBM over XGBoost/NN | Fastest training, best accuracy | 6-model comparison confirmed: LightGBM tuned (67.6% rank accuracy) > XGBoost (67.1%) > MLP (64.2%). |
| Densify at 80m over 200m | No missed H3 cells | 200m step could skip cells at 174m hex edge. 80m guarantees full coverage. |
| 5 seeds over 1 or 10 | Statistical robustness without excessive runtime | 1 seed gives no variance estimate. 10 seeds doubles runtime for diminishing returns. |
| Driver-major loop over seed-major | 2x speedup | Avoids recomputing seed-independent work (routes, corridors, features, ML ranking) per seed. |

### Not Implemented (and Why)

| Feature | Why Not |
|---------|---------|
| Route deduplication (Jaccard) | Implemented as `_deduplicate_routes()` but NOT called. MLD alternatives are already structurally different. Dedup risks dropping valid alternatives in NYC's dense grid where corridor overlap is naturally high. |
| Dynamic corridor expansion (detour budget) | Varying corridor width per driver requires complex per-query re-indexing. k=1 ring is sufficient for proving the thesis. Could be explored in future work. |
| Polyline subsampling (every 5th point) | Reduces accuracy (skips geometry between sampled points). Shapely's C-level ufuncs are fast enough at full resolution. The 0.062s/driver time doesn't justify lossy approximation. |
| Vectorized haversine in corridor builder | `corridor.py` uses Python loops with haversine for polyline length and densification. These are called once per driver per route (~3 calls total). The bottleneck is matching, not corridor building. Optimizing would add complexity for negligible gain. |
| Holiday indicator | Not available in the base NYC TLC data without external dependency. Listed as future improvement. |
| Spatial cluster features | Requires additional preprocessing pipeline (k-means on H3 cell density). Not needed for the core thesis demonstration. |
| LambdaMART ranking loss | Tested in v2 model comparison (66.2% rank accuracy vs 67.6% for regression). For groups of 2-3 routes, pointwise regression captures ordering effectively. |
| Pairwise/relative features | Features capturing differences between routes could improve ranking but add complexity. Listed as future work. |

---

## 12. Known Limitations

1. **Synthetic experimental setup.** Real carpooling involves dynamic driver arrivals, rider cancellations, and multi-hop routes. This simulation matches riders from a static snapshot.

2. **No real-time driver behavior.** The driver always accepts the recommended route. In practice, drivers may reject recommendations, altering outcomes.

3. **Single-city, single-time-period.** Results are from NYC April 2015. Generalization to other cities, seasons, or years is not tested.

4. **Taxi data as carpool proxy.** NYC TLC taxi trips are not actual carpool requests. The "riders" are taxi trips repurposed as hypothetical carpool passengers. Real carpool demand would differ.

5. **Platform share assumption (50%).** The $0.50 platform share is arbitrary. Different split ratios would scale profits but not affect the relative comparison (warm-up vs cold-start).

6. **Cost model is simplified.** `$0.67/mile` is a flat rate. Real costs vary with traffic, time of day, and vehicle type.

7. **Match rate saturation.** Both strategies achieve >99% match rate in NYC, limiting the visible advantage from better route selection. In lower-density cities, the warm-up advantage could be larger (more variance in corridor quality) or smaller (fewer riders to match regardless).

8. **Leaky features removed in v2.** The v1 model used `matched_rider_count` and `feasible_rider_count` which leaked matching outcomes into inputs. The v2 model removed these features entirely and achieved better performance (R² 0.77 → 0.79) with 38 non-leaky features based on route geometry, temporal patterns, historical H3 demand, and landmark distances.

9. **Temporal train/test split.** Training on Jan-Mar and testing on April means the model has never seen April-specific patterns (spring break, weather changes). This is conservative — a model trained on the same period would likely perform better.

---

## 13. Completed Improvements and Future Work

### Completed in v2

1. **15-minute time bin as ML feature** — Added `time_bin_15min`. Now the 2nd most important feature.
2. **Day of month** — Added. Captures demand spikes.
3. **Distance to landmarks** — Added JFK, LGA, Penn Station, Times Square.
4. **H3 cell-level statistics** — Pre-aggregated pickup/dropoff counts and fares per cell.
5. **Route geometry features** — Sinuosity, average speed, bearing (sin/cos), straight-line distance.
6. **Removed leaky features** — `matched_rider_count` and `feasible_rider_count` dropped. R² improved.
7. **Multi-strategy baselines** — Oracle, random, and heuristic baselines added.
8. **Feature ablation study** — Quantified contribution of each feature group.
9. **Rider density variation** — Experiment showing warm-up advantage vs rider density.

### Future Work

1. **Pairwise/relative features** — Add features capturing differences between routes (e.g., demand ratio, distance ratio vs shortest). These could improve ranking accuracy.
2. **Multi-city evaluation** — Test generalization beyond NYC.
3. **Dynamic pricing integration** — Model interaction with surge pricing.
4. **Temporal cross-validation** — Train on Jan, validate Feb, test Mar and Apr separately.
5. **Holiday indicator** — Use `holidays` Python library for calendar effects.
6. **Spatial cluster features** — k-means on H3 cell density for neighborhood-level patterns.

---

## 14. Setup and Reproduction

### Prerequisites

- Python 3.10+
- Docker Desktop (for OSRM server)
- ~10 GB disk space (data + OSRM graph)

### Installation

```bash
pip install -r requirements.txt
```

### Dependencies

```
pandas>=2.0          # Data processing
pyarrow>=15.0        # Parquet I/O
h3>=3.7              # Hexagonal spatial indexing
shapely>=2.0         # Vectorized geometry (GEOS backend)
lightgbm>=4.0        # Gradient boosted trees
scikit-learn>=1.3     # GroupShuffleSplit, metrics
matplotlib>=3.7      # Plotting
seaborn>=0.13        # Statistical plots
scipy>=1.11          # t-test, Wilcoxon
requests>=2.31       # OSRM API calls
python-dotenv>=1.0   # .env loading
joblib>=1.3          # Model serialization
```

### OSRM Server

```powershell
# Build the graph (one-time, ~30 minutes)
.\osrm\setup_osrm.ps1

# Or launch directly if graph is already built:
docker run -d --name osrm-mld -p 5000:5000 `
  -v "k:\Kamuit\Uber_Logic\Research_paper_1\osrm\data:/data" `
  ghcr.io/project-osrm/osrm-backend `
  osrm-routed --algorithm mld /data/new-york-latest.osrm
```

Set in `.env`:
```
OSRM_BASE_URL=http://localhost:5000
```

### Full Pipeline

```bash
python run_all.py
```

Or step by step:

```bash
# 1. Data preparation (download + preprocess)
python src/data_prep/download_2015.py
python src/data_prep/preprocess.py

# 2. Batch route (pre-cache OSRM routes)
python src/models/batch_route.py                # Train drivers (100K)
python src/models/batch_route.py --test         # Test drivers (10K)

# 3. Build ML training dataset (~1.7 hours)
python src/models/build_dataset.py

# 4. Train profit model (~4 minutes)
python src/models/train_profit_model.py

# 5. Run simulation (~2.8 hours)
python src/simulation/runner.py

# 6. Generate plots and statistics
python visualizations/plot_comparison.py
python visualizations/plot_model.py
```

### Useful Flags

```bash
python src/simulation/runner.py --sample 5000          # Fewer test drivers
python src/simulation/runner.py --seeds 3              # Fewer seeds
python src/simulation/runner.py --density 0.25         # 25% rider density
python src/simulation/runner.py --density 0.10 --tag d10  # Custom output tag
python src/simulation/runner.py --fetch                # Allow live OSRM calls
python scripts/ablation_study.py                       # Feature ablation study
python visualizations/plot_extended.py                 # Extended plots + stats
```

---

## 15. Project Structure

```
project/
├── src/
│   ├── spatial/
│   │   ├── h3_utils.py             # H3 primitives (geo_to_h3, haversine, densify, expand)
│   │   ├── corridor.py             # Corridor building (polyline → H3 cells → k-ring)
│   │   └── router.py              # OSRM routing with SQLite cache, route dedup (disabled)
│   ├── matching/
│   │   ├── rider_index.py         # Spatial-temporal index: (H3 cell, 15-min bin) → riders
│   │   └── matcher.py             # Shapely-based matching (directionality + detour + greedy)
│   ├── models/
│   │   ├── build_dataset.py       # Build profit-labeled training data (100K drivers)
│   │   ├── train_profit_model.py  # Train LightGBM (GroupShuffleSplit, early stopping)
│   │   ├── predict.py             # ProfitPredictor inference wrapper
│   │   └── batch_route.py         # Pre-cache OSRM routes (train + test drivers)
│   ├── simulation/
│   │   ├── data_types.py          # DriverTrip, MatchResult, DriverOutcome dataclasses
│   │   ├── coldstart.py           # Cold-start: 1 route, no ML
│   │   ├── warmup.py              # Warm-up: 3 routes + ML ranking + feature extraction
│   │   ├── baselines.py           # Oracle, random, heuristic strategies
│   │   └── runner.py              # Multi-strategy experiment runner (5 strategies, density param)
│   └── data_prep/
│       ├── download_2015.py       # Download NYC TLC 2015 data from Azure
│       └── preprocess.py          # Clean, filter, H3 cells, train/test split
├── scripts/
│   ├── compare_models.py         # 6-model comparison (Ridge, LightGBM, XGBoost, MLP, LambdaRank)
│   ├── ablation_study.py         # Feature group ablation (geometric, temporal, spatial, landmark)
│   ├── augment_v1_to_v2.py       # Build v2 dataset by augmenting v1 with new features
│   └── build_h3_stats.py         # Pre-compute H3 cell demand statistics
├── visualizations/
│   ├── plot_comparison.py         # 6 comparison plots + statistical summary
│   ├── plot_extended.py           # Extended analysis: baselines, heterogeneity, density, ablation
│   └── plot_model.py             # 3 ML insight plots (importance, scatter, rank accuracy)
├── data/                          # Raw + processed data (not in git)
│   ├── processed/                 # drivers.parquet, riders.parquet
│   ├── ml/                        # training_dataset.parquet
│   └── route_cache.db            # SQLite OSRM route cache (~110K entries)
├── models/                        # Trained ML model (not in git)
│   ├── profit_model.pkl          # Serialized LightGBM booster
│   └── feature_importance.csv    # Feature importance scores
├── results/                       # Experiment outputs
│   ├── {strategy}_outcomes.csv   # Per-strategy outcomes (coldstart, random, heuristic, warmup, oracle)
│   ├── {strategy}_outcomes_d{N}.csv  # Density experiment outcomes (d75, d50, d25, d10)
│   ├── experiment_config.json    # Experiment parameters
│   ├── ablation_results.csv      # Feature ablation study results
│   ├── model_comparison.csv      # 6-model comparison table
│   ├── density_results.csv       # Density experiment summary
│   ├── summary.txt               # Statistical summary (cold-start vs warm-up)
│   ├── extended_summary.txt      # Extended summary (all strategies, effect sizes, economics)
│   └── plots/                    # Publication-quality PNGs
├── osrm/                          # OSRM server setup and graph data
├── run_all.py                     # Single entry point for full pipeline
├── requirements.txt               # Python dependencies
└── .env                           # OSRM_BASE_URL configuration
```
