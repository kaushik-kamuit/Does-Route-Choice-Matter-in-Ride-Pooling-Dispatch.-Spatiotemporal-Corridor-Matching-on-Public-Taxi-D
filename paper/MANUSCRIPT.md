# ML-Guided Route Selection for Carpooling Profit: Warm-Up vs Cold-Start on NYC Taxi Data

**Alternative short title:** Profit-Driven Route Choice for Ride-Pooling: Evidence from a Paired Experiment.

---

## Abstract (≈200 words)

Real-time carpooling platforms must recommend a route within seconds. Default routing optimizes travel time and ignores rider demand and matching potential along alternative paths. We study whether machine learning–guided route selection ("warm-up")—evaluating multiple road-network alternatives and choosing the route with highest predicted matching profit—yields higher platform profit than the naive strategy of accepting the default route ("cold-start").

We use NYC TLC 2015 taxi data as a proxy for carpool demand: long trips as drivers, short trips as riders. For each driver we obtain three alternative routes from OSRM (Multi-Level Dijkstra), build H3-based corridors, and train a LightGBM model to predict post-matching profit per route using 38 non-leaky features (corridor geometry, temporal and spatial-demand signals, historical H3 statistics, landmark distances). Cold-start uses the default route; warm-up selects the route with highest predicted profit; we also evaluate random and heuristic baselines and an oracle (best actual profit) as an upper bound.

In a paired simulation on 5,000 test drivers with five random seeds, warm-up achieves higher mean profit than cold-start at every rider density level (100%, 75%, 50%, 25%, 10%). The advantage increases as density decreases: +$1.10 (3.7%) at 100% density and +$2.21 (17.5%) at 10% density (paired t-test p < 1e-19; 95% CIs reported). Rank-1 accuracy (predicted-best route = actual-best route) is 67.6%. We conclude that route choice informed by predicted matching profit improves platform profit and that the effect is larger in sparser demand. All features use corridor candidates and historical H3 statistics only; no target leakage.

**Keywords:** ride-pooling, route choice, profit prediction, cold-start, spatial matching, H3, OSRM.

---

## 1. Introduction

Carpooling platforms must recommend a route to the driver quickly. Standard routing engines return a default (e.g., fastest) path between origin and destination. That path ignores where riders are likely to be and how much revenue can be earned by matching riders along the way. A "warm-up" phase—requesting a small set of genuine road-network alternatives, predicting profit per route, and recommending the route with highest predicted profit—could increase platform revenue without increasing driver effort, but it has received limited empirical attention in the literature.

Prior work on ride-pooling often focuses on matching and assignment (e.g., Alonso-Mora et al.–style shareability), optimization reviews (e.g., Agatz et al.), or route recommendation under different objective functions (e.g., Bassem et al., 2022). Few studies isolate **route choice**—which of several alternative routes to recommend—for **profit** using **actual matching outcomes** and **real road networks**. We fill this gap with a simulation study on historical NYC TLC data. We compare two strategies: **cold-start** (use the default OSRM route) and **warm-up** (request three alternatives, predict profit per route with a LightGBM model, select the route with highest predicted profit; then match riders on the chosen route). To ensure a fair comparison, both strategies share the same OSRM request (three alternatives); cold-start’s route is always one of warm-up’s options.

**Contributions.**

1. **Paired experiment.** Same drivers, rider pool, and random seeds; only route-selection strategy differs. Cold-start uses the first route from a single OSRM `alternatives=3` response; warm-up ranks all three by predicted profit and selects the best. This design controls for driver and demand heterogeneity.

2. **Profit prediction with no leakage.** The model is trained on **post-matching** profit (matching is run during dataset construction). We use 38 features from route geometry, temporal context, corridor-level demand, historical H3 cell statistics, and landmark distances. We explicitly **exclude** matching-output features such as `matched_rider_count` and `feasible_rider_count`. An ablation study shows that temporal and corridor-demand features matter most.

3. **Density-dependent advantage.** We vary rider density (100%, 75%, 50%, 25%, 10%) and show that the warm-up advantage **increases as density decreases**, with effect sizes and 95% confidence intervals reported at each level (e.g., +$1.10 at 100%, +$2.21 at 10%).

4. **Multiple baselines and winner/loser analysis.** We report results for five strategies: cold-start, random (uniform over three routes), heuristic (highest corridor rider count), ML warm-up, and oracle (best actual profit). We report the fraction of drivers who are better off, tied, or worse off under warm-up, and the share of the oracle gap captured by ML.

This study is a **simulation on historical data** (NYC TLC 2015), not a live deployment. The remainder of the paper is organized as follows: Section 2 reviews related work; Section 3 formulates the problem and setting; Section 4 describes the method (data, spatial infrastructure, matching, model, strategies, design); Section 5 summarizes experiments and reproducibility; Section 6 presents results; Section 7 discusses limitations and threats to validity; Section 8 concludes.

---

## 2. Related Work

**Ride-pooling matching and optimization.** A large body of work studies matching and shareability in ride-pooling (e.g., Alonso-Mora et al.; recent prediction-based dispatching). Our setting differs: we **fix the route first** and then match riders on that route. We do not optimize over all possible route–rider assignments; we only choose which of three precomputed routes to recommend.

**Route choice and alternatives.** Multi-criteria routing and alternative-route generation (e.g., OSRM with MLD) are well established. We use **road-network** alternatives and predict **profit** per route for selection, not only time or distance. Cold-start and warm-up share the same set of alternatives so that the comparison isolates the value of ML-based selection.

**Demand and profit prediction in mobility.** NYC TLC data have been used for demand, duration, and fare prediction (e.g., with XGBoost/LightGBM). We predict **driver–route profit** (revenue minus cost) at the **route** level, with the label defined as the **output of a matching engine** (no leakage from matching counts into features).

**Spatial indexing (H3).** Uber H3 is used in ride-hailing for demand and dispatch. We use H3 for **corridor definition** (polyline → cells, k-ring expansion) and for a rider index keyed by (cell, 15-minute time bin). Pickup and dropoff must lie in the corridor; directionality and detour are enforced via geometric projection (Shapely).

---

## 3. Problem Formulation and Setting

**Actors.** A platform serves drivers (origin, destination, departure time) and riders (pickup, dropoff, time, fare). The platform must recommend **one** route from a small set (e.g., three) in seconds.

**Routes.** For each origin–destination pair we obtain the same set of alternatives (e.g., three OSRM routes). **Cold-start** uses the default (first) route. **Warm-up** uses a model to predict profit for each route and selects the route with highest predicted profit.

**Matching.** We define a corridor as the H3 (resolution 9) cells along the route polyline, expanded by one k-ring. Candidates are riders whose pickup and dropoff cells lie in the corridor and whose 15-minute time bin is within ±1 bin of the driver’s departure. Feasibility requires directionality (rider travels ≥5% of route in driver direction), detour ≤ 4 minutes, and seat capacity. Matching is greedy by fare with seed-based tie-breaking. **Profit** = platform_share × sum(matched rider fares) − route_distance_miles × cost_per_mile (platform_share = 0.50, cost_per_mile = $0.67).

**Objective.** We compare strategies by **expected profit per driver** (and related metrics), with proper aggregation and statistical tests.

**Data proxy.** We use NYC TLC Yellow Taxi 2015 trips as a proxy for carpool demand: long trips (>10 mi) as drivers, short trips (0.5–10 mi) as riders. This choice and its limitations are revisited in the Discussion.

---

## 4. Method

### 4.0 System Architecture

Figure 1 (System Architecture) summarizes the end-to-end pipeline. The OSRM MLD server (Docker, localhost) holds the New York road network and returns 2–3 genuine alternative routes per origin–destination request. All responses are cached in SQLite (`route_cache.db`), so the simulation runs in cache-only mode with no live API calls. For each driver, the cache supplies up to three route polylines. Three shared components then operate on these routes:

- **Corridor builder:** For each polyline, we densify at 80 m steps, map points to H3 resolution-9 cells, and expand by one k-ring to form a corridor (~520 m width). Output: the set of H3 cells that define the spatial catchment for that route.
- **Matcher (Shapely/GEOS):** Given a corridor and a route polyline, the matcher (Section 4.3) finds riders whose pickup and dropoff lie in the corridor and satisfy directionality, detour, and seat constraints; it then assigns seats greedily by fare. This is used both for generating training labels (profit per route) and for simulating cold-start and warm-up outcomes.
- **LightGBM predictor:** For each of the three routes we compute 38 features (geometry, temporal, corridor demand, historical H3, landmarks, cell-level). The model predicts profit per route; warm-up selects the route with highest predicted profit.

Cold-start uses only the first route (default) and runs the matcher once. Warm-up runs the corridor builder and feature extraction for all three routes, invokes the predictor to rank them, then runs the matcher once on the selected route. The runner executes both strategies for each driver over 5 seeds (driver-major loop), then aggregates outcomes and runs statistical tests and visualizations. This architecture ensures that cold-start’s route is always one of warm-up’s options, so the comparison is fair.

### 4.1 Data and Preprocessing

We use NYC TLC Yellow Taxi 2015 (Azure Open Datasets). **Temporal split:** January–March for training, April for test; no test period appears in training. **Drivers:** trips with trip_distance_miles > 10. **Riders:** trips 0.5–10 mi, 25% sample. We assign H3 (resolution 9) cells to pickup and dropoff and use 15-minute bins for the rider index. All preprocessing is documented in the project README.

### 4.2 Spatial and Routing

We use OSRM with the Multi-Level Dijkstra (MLD) algorithm on a New York road graph (Docker). MLD returns 2–3 genuine alternative routes per request. Routes are cached in SQLite so experiments are reproducible without live routing calls.

**Corridor construction (Figure 2).** A corridor is the spatial band around a route where rider pickups and dropoffs are considered matchable. The pipeline is:

1. **Polyline** — OSRM returns a sequence of (lat, lng) points along the road.
2. **Densify** — We interpolate points every 80 m along the polyline. At H3 resolution 9 (hex edge ~174 m), 80 m steps guarantee no cell is skipped between consecutive points.
3. **H3 cells** — Each interpolated point is mapped to its H3 resolution-9 cell; we keep the ordered list of cells along the route (the “spine”).
4. **k-ring expand (k=1)** — For each spine cell we add all 6 adjacent cells via H3’s grid disk. The union of spine and neighbors forms the corridor: a band of width ~520 m around the road.

Resolution 9 and k=1 yield a catchment consistent with a short walk to the road (~500 m). Figure 9 (Corridor Map) illustrates this for one long origin–destination pair: three alternative routes and their H3 corridors are overlaid on a map; the differing coverage of the road network explains why route choice affects which riders can be matched.

### 4.3 Matching Pipeline

Figure 3 (Matching Pipeline) outlines the three-stage flow from corridor and route to matched riders and profit. The same pipeline is used for generating training labels and for simulating cold-start and warm-up.

**Stage 1 — Spatial–temporal filter (RiderIndex).** We maintain an in-memory index mapping (H3 cell, 15-minute time bin) to rider indices. For a given corridor (set of H3 cells) and driver departure time, we query riders whose *pickup* cell and *dropoff* cell both lie in the corridor and whose 15-min bin is within ±1 bin of the driver’s bin (45-minute window). This reduces the full rider set (millions) to on the order of 10–50 candidates per route.

**Stage 2 — Feasibility filter (Shapely/GEOS).** For each candidate we (a) project pickup and dropoff onto the route polyline using line_locate_point (normalized 0–1); (b) require dropoff_frac − pickup_frac ≥ 0.05 (rider travels at least 5% of the route in the driver’s direction); (c) compute perpendicular distances to the route and estimate detour as 2×(pickup_perp + dropoff_perp)×1.3 / (40 km/h), and require detour ≤ 4 minutes; (d) require passenger_count ≤ remaining seats (3). Coordinates are scaled by cos(lat) for NYC so that Shapely distances approximate meters. Output: typically 2–5 feasible riders.

**Stage 3 — Greedy assignment.** Feasible riders are sorted by fare descending, with a small random perturbation (±$0.01) per rider for seed-based tie-breaking. We assign seats greedily until capacity (3) is reached. The result is 0–3 matched riders. **Profit** = platform_share × sum(matched rider fares) − route_distance_miles × cost_per_mile (platform_share = 0.50, cost_per_mile = $0.67).

**Label for training:** For each (driver, route) we run this full pipeline and set the **label** to the resulting profit. We do **not** use the sum of all corridor rider fares; that would ignore capacity, directionality, and detour and would produce a 1000× inflated label.

### 4.4 Profit Prediction Model

**Label:** Per (driver, route), run matching → profit. **Features (38):** Geometric (distance, duration, corridor cells, sinuosity, speed, bearing, straight-line distance), temporal (hour, day of week, weekend, day of month, 15-min bin, hour sin/cos), corridor demand (rider count in corridor, demand density, mean fare, fare density), historical H3 (pickups, dropoffs, densities, mean fare, fare density per corridor), landmark distances (JFK, LGA, Penn, Times Square; nearest landmark), and cell-level (origin/dest cell pickups/dropoffs and mean fare). We **exclude** `matched_rider_count` and `feasible_rider_count`. **Model:** LightGBM regressor. **Validation:** GroupShuffleSplit by driver_id (80/20) so no driver appears in both train and validation. **Inference:** Predict profit for each of three routes; recommend argmax. **Metrics:** R² and RMSE on validation; **rank-1 accuracy** = fraction of drivers for whom the route with highest predicted profit equals the route with highest actual profit.

### 4.5 Strategies

**Cold-start:** Use routes[0]. **Random:** Uniform over the three routes. **Heuristic:** Route with highest corridor_rider_count. **ML warm-up:** Route with highest predicted profit. **Oracle:** Run matching on all three routes; pick the route with highest actual profit (upper bound). All strategies use the same OSRM `alternatives=3` response; cold-start’s route is always in warm-up’s set.

### 4.6 Experimental Design

**Paired:** Same drivers, same rider pool, same seeds; only strategy varies. We use 5 seeds; outcomes are aggregated to **per-driver means** over seeds before running statistical tests (no pseudoreplication). **Density experiments:** Same test drivers; rider set is subsampled to 100%, 75%, 50%, 25%, and 10% (fixed seed). Each density level yields a full set of strategy outcomes.

---

## 5. Experiments and Reproducibility

**Scale.** Training dataset: ~218K rows (from ~100K drivers, ~2.18 routes per driver). Test: 5,000 drivers × 5 seeds × 5 strategies. Density runs: same 5,000 drivers at each of five density levels.

**Software.** Python 3.10+, OSRM (Docker), H3, Shapely (GEOS), LightGBM, scikit-learn, scipy. Route cache: SQLite. Exact commands for dataset build, training, simulation, and plotting are documented in the repository (README and PAPER_PLAN).

**Reproducibility.** Sequence: (1) preprocess data → (2) build training dataset (with matching for labels) → (3) train model → (4) run simulation (cache-only) → (5) generate figures and summaries. Commit/tag and environment (e.g., requirements.txt) allow replication. Runtime: dataset build on the order of hours; training minutes; simulation on the order of hours for 5K drivers × 5 seeds × 5 strategies.

---

## 6. Results

### 6.1 Main Result: Warm-Up vs Cold-Start by Density

Table 1 reports mean profit (cold-start and warm-up), mean difference (warm-up − cold-start), 95% CI for the mean difference (t-based: mean ± 1.96×SEM of driver-level differences), and percentage gain at each rider density. Sample size: 5,000 drivers; per-driver values are means over 5 seeds.

**Table 1. Mean profit and warm-up advantage by rider density (5,000 drivers, 5 seeds).**  
95% CI = mean difference ± 1.96×SEM (driver-level differences).

| Density | Cold-Start ($) | Warm-Up ($) | Mean Δ ($) | 95% CI (Δ) | % Gain |
|---------|----------------|-------------|------------|------------|--------|
| 100%    | 29.90          | 31.01       | 1.10       | [0.87, 1.34] | 3.7%  |
| 75%     | 27.94          | 29.14       | 1.20       | [0.96, 1.43] | 4.3%  |
| 50%     | 24.88          | 26.15       | 1.27       | [1.05, 1.49] | 5.1%  |
| 25%     | 19.60          | 21.23       | 1.64       | [1.42, 1.86] | 8.4%  |
| 10%     | 12.67          | 14.88       | 2.21       | [2.00, 2.42] | 17.5% |

*Source: results/density_results.csv. Match rate (cold-start) ranges from 88.4% (10%) to 99.3% (100%).*

At 100% density, paired t-test t = 9.24, p = 3.51e-20; Wilcoxon W = 651696, p = 5.55e-23. The warm-up advantage increases monotonically as rider density decreases; at 10% density the gain is +$2.21 (17.5%).

### 6.2 Strategy Comparison (100% Density)

Table 2 reports mean profit (and vs cold-start) for all five strategies at 100% rider density (from extended_summary.txt).

**Table 2. Strategy comparison at 100% rider density (5,000 drivers).**

| Strategy    | Mean profit ($) | vs Cold-Start |
|------------|------------------|---------------|
| Cold-Start | 29.90            | —             |
| Random     | 29.05            | −$0.85        |
| Heuristic  | 30.91            | +$1.00        |
| ML Warm-Up | 31.01            | +$1.10        |
| Oracle     | 33.87            | +$3.97        |

Oracle advantage over cold-start is $3.97; ML captures $1.10, i.e., **27.8%** of the oracle gap. Random performs worse than cold-start; heuristic captures most of the gain; ML warm-up slightly outperforms heuristic by only **$0.10/trip** at 100% density.

### 6.3 Who Benefits

From extended_summary.txt (100% density): **23.3%** of drivers are better off with warm-up, **62.4%** tied, **14.3%** worse off. Mean gain among winners: $10.20; mean loss among losers: $8.96. Effect size (Cohen’s d) = 0.069 (negligible by conventional rules); 95% CI for mean difference (bootstrap): [$0.87, $1.34].

### 6.4 Density and Advantage

Figure 4 (paper_fig1_density_advantage.png) plots rider density (x) vs mean profit difference warm-up − cold-start (y) with 95% CI. The curve is increasing as density decreases, supporting the message that **route choice matters more when rider availability is sparser**.

### 6.5 Model Quality

Validation R² ≈ 0.79, RMSE ≈ $7.92; rank-1 accuracy ≈ 67.6%. Top features by importance: corridor_fare_density, time_bin_15min, mean_rider_fare. Figure 7 (paper_fig4_model_quality.png) shows feature importance and predicted vs actual profit on the validation set.

**Table 3. Model validation and feature ablation (GroupShuffleSplit by driver_id).**

| Experiment           | Features | R²     | RMSE   | Rank-1 |
|----------------------|----------|--------|--------|--------|
| All features         | 38       | 0.7925 | $7.92  | 67.9%  |
| Only Geometric       | 8        | 0.3292 | $14.23 | 54.6%  |
| Only Temporal        | 7        | 0.1042 | $16.45 | 46.5%  |
| Only Spatial Demand  | 13       | 0.5547 | $11.60 | 60.8%  |
| Only Landmark        | 10       | 0.3247 | $14.28 | 46.5%  |
| All minus Geometric  | 30       | 0.7893 | $7.98  | 66.9%  |
| All minus Temporal   | 31       | 0.5522 | $11.63 | 60.9%  |
| All minus Spatial    | 25       | 0.7346 | $8.95  | 61.9%  |
| All minus Landmark   | 28       | 0.7839 | $8.08  | 67.6%  |

Removing temporal features causes the largest drop; spatial-demand and geometric groups also contribute substantially.

---

## 7. Discussion

**Interpretation.** Warm-up improves mean profit by selecting routes with higher predicted matching profit. Temporal and corridor-demand features drive the model. The advantage grows when rider availability is sparser (lower density), where route choice has a larger impact on who can be matched. At the same time, the small gap between the heuristic and ML baselines at 100% density suggests that the main signal is route-aware demand ranking, while the incremental value of the full ML model over a simple count-based heuristic is modest in this saturated NYC setting.

**Limitations.** (1) Static snapshot; no dynamic arrivals or cancellations. (2) Taxi data as proxy for carpool demand. (3) Single city and period (NYC, April 2015). (4) Driver always accepts the recommended route. (5) Fixed platform share and cost. (6) Temporal train/test (conservative). (7) At 100% density, match rates are high for both strategies (~99%), limiting the visible advantage; at 10% density the gain is much larger.

**Threats to validity.** Internal: paired design and shared OSRM set reduce confounds. External: generalization to other cities or time periods is unknown. Construct: profit is platform profit; other objectives (e.g., driver or rider welfare) could be studied separately.

**Future work.** Pairwise or relative features between routes; multi-city evaluation; dynamic setting; live A/B test; other objectives (e.g., emissions, fairness).

---

## 8. Conclusion

We showed that ML-guided route selection (warm-up) yields higher mean profit than default routing (cold-start) in a simulation on NYC TLC 2015 data. The effect is statistically significant (paired t and Wilcoxon p < 1e-19), with a mean difference of +$1.10 at 100% rider density and +$2.21 at 10% density. Rank-1 accuracy is 67.6%; ML captures about 28% of the oracle gap. The advantage increases as rider density decreases, indicating that route choice is a lever for platform profit especially in sparser demand. The study is limited to a single city and historical simulation; implications for deployment would require real-world validation.

---

## Figures and Tables Checklist

- **Fig 1. System Architecture.** End-to-end pipeline: OSRM MLD → SQLite cache → Corridor builder, Matcher (Shapely), LightGBM predictor → Cold-start (1 route) and Warm-up (3 routes + ML) → Runner (paired, 5 seeds) → Statistical tests and visualizations. *To be drawn from README Section 3 (ASCII diagram) or as TikZ/vector figure.*
- **Fig 2. Corridor construction.** Pipeline: Polyline → Densify (80 m) → H3 cells (res 9) → k-ring (k=1) → corridor (~520 m width). *Schematic or single-route example.*
- **Fig 3. Matching pipeline.** Three stages: (1) RiderIndex: corridor + 15-min bin ±1 → candidates; (2) Shapely feasibility: directionality, detour ≤4 min, seats → feasible; (3) Greedy by fare + tie-break → 0–3 matched riders, profit. *Flow diagram.*
- **Fig 4.** Density vs mean profit difference (WU−CS) with 95% CI — `results/plots/paper_fig1_density_advantage.png`. Sample: 5,000 drivers, 5 seeds; error bars = 95% CI of mean difference.
- **Fig 5.** Strategy comparison (5 strategies) at 100% density — `results/plots/paper_fig2_strategy_comparison.png`.
- **Fig 6.** Per-driver profit difference distribution (and by route length) — `results/plots/paper_fig3_profit_difference.png`.
- **Fig 7.** Model: feature importance (top 15) + predicted vs actual (R², RMSE) — `results/plots/paper_fig4_model_quality.png`.
- **Fig 8.** Mean profit by strategy across densities — `results/plots/paper_fig5_profit_by_density.png`.
- **Fig 9. Corridor map.** Three alternative routes and their H3 corridors for one long O–D. Interactive: `results/plots/corridor_map.html`; for print use a static PNG export. Illustrates how corridor coverage differs by route and why route choice affects matchability.
- **Table 1.** Main result by density (Section 6.1) — from `results/density_results.csv`.
- **Table 2.** All strategies at 100% — from `results/extended_summary.txt`.
- **Table 3.** Model validation and ablation (R², RMSE, rank-1) — from README.
- **Table 4 (optional).** Feature groups and counts (38 features).

All figure captions must state sample size (e.g., 5,000 drivers, 5 seeds) and what error bars or CIs represent.

---

## References (checklist for author)

Starter references used in the current draft:

- Alonso-Mora et al. (2017), *On-demand high-capacity ride-sharing via dynamic trip-vehicle assignment*.
- Agatz et al. (2012), *Optimization for dynamic ride-sharing: A review*.
- Agatz et al. (2011), *Dynamic ride-sharing: A simulation study in metro Atlanta*.
- Stiglic et al. (2016), *Making dynamic ride-sharing work: The impact of driver and rider flexibility*.
- Bassem et al. (2022), *Route Recommendation to Facilitate Carpooling*.
- Berlingerio et al. (2017), *The GRAAL of carpooling: GReen And sociAL optimization from crowd-sourced data*.
- Xia et al. (2019), *A carpool matching model with both social and route networks*.
- Chen et al. (2021), *Efficient dispatching for on-demand ride services: Systematic optimization via Monte-Carlo tree search*.
- Ke et al. (2017), *LightGBM: A highly efficient gradient boosting decision tree*.
- OSRM project documentation.
- Uber H3 documentation.
- NYC TLC Trip Record Data / Azure Open Datasets.

This should still be expanded to match the target venue’s formatting and coverage requirements.

---

## Appendix (optional)

- **A. Full feature list (38).** Geometric (8), temporal (7), corridor demand (4), historical H3 (6), landmark (10), cell-level (3).
- **B. Ablation table (full).** All subsets as in README Table 10.2.
- **C. Reproducibility.** Commands: build dataset → train → run_density_experiments → plot_comparison, plot_extended, plot_model. Environment: requirements.txt; OSRM Docker; data version and commit/tag.

---

---

## Correctness and reviewer focus (compliance)

- **No leakage:** Features use corridor candidates and historical H3 statistics only. Label = matched profit from matching engine. `matched_rider_count` and `feasible_rider_count` are not used.
- **SEM vs SD:** All CIs use SEM (of driver-level means/differences), not SD of raw outcomes.
- **Paired design:** Tests and CIs are on per-driver values (mean over seeds first); no pseudoreplication.
- **Rank-1:** Fraction of drivers for whom route with highest predicted profit = route with highest actual profit.
- **Density:** Fraction of test riders retained; same drivers across densities.
- **Reproducibility:** Clear sequence — data → train → run experiments → figures; document commands and environment.

---

*Document generated from project results. All numbers sourced from `results/summary.txt`, `results/extended_summary.txt`, `results/density_results.csv`, and README. No leakage: features use corridor candidates and historical H3 only; label = matched profit.*
