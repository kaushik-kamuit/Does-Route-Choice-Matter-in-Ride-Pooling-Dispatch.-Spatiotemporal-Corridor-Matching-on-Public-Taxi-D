# Route-Aware Matching-Ball Dispatch for Ride-Pooling

This repository is a realism-first research artifact for route-aware ride-pooling evaluation on NYC TLC 2015 taxi data. The codebase supports both:

- a controlled single-driver paired route-choice study
- a rolling-horizon multi-driver dispatch study

The core question is simple: if a platform evaluates a few genuine road-network alternatives before committing a driver to a route, can it choose a better corridor than the default route, and does that advantage survive once multiple drivers compete for the same rider pool?

The current artifact is centered on:

- NYC Yellow Taxi 2015 as the primary public domain
- NYC Green Taxi 2015 as the public robustness domain when the selected four-month window clears the volume thresholds
- long trips as proxy drivers and short trips as proxy riders
- H3 corridor indexing plus a matching-ball feasibility filter
- a LightGBM route-profit predictor trained on 38 features
- a realism-first request-time filter that separates 15-minute index bins from the true matching window
- a rolling-horizon dispatch simulator with 60-second batching and rider exclusivity

## Headline dispatch result

The main paper scenario uses:

- rider pre-sample: `25%`
- dispatch cadence: `60-second batches`
- primary retained-sample density: `10%`
- exact request window: `5 minutes`
- index lookup: `15-minute bins` with `±1` adjacent bins
- detour cap: `4 minutes`

In the primary Yellow dispatch scenario, mean scenario profit per launched driver is:

| Policy | Yellow dispatch profit/driver |
|---|---:|
| `Cold-start` | `-$8.08` |
| `Best heuristic (feasible count)` | `-$7.30` |
| `ML warm-up` | `-$7.15` |
| `Oracle (within route set)` | `-$7.02` |

The public Green robustness run under the same primary scenario is:

| Policy | Green dispatch profit/driver |
|---|---:|
| `Cold-start` | `-$7.66` |
| `Best heuristic (feasible count)` | `-$7.54` |
| `ML warm-up` | `-$7.27` |
| `Oracle (within route set)` | `-$7.25` |

The larger isolated single-driver study is retained as controlled secondary evidence. There, the same `10%` / `5-minute` scenario moves from `-$7.40` under cold-start to `-$6.17` under warm-up, with the strongest heuristic at `-$6.28`.

Important semantic note: `100%` in the retained-sample density sweeps still means the full retained `25%` rider sample used by the artifact, not full city demand.

## Main takeaways

- Route-aware dispatch beats cold-start in both Yellow and Green under the 5-minute exact-window scenario.
- In rolling dispatch, the main gain comes from route-aware retrieval versus cold-start; the ML edge over the strongest heuristic is positive but modest.
- The controlled single-driver study shows a clearer ML-over-heuristic signal, which helps explain why rider exclusivity compresses the learned advantage in dispatch.
- Exact request-window assumptions matter more than moderate detour changes. In Yellow dispatch at `10%` density, ML warm-up profit is `-$8.07` under a 2-minute window, `-$7.15` under 5 minutes, and `-$5.95` under 10 minutes.
- Temporal holdout model selection uses Jan--Feb 2015 for training and Mar 2015 for validation, reaching `R^2 = 0.801` and `RMSE = $5.85` for tuned LightGBM.
- These are **scenario profit** values under fixed share/cost assumptions, not calibrated platform margins.

## Repository layout

- `src/`: matching, routing, simulation, dispatch, and model-training code
- `scripts/`: artifact orchestration, summaries, validators, and analysis utilities
- `visualizations/`: publication and exploratory plotting scripts
- `paper/`: standalone IEEE-style manuscript package
- `results/`: generated CSV summaries, dispatch tables, paired outcome tables, and publication figures
- `artifacts/legacy_optimistic_45min/`: archived pre-remediation optimistic artifact

## Rebuild the artifact

Create a virtual environment, install dependencies, and run:

```powershell
python -m pip install -r requirements.txt
python run_all.py
```

`run_all.py` now runs both the single-driver realism artifact and the rolling dispatch artifact. The two main entry points are:

- `scripts/run_realism_artifact.py` for the controlled single-driver study
- `scripts/run_dispatch_artifact.py` for the rolling dispatch study and public-domain robustness

The single-driver artifact:

1. archives the legacy optimistic 45-minute-window outputs
2. rebuilds the 5-minute training dataset
3. retrains the profit model with temporal holdout model selection
4. runs the primary density sweep and stronger-baseline comparisons
5. runs request-window, detour, H3, and economics sensitivity experiments
6. regenerates figures, summaries, and the paper validator outputs

The dispatch artifact:

1. reuses the realism-first route-ranking stack
2. runs rolling 60-second dispatch batches with exact request windows
3. compares cold-start, the strongest heuristic, ML warm-up, and oracle-style upper bounds
4. writes dispatch summaries for Yellow primary scenarios and Green domain robustness when available

Useful shortcuts:

```powershell
python run_all.py --skip-dataset --skip-train
python run_all.py --primary-only
python run_all.py --single-driver-only
python run_all.py --dispatch-only --sample 1000 --seeds 3
python run_all.py --sample 5000 --seeds 5
python scripts\summarize_realism_results.py
python scripts\run_dispatch_artifact.py --sample 1000 --seeds 3 --primary-only
python scripts\analyze_strategy_gaps.py
python visualizations\plot_paper_figures.py
python scripts\validate_paper_consistency.py
```

When you run dispatch smoke tests with a smaller `--sample` or fewer `--seeds`, the dispatch summary CSVs record `driver_sample_size` and `n_seeds` explicitly so they are not mistaken for manuscript-grade runs.

## Key output files

- `results/realism_primary_summary.csv`: main 5-minute single-driver density sweep
- `results/paper_primary_summary.csv`: single-row single-driver anchor
- `results/window_sensitivity.csv`: single-driver request-window sensitivity grid
- `results/detour_sensitivity.csv`: single-driver detour sensitivity at 10% density
- `results/strong_baseline_comparison.csv`: strongest non-ML baseline comparison
- `results/temporal_generalization.csv`: Yellow Jan--Feb to Mar temporal holdout metrics
- `results/h3_corridor_sensitivity.csv`: matching-ball geometry sensitivity at 10% density
- `results/economics_sensitivity.csv`: economics sensitivity at 10% density
- `results/runtime_profile.csv`: single-driver compute-time profile
- `results/dispatch_yellow_primary.csv`: primary rolling-dispatch summary on NYC Yellow
- `results/dispatch_green_primary.csv`: public robustness-domain dispatch summary on NYC Green when available
- `results/domain_transfer_summary.csv`: Yellow-vs-Green dispatch comparison
- `results/domain_temporal_generalization.csv`: per-domain temporal validation metrics
- `results/dispatch_density_summary.csv`: dispatch density sweep
- `results/dispatch_service_wait_summary.csv`: dispatch service-rate and wait-time summary
- `results/dispatch_window_sensitivity.csv`: dispatch request-window sensitivity
- `results/dispatch_detour_sensitivity.csv`: dispatch detour sensitivity
- `results/scenario_assumptions.csv`: primary scenario assumptions used by the paper
- `results/strategy_gap_results.csv`: paired policy-gap statistics
- `results/model_comparison.csv`: model-family comparison
- `results/ablation_results.csv`: feature ablation study
- `results/plots/paper_fig*.png`: current paper-facing figures

## Core methodology in one paragraph

For each driver trip, the artifact requests three OSRM alternatives, densifies each polyline, maps it to ordered H3 resolution-9 cells, and expands those cells with a one-ring neighborhood to form a route corridor. A `RiderIndex` retrieves riders whose pickup and drop-off cells both lie in the corridor and whose coarse index bins are near the driver departure. A second exact filter then enforces the true request window in minutes. Feasible riders must also satisfy directionality, detour, and seat-capacity checks. The warm-up policy uses a LightGBM model to predict post-matching scenario profit for each route and chooses the best one.

In the dispatch extension, drivers are launched in rolling 60-second batches, riders enter an open pool when their request batch arrives, expired requests are removed once they exceed the exact request window, and competing drivers claim riders through a greedy exclusivity pass that recomputes later drivers against the reduced rider pool.

## Paper package

The standalone manuscript lives in `paper/`:

- `paper/ieee_submission.tex`
- `paper/references.bib`
- `paper/figures/`

The current main paper figure set is:

- `paper/figures/paper_fig1_dispatch_architecture_v2.png`
- `paper/figures/paper_fig2_matching_ball_mechanism.png`
- `paper/figures/paper_fig3_dispatch_density.png`
- `paper/figures/paper_fig4_cross_domain.png`
- `paper/figures/paper_fig5_single_driver_mechanism.png`
- `paper/figures/paper_fig6_model_support.png`
- `paper/figures/paper_fig7_sensitivity.png`

Upload the contents of `paper/` directly to Overleaf to compile the standalone package.

## License

This repository is released under the [MIT License](LICENSE). Third-party data sources, map tiles, OSRM services, and other external resources remain subject to their original terms.

## Legacy note

The repository still contains historical notes and archived outputs from the earlier optimistic artifact. The current publication-facing story, however, is the dispatch-first 5-minute exact-window configuration described above. Use the files in `results/`, `paper/`, and `artifacts/legacy_optimistic_45min/` accordingly.
