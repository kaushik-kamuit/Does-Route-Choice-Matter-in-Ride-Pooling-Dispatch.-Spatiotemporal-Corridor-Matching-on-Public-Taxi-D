# Rendezvous-Aware Route Choice Under Occlusion

This branch is a fresh research workspace for studying route choice in urban ride-pooling through feasible and observable rendezvous opportunities. The main question is no longer whether route-aware corridors help in general, but whether routes should be valued by the common meeting opportunities they induce and by how reliable those opportunities remain under urban occlusion.

The implementation is intentionally narrow:

- `src/data_prep/`, `src/spatial/`, and `src/matching/rider_index.py` are retained as low-level infrastructure.
- `scripts/build_urban_context.py` downloads official NYC street, sidewalk, building, and PLUTO layers, then aggregates them to H3 for observability-aware experiments.
- `src/rendezvous/` is the main Paper 2 package and contains the new route evaluator, meeting-point logic, observability proxy, and dispatch validation shell.
- `paper_rendezvous/` is the only manuscript package on this branch.
- `results/runs/<run_id>/` stores immutable raw and per-run derived artifacts.
- `results/rendezvous_*` and `results/plots/rendezvous_fig*` are the paper-facing summary views rebuilt from the run registry.

The active policy family is:

- `corridor_only`: near-route rider compatibility only
- `rendezvous_only`: feasible common meeting opportunities without observability
- `rendezvous_observable`: feasible common meeting opportunities with observability-aware valuation
- `ml_meeting_point_comparator`: ML-based meeting-point ranking on top of the same route valuation shell

## Quick Start

```powershell
python -m pip install -r requirements.txt
python scripts\build_urban_context.py --resolution 9
python run_all.py --single-driver-only --sample 250 --seeds 1
python scripts\run_rendezvous_dispatch.py --sample 100 --seeds 1
python scripts\prepare_rendezvous_submission.py
```

`prepare_rendezvous_submission.py` is the easiest safe refresh path. It backfills legacy runs into the registry if needed, rebuilds paper summaries, refreshes figures, rebuilds case studies, and syncs the slim Overleaf package in the correct order.

## Repo Layout

- `src/rendezvous/`: Paper 2 method and evaluation logic
- `data/urban_context/`: official NYC context layers and H3 summaries used by the occlusion proxy
- `scripts/`: dataset building, experiment runners, training, and summarization
- `paper_rendezvous/`: standalone manuscript package
- `results/runs/`: immutable experiment runs with manifests
- `results/`: paper summaries and figure outputs rebuilt from registered runs

## Notes

This branch descends from earlier route-choice work, but it deliberately avoids reusing the earlier manuscript framing, result namespace, figure namespace, and route-ranking story. The reused components are infrastructure only.
