# Rendezvous-Aware Route Choice Under Occlusion

This repository contains the code, experiments, and manuscript package for studying route choice in urban ride-pooling through feasible and observable rendezvous opportunities. The central question is whether routes should be valued by the common meeting opportunities they induce, rather than by corridor proximity alone.

## Repository Contents

- `src/rendezvous/`: route evaluation, meeting-point generation, observability proxy, and dispatch validation logic
- `scripts/`: data preparation, experiment runners, result summarization, and manuscript asset generation
- `data/urban_context/`: official NYC street, sidewalk, building, and PLUTO layers plus derived H3 summaries
- `results/runs/<run_id>/`: immutable run-scoped artifacts and manifests
- `paper_rendezvous/`: manuscript source package
- `paper_rendezvous_overleaf/`: slim Overleaf-ready manuscript package

## Policy Variants

- `corridor_only`: near-route rider compatibility only
- `rendezvous_only`: feasible common meeting opportunities without observability-aware weighting
- `rendezvous_observable`: feasible common meeting opportunities with observability-aware valuation
- `ml_meeting_point_comparator`: learned meeting-point ranking on the same valuation shell

## Quick Start

```powershell
python -m pip install -r requirements.txt
python scripts\build_urban_context.py --resolution 9
python run_all.py --single-driver-only --sample 250 --seeds 1
python scripts\run_rendezvous_dispatch.py --sample 100 --seeds 1
python scripts\prepare_rendezvous_submission.py
```

`prepare_rendezvous_submission.py` is the safest end-to-end refresh path. It backfills legacy runs into the registry if needed, rebuilds summaries, regenerates figures and case studies, and synchronizes the Overleaf package.

## Reproducibility

The full command sequence and output layout are documented in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Manuscript Package

The paper source lives in `paper_rendezvous/`. The synchronized submission package in `paper_rendezvous_overleaf/` is organized so it can be uploaded directly to Overleaf as a paper-only project.
