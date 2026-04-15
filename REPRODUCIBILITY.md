# Reproducibility Guide

This repository is organized around the rendezvous-aware route choice workflow under occlusion.

## Environment

```powershell
python -m pip install -r requirements.txt
```

The active code path depends on:

- `pandas`, `pyarrow`
- `h3`, `shapely`
- `scikit-learn`, `joblib`
- `matplotlib`
- `requests`, `python-dotenv`, `tqdm`

## Main Commands

Build the official NYC urban-context layer used by the observability proxy:

```powershell
python scripts\build_urban_context.py --resolution 9
```

Use `--max-rows-per-asset` for a quick smoke run, or `--metadata-only` if you only want the source manifest.

Build a meeting-point training dataset:

```powershell
python scripts\build_rendezvous_dataset.py --sample 1000
```

Train the ML meeting-point comparator:

```powershell
python scripts\train_meeting_point_model.py
```

Run the controlled single-driver study:

```powershell
python scripts\run_rendezvous_artifact.py --sample 1000 --seeds 3
```

Run the rolling dispatch validation:

```powershell
python scripts\run_rendezvous_dispatch.py --sample 500 --seeds 3
```

Summarize the outputs:

```powershell
python scripts\summarize_rendezvous_results.py
```

If you have older flat files in `results/`, register them into immutable run folders first:

```powershell
python scripts\backfill_rendezvous_run_registry.py
```

For the simplest end-to-end paper refresh, use:

```powershell
python scripts\prepare_rendezvous_submission.py
```

That command safely runs backfill, summary rebuild, figure generation, case-study generation, and Overleaf package sync in sequence.

Generate publication figures:

```powershell
python visualizations\plot_rendezvous_figures.py
```

## Result Files

Raw experiment artifacts are now run-scoped and manifest-backed:

- `results/runs/<run_id>/manifest.json`
- `results/runs/<run_id>/rendezvous_driver_outcomes.csv`
- `results/runs/<run_id>/rendezvous_route_evaluations.csv`
- `results/runs/<run_id>/rendezvous_dispatch_summary.csv`

- `data/urban_context/processed/urban_context_h3_res9.parquet`
- `data/urban_context/processed/urban_context_sources.json`
- `results/rendezvous_primary_summary.csv`
- `results/rendezvous_nominal_realized_gap.csv`
- `results/rendezvous_meeting_point_comparison.csv`
- `results/plots/rendezvous_fig*.png`

## Paper Package

The standalone manuscript lives in `paper_rendezvous/`:

- `paper_rendezvous/ieee_submission.tex`
- `paper_rendezvous/references.bib`
- `paper_rendezvous/figures/`

Upload that directory directly to Overleaf if you want a clean paper-only project.
