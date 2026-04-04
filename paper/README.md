# Paper Package

This directory contains the standalone IEEE journal-style manuscript package for the dispatch-first, realism-aware NYC route-selection study.

## Files

- `ieee_submission.tex`: main manuscript source
- `references.bib`: bibliography
- `figures/`: local figure assets used by the paper

## Figures referenced by the manuscript

- `figures/paper_fig1_dispatch_architecture_v2.png`
- `figures/paper_fig2_matching_ball_mechanism.png`
- `figures/paper_fig3_dispatch_density.png`
- `figures/paper_fig4_cross_domain.png`
- `figures/paper_fig5_single_driver_mechanism.png`
- `figures/paper_fig6_model_support.png`
- `figures/paper_fig7_sensitivity.png`

Supporting locked source assets that should not be replaced accidentally:

- `figures/paper_fig1_dispatch_architecture_source.png`: canonical architecture diagram source
- `figures/paper_fig2a_corridor_map.jpg`: mandatory corridor-map source for Figure 2(a)

Refresh publication figures from the repo root with:

```powershell
python visualizations\plot_paper_figures.py
python scripts\validate_paper_consistency.py
```

## Overleaf

Upload the full contents of `paper/` to a new Overleaf project. No parent-directory paths are required.

That folder contains:

- `ieee_submission.tex`
- `references.bib`
- `figures/`

## Result anchors

The manuscript narrative is anchored to the dispatch-first and realism-first result files under `../results/`, especially:

- `../results/dispatch_yellow_primary.csv`
- `../results/dispatch_green_primary.csv`
- `../results/dispatch_density_summary.csv`
- `../results/dispatch_service_wait_summary.csv`
- `../results/dispatch_window_sensitivity.csv`
- `../results/dispatch_detour_sensitivity.csv`
- `../results/domain_transfer_summary.csv`
- `../results/domain_temporal_generalization.csv`
- `../results/realism_primary_summary.csv`
- `../results/paper_primary_summary.csv`
- `../results/strong_baseline_comparison.csv`
- `../results/strategy_gap_results.csv`
- `../results/model_comparison.csv`
- `../results/ablation_results.csv`
- `../results/h3_corridor_sensitivity.csv`
- `../results/economics_sensitivity.csv`
- `../results/runtime_profile.csv`

The paper is organized so that rolling dispatch is the main system-level evidence and the larger single-driver study is controlled secondary evidence.

## Compile

Compile from the `paper/` directory with a standard LaTeX IEEE toolchain, for example:

```powershell
latexmk -pdf ieee_submission.tex
```

If `latexmk` is unavailable, run:

```powershell
pdflatex ieee_submission.tex
bibtex ieee_submission
pdflatex ieee_submission.tex
pdflatex ieee_submission.tex
```
