# Paper Package

This directory contains the standalone manuscript package for the rendezvous-aware route-choice paper.

## Files

- `ieee_submission.tex`
- `references.bib`
- `figures/`

## Figures

Paper-facing figures are written to:

- `figures/rendezvous_fig1_concept.png`
- `figures/rendezvous_fig2_primary.png`
- `figures/rendezvous_fig3_gap.png`
- `figures/rendezvous_fig4_dispatch.png`
- `figures/rendezvous_fig5_ml_comparator.png`
- `figures/rendezvous_fig6_sensitivity.png`
- `figures/rendezvous_fig7_strong_baselines.png`
- `figures/rendezvous_fig8_context_ablation.png`

Generate them from the repo root with:

```powershell
python visualizations\plot_rendezvous_figures.py
```

## Data Note

The recommended data stack on this branch is:

- NYC TLC Yellow trips as the primary demand source
- official NYC Centerline, Sidewalk Centerline, Building Footprints, PLUTO, and Elevation Points layers aggregated to H3 for the observability proxy

Build the urban-context layer from the repo root with:

```powershell
python scripts\build_urban_context.py --resolution 9
```
