# Data and Assets

This repository combines original code with third-party data sources and generated research artifacts.

## What is tracked in Git

- source code under `src/`, `scripts/`, `visualizations/`, and `tests/`
- paper sources under `paper/`
- selected publication-facing CSV summaries under `results/`
- publication-facing PNG figures under `results/plots/`

## What is not tracked in Git

- raw taxi parquet files
- large processed parquet files
- SQLite route caches and other large local artifacts
- local paper build outputs

Those files are excluded through `.gitignore` to keep the public repository lightweight and legally cleaner.

## Public data source

The artifact is built around public NYC Taxi and Limousine Commission trip data accessed through Azure Open Datasets workflows used by the repository scripts.

Before redistributing raw or transformed data, confirm the current terms for:

- NYC TLC data
- Azure Open Datasets mirrors or access layers

## Routing and spatial dependencies

This project relies on third-party tools and standards including:

- OSRM for route alternatives
- H3 for spatial indexing

Those tools remain governed by their own licenses and usage terms.

## Figures and result tables

The checked-in figures under `results/plots/` and `paper/figures/` are generated research assets intended to support the manuscript and public repository landing page.

If you regenerate them, prefer doing so from the checked-in scripts:

```powershell
python visualizations\plot_paper_figures.py
python scripts\validate_paper_consistency.py
```

## Public release checklist

Before pushing a public repository or attaching it to a paper submission, verify:

- no raw taxi files are staged
- no local caches or credentials are staged
- no generated temporary LaTeX files are staged
- the paper still points to the current checked-in figure set
- the README and reproducibility guide reflect the current headline scenario

## License summary

The repository code is released under the [MIT License](LICENSE).

That MIT license does not override the terms of:

- external datasets
- OSRM or map-data components
- any other third-party software or assets used during rebuild
