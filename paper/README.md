# Paper: ML-Guided Route Selection for Carpooling Profit

This folder contains the current manuscript package and full LaTeX for the warm-up vs cold-start route selection study.

## Files

- **main_full.tex** — Complete LaTeX (single file) for Overleaf: copy entire contents into a new Overleaf project as \texttt{main.tex}. Includes TikZ for Fig 1 (architecture), Fig 2 (corridor pipeline), Fig 3 (matching pipeline); \texttt{includegraphics} for Fig 4–9. Upload the PNG figures to the project root (or set \texttt{\textbackslash graphicspath}).
- **MANUSCRIPT.md** — Full paper in Markdown (title, abstract, introduction through conclusion, tables, figure checklist, references checklist, appendix outline). Target length: ~10–12 pages + refs (e.g., IEEE T-ITS or Transportation Research Part C).
- **figures_architecture_matching.tex** — Standalone TikZ for Figs 1–3 (optional; \texttt{main_full.tex} inlines these).

## Data sources (all numbers in manuscript)

- `../results/summary.txt` — Paired tests, mean profit by strategy and route length.
- `../results/extended_summary.txt` — All 5 strategies, effect size (Cohen’s d), 95% CIs, winner/loser %, oracle gap.
- `../results/density_results.csv` — Mean profit and warm-up advantage by rider density (10%, 25%, 50%, 75%, 100%).
- `../README.md` — Method details, ablation table, model comparison, design decisions.

## Figures (in manuscript order)

- **Fig 1. System Architecture** — Use `figures_architecture_matching.tex` (TikZ) in Overleaf, or draw from README Section 3.
- **Fig 2. Corridor construction** — TikZ in `figures_architecture_matching.tex`.
- **Fig 3. Matching pipeline** — TikZ in `figures_architecture_matching.tex`.
- **Fig 4:** `../results/plots/paper_fig1_density_advantage.png`
- **Fig 5:** `../results/plots/paper_fig2_strategy_comparison.png`
- **Fig 6:** `../results/plots/paper_fig3_profit_difference.png`
- **Fig 7:** `../results/plots/paper_fig4_model_quality.png`
- **Fig 8:** `../results/plots/paper_fig5_profit_by_density.png`
- **Fig 9. Corridor map:** `../results/plots/corridor_map.html` (interactive). For PDF, export a static PNG (e.g. screenshot or run a headless export from the HTML).

## Next steps for author

1. Copy manuscript into target journal template (LaTeX or Word).
2. Expand the starter bibliography into the target journal's final reference format and count.
3. Insert figures and ensure captions match checklist (sample size, CI definition).
4. Run compliance check: every claimed number exists in repo; CIs and tests defined; limitations and threats stated.
5. Confirm sample size consistency (e.g., 5,000 drivers for main comparison; 5 seeds; per-driver aggregation before tests).
