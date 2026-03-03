# Paper Plan: Warm-Up vs Cold-Start Route Selection

**Goal:** Publish a paper showing that ML-guided route selection (warm-up) is better than default routing (cold-start) for carpooling profit.

**Decisions (locked):**
- **Route length categories:** Keep existing NYC-oriented thresholds (short <13 mi, medium 13–18 mi, long ≥18 mi). See `src/simulation/data_types.py`.
- **Density focus:** Lead with reduced rider density (50%, 25%, 10%). At 100% density, NYC route alternatives overlap heavily and corridors saturate, so the warm-up advantage is small; at lower density, route choice matters more.
- **Run scale:** Longer runs are acceptable — 50K train drivers for the v2 dataset, 5K–10K test drivers per density, 5 seeds.
- **Single training source:** Official training data is produced only by `scripts/build_enhanced_dataset.py` → `data/ml/training_dataset_v2.parquet`. Model: `src/models/train_profit_model.py` → `models/profit_model_v2.pkl`.

**Pipeline (reproducible):**
1. Build v2 dataset: `python scripts/build_enhanced_dataset.py --sample 50000`
2. Train model: `python src/models/train_profit_model.py`
3. Run density experiments: `python scripts/run_density_experiments.py --sample 5000` (or 10000)
   - Produces `results/{strategy}_outcomes.csv` (100%) and `results/{strategy}_outcomes_d75.csv`, `_d50.csv`, `_d25.csv`, `_d10.csv`
4. Plots and stats: `python visualizations/plot_comparison.py`, `python visualizations/plot_extended.py`, `python visualizations/plot_model.py`

**Paper narrative:**
- Warm-up beats cold-start in mean profit at all densities.
- The advantage is **larger at lower density** (e.g. 25%, 10%) because route alternatives overlap less in impact when fewer riders are available; at 100% density, many corridors fill regardless of route choice.
- Report effect size (Cohen's d) and 95% CIs by density; use density vs advantage and density vs match rate as main figures.

**Outputs to use in the paper:**
- `results/extended_summary.txt` — strategy comparison, effect sizes, oracle gap
- `results/plots/density_advantage.png` — warm-up advantage vs rider density
- `results/plots/baseline_comparison.png` — mean profit by strategy (can stratify by density in text)
- `results/summary.txt` — paired tests (cold-start vs warm-up)

**After the 50K dataset build completes:**
1. Retrain: `python src/models/train_profit_model.py`
2. Run density experiments: `python scripts/run_density_experiments.py --sample 5000` (or `--sample 10000`)
3. Regenerate: `python visualizations/plot_comparison.py`, `python visualizations/plot_extended.py`, `python visualizations/plot_model.py`
