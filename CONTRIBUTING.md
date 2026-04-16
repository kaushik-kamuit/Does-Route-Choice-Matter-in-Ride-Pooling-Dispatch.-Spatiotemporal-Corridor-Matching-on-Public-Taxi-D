# Contributing

Thanks for helping improve the artifact.

## Good contributions

- documentation fixes and public-release polish
- reproducibility fixes
- bug fixes with focused tests
- result-validation and consistency improvements

## Workflow

1. Create a branch for your change.
2. Keep edits scoped to a single issue or improvement.
3. Run the smallest relevant validation before opening a PR.
4. Update docs if your change affects commands, outputs, or assumptions.

## Validation

For small documentation changes, a careful link and path check is usually enough.

For code or result-pipeline changes, prefer running one or more of:

```powershell
python -m unittest tests.test_artifact_sanity
python scripts\validate_paper_consistency.py
python visualizations\plot_paper_figures.py
```

If a full artifact rerun is needed:

```powershell
python run_all.py
```

## Style expectations

- Keep changes readable and well-scoped.
- Preserve publication-facing terminology used in the paper and result summaries.
- Avoid silently changing scenario semantics, especially request-window, density, oracle-scope, or retained-sample definitions.

## Questions

If you are preparing a public fork or paper companion release, keep `README.md`, `REPRODUCIBILITY.md`, `DATA_AND_ASSETS.md`, and `paper/README.md` aligned.
