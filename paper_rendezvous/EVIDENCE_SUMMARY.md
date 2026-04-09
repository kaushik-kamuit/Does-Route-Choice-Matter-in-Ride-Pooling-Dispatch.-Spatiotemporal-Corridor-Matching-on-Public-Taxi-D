# Evidence Summary

This note summarizes the current evidence on branch `codex/rendezvous-aware-route-choice-under-occlusion` after the calibration and robustness pass.

## What Changed In This Pass

Two Q1-oriented upgrades were added on top of the repaired pipeline:

- a non-test observability calibration pass using a larger Yellow meeting-point dataset (`19,798` rows split into train/valid/test)
- a time-slice robustness layer in the headline `sparse_high_occlusion` regime

The calibrated weight profile saved in [observability_weights_yellow.json](K:\Kamuit\Uber_Logic\Research_paper_2\models\observability_weights_yellow.json) is:

- straightness: `0.20`
- turn: `0.30`
- ambiguity: `0.30`
- clutter: `0.20`

Calibration helps only modestly. The test Brier score improves from `0.13636` to `0.13634` and test ROC AUC improves from `0.5348` to `0.5403`, while validation AUC is slightly worse. So the calibrated profile is useful mainly because it is less heuristic and more disciplined, not because it produces a dramatic standalone lift.

## Main Takeaways

- `corridor_only` is still decisively weakest.
- The main gain continues to come from feasible rendezvous valuation.
- `rendezvous_observable` is the strongest deterministic policy in the paper's headline `sparse_high_occlusion` regime.
- That observability gain is still modest relative to `rendezvous_only`, so the correct headline remains regime-dependent rather than universal.
- Turning off urban context reduces hard-regime performance for the rendezvous-aware methods.
- The hard-regime ordering survives a time-slice robustness check: morning peak is harder than all-day, but observability-aware valuation still helps most there.

## Single-Driver Results

### Primary (`density=100`, `occlusion_lambda=0.25`, calibrated, all day)

- `corridor_only`: `6.76`
- `time_only_baseline`: `16.03`
- `feasible_count_baseline`: `17.12`
- `walk_aware_rendezvous`: `17.30`
- `rendezvous_only`: `17.60`
- `rendezvous_observable`: `17.57`
- `ml_meeting_point_comparator`: `17.77`

Bootstrap 95% intervals from [rendezvous_policy_bootstrap_ci.csv](K:\Kamuit\Uber_Logic\Research_paper_2\results\rendezvous_policy_bootstrap_ci.csv):

- `corridor_only`: `[6.11, 7.45]`
- `rendezvous_only`: `[17.00, 18.22]`
- `rendezvous_observable`: `[16.96, 18.22]`

Interpretation:

- the stronger baselines keep the paper honest
- rendezvous-aware valuation remains much better than route exposure alone
- in the easier primary regime, `rendezvous_only` still edges `rendezvous_observable`

### Sparse High Occlusion (`density=25`, `occlusion_lambda=0.40`, calibrated, all day)

- `corridor_only`: `0.59`
- `time_only_baseline`: `4.08`
- `feasible_count_baseline`: `5.20`
- `walk_aware_rendezvous`: `5.40`
- `rendezvous_only`: `5.52`
- `rendezvous_observable`: `5.72`
- `ml_meeting_point_comparator`: `5.80`

Bootstrap 95% intervals:

- `corridor_only`: `[0.11, 1.06]`
- `rendezvous_only`: `[5.01, 6.07]`
- `rendezvous_observable`: `[5.24, 6.26]`
- `feasible_count_baseline`: `[4.69, 5.72]`
- `walk_aware_rendezvous`: `[4.87, 6.00]`

Paired deltas versus `corridor_only` from [rendezvous_pairwise_deltas_vs_corridor.csv](K:\Kamuit\Uber_Logic\Research_paper_2\results\rendezvous_pairwise_deltas_vs_corridor.csv):

- `time_only_baseline`: `+3.49`
- `feasible_count_baseline`: `+4.61`
- `walk_aware_rendezvous`: `+4.81`
- `rendezvous_only`: `+4.93`
- `rendezvous_observable`: `+5.13`

Paired delta versus `rendezvous_only` from [rendezvous_pairwise_deltas_vs_rendezvous_only.csv](K:\Kamuit\Uber_Logic\Research_paper_2\results\rendezvous_pairwise_deltas_vs_rendezvous_only.csv):

- `rendezvous_observable`: `+0.20`
- bootstrap 95% interval: `[-0.15, 0.54]`

Interpretation:

- this remains the cleanest regime for the paper's full claim
- `rendezvous_observable` is the strongest deterministic policy
- the gain over `rendezvous_only` is positive but still modest

## Time-Slice Robustness

The additional robustness layer keeps the same hard regime but changes the temporal slice.

### Sparse High Occlusion, Morning Peak (`07:00-10:00`)

- `corridor_only`: `-1.65`
- `time_only_baseline`: `2.89`
- `feasible_count_baseline`: `3.60`
- `walk_aware_rendezvous`: `3.41`
- `rendezvous_only`: `3.76`
- `rendezvous_observable`: `4.10`
- `ml_meeting_point_comparator`: `3.54`

Interpretation:

- morning peak is harsher than the all-day slice
- the observability-aware deterministic policy is still best in this harder time window

### Sparse High Occlusion, Evening Peak (`16:00-19:00`)

- `corridor_only`: `1.35`
- `time_only_baseline`: `5.65`
- `feasible_count_baseline`: `7.04`
- `walk_aware_rendezvous`: `7.12`
- `rendezvous_only`: `7.62`
- `rendezvous_observable`: `7.60`
- `ml_meeting_point_comparator`: `7.13`

Interpretation:

- evening peak is easier than morning peak
- the rendezvous-aware ordering still holds
- the observability benefit narrows, which fits the paper's regime-dependent framing

## Dispatch Results

### Primary Dispatch (calibrated, all day)

- `corridor_only`: `3.87`
- `time_only_baseline`: `10.51`
- `feasible_count_baseline`: `11.53`
- `walk_aware_rendezvous`: `11.81`
- `rendezvous_only`: `12.10`
- `rendezvous_observable`: `12.19`
- `ml_meeting_point_comparator`: `12.29`

### Sparse High Occlusion Dispatch (calibrated, all day)

- `corridor_only`: `-2.67`
- `time_only_baseline`: `-0.97`
- `feasible_count_baseline`: `0.37`
- `walk_aware_rendezvous`: `0.41`
- `rendezvous_only`: `0.57`
- `rendezvous_observable`: `0.57`
- `ml_meeting_point_comparator`: `0.47`

### Sparse High Occlusion Dispatch, Morning Peak

- `corridor_only`: `-4.85`
- `time_only_baseline`: `-3.29`
- `feasible_count_baseline`: `-2.84`
- `walk_aware_rendezvous`: `-2.35`
- `rendezvous_only`: `-2.53`
- `rendezvous_observable`: `-2.08`
- `ml_meeting_point_comparator`: `-2.79`

Interpretation:

- the systems story survives the robustness pass
- in the morning hard regime, observability-aware valuation again provides the strongest deterministic result

## Urban-Context Ablation

The cleanest observability-specific evidence is still the urban-context ablation in all-day sparse high occlusion:

- `rendezvous_only`: `5.52` with context, `5.05` without
- `rendezvous_observable`: `5.72` with context, `5.19` without
- `ml_meeting_point_comparator`: `5.80` with context, `5.20` without

Interpretation:

- urban context helps the hard-regime ranking
- this is stronger evidence than claiming the current component weights are individually optimal

## Boundary Cases

The hard-regime story should remain disciplined because the boundary cases are mixed:

- in `very_sparse_low_occlusion`, `walk_aware_rendezvous` and the ML comparator slightly edge the deterministic observability policy
- in `very_sparse_extreme_occlusion`, `rendezvous_only` slightly edges `rendezvous_observable`

Those are not failures. They show that the observability layer is not universally dominant, which is exactly why the paper should emphasize difficult urban regimes rather than make a universal claim.

## ML Comparator Status

The ML comparator remains secondary.

Current held-out diagnostics from [rendezvous_meeting_point_metrics_yellow.json](K:\Kamuit\Uber_Logic\Research_paper_2\models\rendezvous_meeting_point_metrics_yellow.json):

- validation observed ROC AUC: `0.6543`
- test observed ROC AUC: `0.5641`

Interpretation:

- it is competitive enough to retain as a comparison
- it is still not strong enough to become the paper's main method

## Best Current Paper Framing

The strongest honest headline is:

`Routes should be evaluated by feasible rendezvous opportunities, and an observability-aware version of that evaluation is especially valuable in sparse, high-occlusion regimes. That hard-regime advantage survives stronger baselines, urban-context ablation, and time-slice robustness checks.`
