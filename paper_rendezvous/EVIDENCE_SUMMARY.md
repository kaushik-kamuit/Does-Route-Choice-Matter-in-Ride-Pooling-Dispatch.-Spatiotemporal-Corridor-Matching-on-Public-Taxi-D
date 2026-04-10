# Evidence Summary

This note summarizes the current Q1-push evidence on branch `rendezvous-aware-route-choice-under-occlusion`.

## Current Headline

- `corridor_only` is decisively the weakest primitive.
- The main gain comes from moving from corridor exposure to feasible rendezvous opportunities.
- The observability layer is most convincing in the Yellow hard regime, especially once we isolate matched route pairs.
- Green transfer supports the feasible-rendezvous framing, but it does **not** support a universal observability-dominance claim.

## Yellow Primary

Default calibrated all-day results from [rendezvous_policy_summary.csv](K:\Kamuit\Uber_Logic\Research_paper_2\results\rendezvous_policy_summary.csv):

- `corridor_only`: `6.76`
- `time_only_baseline`: `16.03`
- `feasible_count_baseline`: `17.12`
- `walk_aware_rendezvous`: `17.30`
- `rendezvous_only`: `17.60`
- `rendezvous_observable`: `17.57`

Interpretation:

- the paper still should not oversell observability in the easy regime
- the important point is that feasible rendezvous valuation is far stronger than corridor exposure

## Yellow Hard Regime

Default calibrated `sparse_high_occlusion` results:

### All day

- `corridor_only`: `-3.39`
- `time_only_baseline`: `-1.64`
- `feasible_count_baseline`: `-1.01`
- `walk_aware_rendezvous`: `-0.79`
- `rendezvous_only`: `-0.70`
- `rendezvous_observable`: `-0.68`

### Morning peak

- `corridor_only`: `-4.57`
- `time_only_baseline`: `-2.51`
- `feasible_count_baseline`: `-2.25`
- `walk_aware_rendezvous`: `-2.11`
- `rendezvous_only`: `-1.96`
- `rendezvous_observable`: `-1.66`

Interpretation:

- absolute profit is negative because this regime is intentionally hard
- the route-ordering signal is still strong and consistent:
  - `rendezvous_observable > rendezvous_only > corridor_only`

## Pairwise Policy Evidence

From [rendezvous_pairwise_deltas_vs_corridor.csv](K:\Kamuit\Uber_Logic\Research_paper_2\results\rendezvous_pairwise_deltas_vs_corridor.csv):

### Yellow sparse high occlusion, all day

- `rendezvous_observable - corridor_only`: `+2.71`
- 95% CI: `[2.43, 3.00]`

### Yellow sparse high occlusion, morning peak

- `rendezvous_observable - corridor_only`: `+2.91`
- 95% CI: `[2.40, 3.44]`

From [rendezvous_pairwise_deltas_vs_rendezvous_only.csv](K:\Kamuit\Uber_Logic\Research_paper_2\results\rendezvous_pairwise_deltas_vs_rendezvous_only.csv):

### Yellow sparse high occlusion, all day

- `rendezvous_observable - rendezvous_only`: `+0.025`
- 95% CI: `[-0.22, 0.26]`

### Yellow sparse high occlusion, morning peak

- `rendezvous_observable - rendezvous_only`: `+0.30`
- 95% CI: `[-0.17, 0.78]`

Interpretation:

- aggregate policy bars show a clear gain over corridor-only
- aggregate gains over `rendezvous_only` are positive but still modest
- that is why the matched-pair analysis matters

## Matched Observability Isolation

From [rendezvous_observability_matched_summary.csv](K:\Kamuit\Uber_Logic\Research_paper_2\results\rendezvous_observability_matched_summary.csv):

### Yellow sparse high occlusion, all day

- matched higher-observability route minus lower-observability route: `+5.28`
- 95% CI: `[4.21, 6.37]`
- higher-observability win rate: `0.675`

### Yellow sparse high occlusion, morning peak

- matched higher-observability route minus lower-observability route: `+0.89`
- 95% CI: `[0.06, 1.71]`
- higher-observability win rate: `0.585`

Interpretation:

- this is the strongest direct evidence for the observability claim
- once corridor exposure and feasible opportunity count are approximately matched, observability still changes realized value

## Dispatch

From [rendezvous_dispatch_policy_summary.csv](K:\Kamuit\Uber_Logic\Research_paper_2\results\rendezvous_dispatch_policy_summary.csv):

### Yellow primary dispatch

- `corridor_only`: `3.87`
- `rendezvous_only`: `12.10`
- `rendezvous_observable`: `12.19`

### Yellow sparse high occlusion dispatch, all day

- `corridor_only`: `-5.71`
- `rendezvous_only`: `-4.54`
- `rendezvous_observable`: `-4.63`

### Yellow sparse high occlusion dispatch, morning peak

- `corridor_only`: `-6.16`
- `rendezvous_only`: `-5.24`
- `rendezvous_observable`: `-5.03`

Interpretation:

- the systems story survives shared competition even in the harsher density-10 shared-pool setting
- dispatch primarily strengthens the feasible-rendezvous claim; the observability edge is modest and more visible in morning peak than in all-day dispatch

## Urban-Context Ablation

The best observability-specific ablation evidence is still the urban-context comparison:

- sparse-high-occlusion calibrated density-10 slice:
  - `rendezvous_only`: `-0.70` with context vs `-0.96` without
  - `rendezvous_observable`: `-0.68` with context vs `-1.01` without

Interpretation:

- context helps when the route choice is close
- this is stronger evidence than pretending every hand-tuned component weight is independently optimal

## Green Transfer

From [rendezvous_green_policy_summary.csv](K:\Kamuit\Uber_Logic\Research_paper_2\results\rendezvous_green_policy_summary.csv):

### Green primary

- `corridor_only`: `-0.00`
- `rendezvous_only`: `4.32`
- `rendezvous_observable`: `4.31`

### Green sparse high occlusion

- `corridor_only`: `-6.56`
- `rendezvous_only`: `-6.04`
- `rendezvous_observable`: `-6.10`

From [rendezvous_green_dispatch_policy_summary.csv](K:\Kamuit\Uber_Logic\Research_paper_2\results\rendezvous_green_dispatch_policy_summary.csv):

### Green sparse high occlusion dispatch

- `corridor_only`: `-7.18`
- `rendezvous_only`: `-6.82`
- `rendezvous_observable`: `-6.86`

Interpretation:

- Green supports the paper's main primitive shift:
  - corridor-only remains clearly weakest
  - rendezvous-aware valuation transfers
- Green does **not** support a universal observability-win claim

## Curated Case Studies

From [rendezvous_case_study_agreement.csv](K:\Kamuit\Uber_Logic\Research_paper_2\results\rendezvous_case_study_agreement.csv):

- curated route-pair panels: `8`
- rubric agreement with observability-aware preference: `8/8`

The panels are in [paper_rendezvous/figures](K:\Kamuit\Uber_Logic\Research_paper_2\paper_rendezvous\figures):

- [rendezvous_fig2_matched_pair_mechanism.png](K:\Kamuit\Uber_Logic\Research_paper_2\paper_rendezvous\figures\rendezvous_fig2_matched_pair_mechanism.png)
- [rendezvous_fig9_case_studies.png](K:\Kamuit\Uber_Logic\Research_paper_2\paper_rendezvous\figures\rendezvous_fig9_case_studies.png)
- [rendezvous_appendix_case_studies.png](K:\Kamuit\Uber_Logic\Research_paper_2\paper_rendezvous\figures\rendezvous_appendix_case_studies.png)

Interpretation:

- this is a structured local-geometry sanity check, not a user study
- it strengthens the paper by showing that the selected route pairs visibly differ in anchor quality and local context

## ML Comparator

The ML comparator remains secondary.

- it is still competitive in some Yellow slices
- it is not needed for the main claim
- it should stay out of the abstract's central contribution statement

## Best Honest Framing

The strongest honest framing is:

`Routes should be evaluated by feasible rendezvous opportunities, and an observability-aware version of that evaluation is especially valuable in sparse, high-occlusion regimes. That claim is supported by stronger baselines, matched route-pair isolation, Green transfer, and curated local-geometry case studies, without requiring a universal observability-win claim.`
