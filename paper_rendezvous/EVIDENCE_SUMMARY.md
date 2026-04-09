# Evidence Summary

This note summarizes the current state of the Paper 2 evidence on branch `codex/rendezvous-aware-route-choice-under-occlusion`.

## Main Takeaways

- `corridor_only` is consistently the weakest policy in both the controlled single-driver study and the rolling dispatch study.
- `rendezvous_observable` is the strongest deterministic policy in the `sparse_high_occlusion` regime and in primary dispatch.
- The ML meeting-point comparator is competitive and often strongest overall, but it should be framed as a comparator rather than the main contribution.
- The observability-aware deterministic policy is not uniformly best in every sparse regime; in the `very_sparse_extreme_occlusion` single-driver setting, `rendezvous_only` is slightly better on mean actual profit.

## Single-Driver Results

### Primary (`density=100`, `occlusion_lambda=0.25`)

- `corridor_only`: `1.3126`
- `rendezvous_only`: `1.5597`
- `rendezvous_observable`: `1.6695`
- `ml_meeting_point_comparator`: `1.7421`

Interpretation: all rendezvous-aware policies beat the corridor baseline. The observability-aware deterministic policy beats the nominal rendezvous baseline by `+0.1098` mean actual profit.

### Sparse High Occlusion (`density=25`, `occlusion_lambda=0.40`)

- `corridor_only`: `-4.9404`
- `rendezvous_only`: `-4.7449`
- `rendezvous_observable`: `-4.6847`
- `ml_meeting_point_comparator`: `-4.7346`

Interpretation: this is the cleanest regime for the paper's core claim. The observability-aware deterministic policy is best and improves over `corridor_only` by `+0.2557` and over `rendezvous_only` by `+0.0601`.

### Very Sparse Low Occlusion (`density=10`, `occlusion_lambda=0.10`)

- `corridor_only`: `-6.9453`
- `rendezvous_only`: `-6.7462`
- `rendezvous_observable`: `-6.7519`
- `ml_meeting_point_comparator`: `-6.7205`

Interpretation: simpler rendezvous-aware methods remain competitive when occlusion is light. This is useful negative evidence and should stay in the paper.

### Very Sparse Extreme Occlusion (`density=10`, `occlusion_lambda=0.55`)

- `corridor_only`: `-7.2515`
- `rendezvous_only`: `-7.0842`
- `rendezvous_observable`: `-7.1123`
- `ml_meeting_point_comparator`: `-7.1017`

Interpretation: all rendezvous-aware methods still beat `corridor_only`, but the deterministic observability-aware policy is not the best variant here. This regime should be discussed honestly as mixed evidence.

## Dispatch Results

### Primary (`density=100`, `occlusion_lambda=0.25`)

- `corridor_only`: `-2.4024`
- `rendezvous_only`: `-2.1214`
- `rendezvous_observable`: `-2.0133`
- `ml_meeting_point_comparator`: `-2.0332`

Interpretation: in dispatch, `rendezvous_observable` is the best deterministic policy in the primary regime.

### Sparse High Occlusion (`density=25`, `occlusion_lambda=0.40`)

- `corridor_only`: `-6.8675`
- `rendezvous_only`: `-6.6798`
- `rendezvous_observable`: `-6.6507`
- `ml_meeting_point_comparator`: `-6.6700`

Interpretation: this is the strongest systems-level evidence for the paper. The observability-aware deterministic policy is best and improves over `corridor_only` by `+0.2167` mean profit per driver.

## Honest Limits

- The ML comparator is trained on a relatively small rendezvous dataset and should not be oversold.
- The observability-aware deterministic policy is strongest in the moderate sparse/high-occlusion regime, but not uniformly best in the most extreme sparse setting tested so far.
- Route coverage is still limited by available OSRM fetches and cache growth. Larger fetched route sets would improve confidence.
- The current manuscript should claim that observability-aware rendezvous valuation is especially helpful in hard urban regimes, not that it dominates in every regime.

## Recommended Headline

Use `sparse_high_occlusion` as the main headline regime, with `primary` and `very_sparse_*` scenarios as supporting context and boundary cases.
