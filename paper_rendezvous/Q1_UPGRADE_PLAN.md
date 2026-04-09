# Q1 Upgrade Plan

This memo turns the Q1 push into an execution checklist.

## Goal

Move the paper from:

- strong Q2 with a good idea

to:

- lower-end / borderline Q1 with a stronger evidence package and tighter claims

## What Is Already Done

- fresh Paper 2 branch and codebase split
- repaired corridor, dispatch, and urban-context pipeline
- official NYC urban-context ingestion
- stronger baselines:
  - `time_only_baseline`
  - `feasible_count_baseline`
  - `walk_aware_rendezvous`
- bootstrap confidence intervals
- paired policy deltas
- no-urban-context ablation
- focused hard-regime observability sensitivity runs

## What The Evidence Supports Right Now

1. corridor-only route valuation is the wrong primitive
2. rendezvous feasibility is the main jump
3. observability-aware valuation is especially useful in sparse, high-occlusion settings
4. that story still holds against stronger baselines

## What Still Keeps This From A Clean Q1

1. the gain of `rendezvous_observable` over `rendezvous_only` is still modest
2. the observability proxy is still handcrafted and only partly calibrated
3. the ML comparator is too small and too weak to elevate the paper
4. the current component-weight ablations are mixed rather than uniformly supportive

## Priority Next Steps

### P0: Manuscript tightening

- rewrite the method section around the stronger baseline set
- use the hard-regime story as the headline
- state clearly that observability is regime-dependent, not universally dominant
- keep ML secondary

### P1: Better observability calibration

- calibrate observability weights on non-test data
- compare calibrated weights to the current equal-weight proxy
- keep the test split untouched for final reporting

### P2: Stronger robustness

- add Green as a robustness-only dataset
- add at least one time-slice split or borough-level slice
- report whether policy ordering survives these shifts

### P3: Better real-world grounding

- add qualitative map cases where corridor exposure is similar but rendezvous quality differs
- optionally audit a small set of route-anchor neighborhoods manually

## Recommended Final Figure Set

1. concept figure
2. main primary comparison
3. nominal-vs-realized gap
4. dispatch validation
5. strong-baseline comparison with intervals
6. urban-context ablation
7. sensitivity / boundary-case figure

## Recommended Final Table Set

1. policy definitions
2. scenario defaults
3. primary and sparse-high-occlusion results with stronger baselines
4. paired deltas and bootstrap intervals
5. robustness / limitations summary
