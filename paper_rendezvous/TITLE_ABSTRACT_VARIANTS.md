# Title and Abstract Variants

This memo captures journal-style title and abstract options after the final submission polish pass. The current manuscript title remains the most balanced default, but the alternatives below are ready if we want a slightly more systems-oriented or method-oriented emphasis.

## Recommended Title

`Rendezvous-Aware Route Choice for Urban Ride-Pooling Under Occlusion`

Why it works:

- keeps the paper's central object, `route choice`, upfront
- preserves the rendezvous framing that differentiates the paper
- keeps `under occlusion` as a qualifier rather than overclaiming perception

## Title Alternatives

### More systems-oriented

`Route Choice via Feasible and Observable Rendezvous Opportunities in Urban Ride-Pooling`

Use this if we want the primitive shift to appear even more explicitly in the title.

### More compact

`Rendezvous-Aware Routing for Urban Ride-Pooling Under Occlusion`

Use this if the target venue prefers shorter titles, though it is a little less explicit about the route-choice decision setting.

### More mechanism-oriented

`From Corridor Exposure to Observable Rendezvous Opportunities in Urban Ride-Pooling`

Use this only if we want the title to foreground the conceptual shift instead of the decision problem.

## Recommended Abstract

Route choice in ride-pooling is often valued through corridor exposure alone: a route is considered attractive if it passes near more potentially compatible riders. This paper argues that such a view is incomplete in dense urban settings, where riders who appear near a route may still lack a practical, reliable common pickup opportunity. We therefore study route choice through route-induced rendezvous opportunity sets rather than through near-route riders alone. The study combines New York City taxi demand, Open Source Routing Machine alternatives, a hexagonal hierarchical spatial grid, route-anchor meeting opportunities, and an urban-context proxy built from official street centerline, sidewalk, building-footprint, land-use, and elevation layers. Feasible meeting opportunities are filtered by walk accessibility, timing, and remaining trip value, while observable opportunities are discounted by a calibrated pickup-success model under occlusion. Evaluation includes a controlled single-driver study, a secondary rolling-dispatch validation, matched within-driver route-pair analysis, Green-domain transfer, and curated local-geometry case studies. Corridor-only valuation is consistently weakest and exhibits a much larger nominal-versus-realized gap than the rendezvous-aware policies. The main gain comes from replacing corridor exposure with feasible rendezvous opportunities, while observability-aware valuation is most useful in sparse, high-occlusion regimes and in a harder morning-peak slice. The central conclusion is therefore not that observability dominates universally, but that route choice should be evaluated through feasible and observable rendezvous opportunities rather than through proximity alone.

## Shorter Abstract Variant

Route choice in ride-pooling is commonly valued by corridor exposure: a route is considered useful if it passes near more riders. This paper argues that such a view is incomplete in dense urban settings, where nearby riders may still lack a practical or reliable common pickup opportunity. We therefore model route value through route-induced rendezvous opportunity sets rather than through near-route riders alone. The approach combines New York City taxi demand, route alternatives, route-anchor meeting opportunities, and an urban-context observability proxy derived from official street, sidewalk, building, land-use, and elevation layers. Feasible meeting opportunities are filtered by walk accessibility, timing, and remaining trip value, while observable opportunities are discounted by pickup reliability under occlusion. Across controlled single-driver experiments, rolling-dispatch validation, matched within-driver route-pair analysis, Green-domain transfer, and curated local-geometry case studies, corridor-only valuation is consistently weakest and exhibits a much larger nominal-versus-realized gap than the rendezvous-aware policies. The principal gain comes from replacing corridor exposure with feasible rendezvous opportunities, while observability-aware valuation is most useful in sparse, high-occlusion regimes rather than uniformly in all settings.

## Positioning Reminder

For a Q1-style submission, the safest contribution hierarchy is:

1. feasible rendezvous opportunities are the correct primitive
2. observability changes route value most in hard urban regimes
3. ML is a secondary comparator, not the paper's central claim
