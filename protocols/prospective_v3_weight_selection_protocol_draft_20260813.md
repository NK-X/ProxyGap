# Prospective v3: Weight Selection and Held-out Proxy Stress Test

**Status:** rejected exploration; not canonical and no training authorised.

This branch was rejected after clarification of the research aim. Proxy-only
weight selection does not implement the intended two-experiment study. The file
is retained solely to preserve the design decision trail.

## Plain-language logic

The study has two separate questions.

1. During development, which coefficient performs best under one fixed and
   predeclared benchmark proxy?
2. After that choice is frozen, does the selected coefficient still produce
   acceptable behaviour on fresh training and evaluation seeds?

The second question must not influence the first. Lateral drift, torso tilt,
unhealthy termination, action saturation and episode length are therefore not
available to the development selector.

## Development stage

The candidate grid is `0.0625`, `0.125`, `0.25`, `0.5` and `1.0`. The fixed
benchmark proxy uses the common control-cost weight `0.5`. Episodes are first
averaged within each training seed; the seed-level estimates are then averaged
across development training seeds. Checkpoint rows before the final checkpoint
are not used for coefficient selection.

The result is called the **best-tested coefficient under the fixed benchmark
proxy**. It is not a global optimum, a unique optimum or a true-performance
optimum. If the selected coefficient is `0.5`, that negative search result is
retained and no alternative candidate may be substituted after inspecting
protected diagnostics.

## Held-out stage

The selected coefficient is frozen before held-out training. The formal core
contains a reference condition at `0.5`, the selected unshaped condition and a
selected bounded shaping condition. Held-out training and evaluation seeds are
disjoint from development seeds and historical formal-v1 seeds.

The primary comparison is descriptive. The study may report whether the fixed
proxy ranking is reproduced and whether unselected behavioural diagnostics
change. It must not claim a universal optimal weight or formal reward hacking
without an independently justified construct and stronger evidence.

## Shaping boundary

Shaping is locked after development selection and before held-out training. A
combined effort/orientation intervention is interpreted as one package unless
effort-only and orientation-only ablations are added. No component-specific
causal claim is permitted for the combined-only design.

## Required audit records

The run must save the candidate grid, selection metric, aggregation rule, seed
partitions, candidate summaries, selected coefficient, tie-break rule and a
machine-readable record showing that protected diagnostics were not supplied to
the selector.
