# ProxyGap Two-Experiment Protocol

**Status:** the development screen is specified; held-out confirmation remains
blocked until a candidate condition and shaping parameters are locked.

## Research logic

Experiment 1 asks whether plausible under-penalisation of Ant-v5 control effort
creates an opportunity for PPO to increase its fixed condition objective while
behaviour outside that objective deteriorates. Experiment 2 asks whether a
bounded intervention targeting the diagnosed behavioural failure reduces that
divergence while the diagnosed `ctrl_cost_weight` remains unchanged.

The design does not promise that divergence will occur. Failure to identify a
candidate in the prespecified range is a valid negative result and stops the
mitigation experiment.

## Experiment 1: detection

The core coefficients are `0.5`, `0.375`, `0.25` and `0.125`. They correspond
to 100%, 75%, 50% and 25% of the documented Ant-v5 default. The historical
`0.0625` condition is a labelled boundary sensitivity check and cannot become
the primary candidate merely because the core produces a negative result.

At the fixed endpoint, each reduced-weight policy is paired with the reference
policy trained under the same seed. Both realised trajectory sets are rescored
using the candidate formula `R_w`. A candidate requires the reduced-weight
policy to score higher under this one fixed proxy in both development seeds and
at least two external diagnostics to deteriorate in the same direction in both
seeds. Returns from different formulas are never directly ranked.

As a secondary optimisation-pressure test, the mean of the 50k and 100k
checkpoints is contrasted with the mean of the 250k and 300k checkpoints within
each fixed coefficient and training seed. This asks whether the same objective
improves over training while diagnostics deteriorate. Failure of this secondary
pattern does not erase a pairwise fixed-proxy divergence, but it prohibits a
claim that extended optimisation amplified the failure.

If several core coefficients qualify, the largest coefficient is selected
because it is the smallest departure from the default.

The primary external diagnostic set is forward path efficiency, lateral drift,
torso tilt, unhealthy termination and action saturation. Net forward progress
and squared action remain important reward-component or mechanism measures, but
are not represented as independent hidden performance.

## Experiment 2: mitigation

The diagnosed coefficient is held fixed. The direct comparison is:

```text
C1: detected coefficient, no shaping
C2: same detected coefficient, bounded behaviour-targeted shaping
```

The shaping signal must match a failure observed consistently during
development. The supported prospective signals are lateral offset and torso
orientation. Forward-reward duplication and action-effort shaping are excluded:
the former directly rewards an existing component, while the latter is too
close to restoring the original control-cost penalty for the present question.

The shaping formula, scale and cap are frozen before held-out training. A
combined lateral/orientation condition is one intervention package and cannot
support component-specific causal claims without ablations.

## Held-out evaluation

Fresh paired training seeds are used for the reference, detected and shaped
conditions. Evaluation episodes are nested within policies and are first
aggregated by training seed. The final 300k checkpoint is primary; earlier
checkpoints describe optimisation dynamics. Results are reported as raw paired
effects and disaggregated diagnostics. Five training seeds do not establish a
universal effect or reliable rare-failure rate.

## Stopping rules

- No qualifying core coefficient: stop before mitigation.
- Only the `0.0625` boundary qualifies: report boundary evidence, but do not
  promote it to the primary plausible-range condition without a protocol
  revision and explicit scope change.
- Shaping improves one metric but worsens another: report failure transfer, not
  successful mitigation.
- No replacement seeds or post-hoc checkpoints are permitted without applying
  a predeclared failure rule to every condition.
