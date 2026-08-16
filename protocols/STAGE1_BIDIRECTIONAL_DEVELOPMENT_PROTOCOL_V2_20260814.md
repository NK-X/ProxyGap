# Stage-One Bidirectional Coefficient Development Protocol

**Status:** development rules frozen before the upper-weight run; held-out
formal training and reward shaping remain blocked.

## Research scope

This protocol tests whether reward misspecification can be operationally
identified at one or more tested `ctrl_cost_weight` values in Gymnasium
`Ant-v5` with PPO. Stage one detects proxy-diagnostic divergence only. It does
not introduce shaping and does not claim to recover a scalar true reward.

The intended behaviour is sustained and effective net forward locomotion over
a maximum 1,000-step episode while the Ant remains within the environment's
finite-state and torso-height health rule. Path efficiency, lateral control,
torso orientation and action-command quality remain disaggregated guardrails.

## Primary hypothesis

Within the predeclared tested coefficient range, at least one policy trained at
a non-reference `ctrl_cost_weight` will have proxy performance that is no worse
than the paired `w=0.5` reference by more than the frozen non-inferiority margin,
while at least one predeclared behavioural domain deteriorates by its practical
margin in both development training seeds.

This is a bidirectional existence hypothesis. It does not assume in advance
that only reduced control-cost weights can produce divergence. A coefficient is
a tested point, not a continuous neighbourhood or critical value.

## Coefficient map

Existing development evidence covers `0.5`, `0.375`, `0.25`, `0.21875`,
`0.1875`, `0.15625` and `0.125`. The new upper-side conditions are `0.625` and
`0.75`, representing 25% and 50% increases relative to the reference. Values
`1.0` and `2.0` are excluded from this development extension because they may
primarily test inactivity under strong action suppression rather than a modest
local coefficient change. That exclusion is a scope decision, not evidence
that those values are invalid.

All newly trained policies use training seeds `41101` and `41102`, evaluation
seeds `51101` to `51110`, 300,000 timesteps and checkpoints at 50,000-step
intervals. The endpoint at 300,000 timesteps is primary. Earlier checkpoints
describe onset and persistence but are not independent replications and are not
required to show continuous divergence.

## Matched proxy comparison

For each candidate coefficient `w`, candidate and same-seed reference
trajectories are rescored under the same formula:

```text
R_w = sum_t (forward_t + survive_t + contact_t - w * ||a_t||_2^2).
```

Condition-specific returns produced by different reward formulae are never
directly ranked. The primary proxy gate is relative non-inferiority:

```text
candidate_R_w - reference_R_w >= -0.05 * abs(reference_R_w).
```

The 5% value is a transparent project-operational margin, not an externally
validated universal equivalence boundary. Strict positive proxy gain is always
reported separately. Results are repeated at 0%, 2.5% and 5% proxy margins so
that classification dependence is visible.

## Behavioural guardrails

The frozen development margins remain:

- net forward progress lower by at least 1.0 position unit;
- forward path efficiency lower by at least 0.10;
- unhealthy-termination rate higher by at least 0.20;
- mean absolute lateral drift higher by at least 0.50 position unit;
- torso-tilt RMS higher by at least 0.0872664626 rad (5 degrees);
- action saturation and normalised action roughness both higher by at least
  0.02 for the command-quality domain.

These are smallest effects of interest for this simulation study, not physical
robot safety limits. Half-margin and double-margin sensitivity analyses remain
mandatory.

## Candidate and stopping rules

The primary development classification requires proxy non-inferiority and the
same harmed metric or metric combination in both training seeds. The strict
positive-proxy classification is retained as a stronger secondary result.

At most one lower-side and one upper-side candidate may be nominated. On each
side, select the qualifying coefficient with the smallest absolute departure
from `0.5`. This rule prevents selection of a more extreme coefficient merely
because it produces a more dramatic result.

If no upper-side condition qualifies, retain that negative development result;
do not add a larger coefficient after inspecting the outcomes. If no condition
qualifies on either side, formal confirmation remains blocked. Missing data,
non-finite metrics, technical reruns and videos follow the existing stage-one
exclusion rules. No seed may be replaced because its outcome is inconvenient.

## Formal freeze gate

Fresh formal training may begin only after:

1. all upper-run cells and model checkpoints pass provenance and schema checks;
2. proxy-margin and diagnostic-margin sensitivities are reported;
3. the reference minimum-competence audit is completed;
4. the formal condition-selection rule has been applied without changing it;
5. the formal seed count, exact seed identifiers and run-time estimate are
   frozen in a new versioned configuration;
6. complete-episode video selection is prespecified;
7. the final protocol and configuration hashes are recorded.

Even a positive held-out result will concern only the tested Ant-v5, PPO,
coefficient range, training budget, diagnostics and seed-generating procedure.
It will not establish a globally optimal reward, a universal reward-hacking
threshold or real-robot safety.
