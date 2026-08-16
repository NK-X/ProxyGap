# Fresh Reference-Only 1M Diagnostic: Post-Run Adjudication V7

**Status:** engineering-validated; reference configuration inconclusive;
candidate-weight, formal and shaping runs remain prohibited.

## 1. Scientific role

This run is a prerequisite diagnostic for stage one. It asks whether the
unchanged default Ant-v5/PPO reference configuration produces minimally
competent locomotion reproducibly across fresh training seeds. It does not test
the stage-one proxy-divergence hypothesis itself because no candidate
`ctrl_cost_weight` was trained.

The intended behaviour, reward proxy and research objective remain separate:

1. intended behaviour: sustained net forward locomotion during a maximum
   1,000-step episode while satisfying Gymnasium's finite-state and torso-height
   health rule;
2. proxy optimised by PPO: Ant-v5 forward reward plus healthy reward and contact
   term, minus control cost with `ctrl_cost_weight = 0.5`;
3. research objective: later determine whether a prospectively declared,
   plausible coefficient range contains policies with similar or higher
   common-scale proxy performance but noticeably worse predeclared behavioural
   diagnostics.

No scalar `true_reward` or `true_performance` is defined.

## 2. Frozen matrix and execution

Five policies were trained independently from initialisation with training
seeds 41201-41205. Each used the frozen V6 PPO settings, CPU execution, no
normalisation, no shaping and nominal checkpoints at 250k, 500k, 750k and 1M.
PPO rollout batching produced 1,001,472 actual timesteps at the final endpoint.

Each checkpoint was evaluated on the same 20 evaluation seeds, 51201-51220.
The 20 episodes are nested repeated observations of a fixed trained policy;
they are not 20 independent training replications. The run produced 20 model
archives and 400 episode-level evaluation rows. Successful attempt 3 required
94.29 minutes of wall time and produced 7,433,764 bytes of run evidence,
including 6,173,691 bytes of models.

Two earlier engineering attempts are retained separately. Attempt 1 was
stopped before any checkpoint after an inadequate outer timeout was identified.
Attempt 2 was terminated by stale tool-host state before any policy completed.
No result was inspected before either restart, no partial policy was resumed,
and attempt 3 restarted all five policies from fresh initialisation under the
unchanged scientific configuration.

## 3. Prospective competence rule

At the 1M checkpoint, each independently trained policy had to satisfy both

\[
\widehat p_{\mathrm{unhealthy}} \leq 0.20,
\qquad
\overline v_x \geq 0.10\;\text{position units s}^{-1}.
\]

The frozen configuration-level rule was: four or five passing policies means
`supported`, two or three means `inconclusive`, and zero or one means `failed`.
This is an operational interpretability screen, not a null-hypothesis test,
physical-safety limit or literature-derived universal threshold.

## 4. Results

| Training seed | Unhealthy termination rate | Mean forward velocity | Joint result |
|---:|---:|---:|---|
| 41201 | 0.00 | 0.918 | Pass |
| 41202 | 0.60 | 0.806 | Fail |
| 41203 | 0.60 | 0.894 | Fail |
| 41204 | 0.00 | 0.910 | Pass |
| 41205 | 0.65 | 1.141 | Fail |

All five policies met the forward-velocity threshold. Only two met the joint
health-and-velocity rule. Therefore, the frozen outcome is **2/5 passing:
inconclusive**. The threshold is not weakened and failed seeds are not replaced.

The observed pattern is compatible with substantial training-seed sensitivity
in the tested reference configuration. It does not by itself identify the
cause, prove that PPO is unsuitable, or establish any reward-misspecification
effect.

## 5. Verification

The completed run contains 400 unique evaluation keys, 20 runtime records and
20 reloadable PPO model archives. Independent standard-library recomputation
from the raw CSV reproduced all five policy decisions and the 2/5
classification without importing the primary analysis functions. Maximum
absolute reconstruction errors were approximately `2.39e-11` for base reward
and `9.49e-6` for control cost. All model hashes and recorded timestep counts
were verified. The complete automated suite passed after analysis finalisation.

## 6. Decision and claim boundary

The reference condition is not sufficiently reproducible to act as a stable
comparator under the frozen rule. Candidate-weight confirmation must therefore
not begin. Formal held-out seeds 42001-42008 remain unused, and reward shaping
remains outside stage one.

The next permissible action is a separately frozen, minimal, one-factor
reference-configuration pilot. Its factor, candidate settings, seeds, endpoint
and decision rule must be selected before outcome inspection. Architecture,
normalisation and optimiser settings must not be changed together because that
would prevent attribution.

This run can support only the following statement:

> Under the tested unchanged Ant-v5/PPO reference configuration, all five fresh
> policies met the operational forward-velocity criterion, but only two met the
> joint health-and-velocity criterion; reference competence was therefore
> inconclusive under the prospectively frozen rule.

It cannot establish reward misspecification, reward hacking, absence of
misspecification, an optimal coefficient, PPO generality or real-robot safety.

## 7. Evidence locations

- frozen design: `configs/stage1_reference_fresh_1m_v6_20260814.json`;
- raw run: `artifacts/exploration/stage1_reference_fresh_1m_v6_20260814`;
- primary and independent analysis:
  `artifacts/analysis/stage1_reference_fresh_1m_v6_20260814`;
- machine-readable outcome:
  `configs/stage1_reference_fresh_1m_outcome_v7_20260814.json`;
- interrupted engineering attempts:
  `artifacts/exploration/stage1_reference_fresh_1m_v6_20260814_attempt1_interrupted`
  and
  `artifacts/exploration/stage1_reference_fresh_1m_v6_20260814_attempt2_interrupted_host_timeout`.
