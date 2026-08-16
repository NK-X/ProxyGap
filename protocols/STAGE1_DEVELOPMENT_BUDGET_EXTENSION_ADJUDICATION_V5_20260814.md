# Stage-One Development Budget Extension: Post-Run Adjudication V5

**Status:** engineering-validated development evidence; scientifically
unresolved; formal held-out training and reward shaping remain prohibited.

## 1. Scope and frozen question

This adjudication concerns stage one only. It evaluates whether a plausible
range of Ant-v5 control-cost coefficients contains a policy for which a
common-scale proxy score is higher while one or more predeclared behavioural
diagnostics are materially worse. It does not evaluate reward shaping.

The frozen V4 extension continued the existing `w = 0.5`, `0.21875` and
`0.125` development policies for training seeds 41101 and 41102 from their
301,056-step archives to nominal 500k, 750k and 1M checkpoints. Ten paired
evaluation seeds were used at every checkpoint. All shaping terms remained
zero. Checkpoints and evaluation episodes are repeated measurements nested
within a trained policy; the independent development units remain the two
training seeds.

## 2. Execution and data integrity

All six continuations completed. PPO rollout batching produced actual
checkpoints of 501,760, 751,616 and 1,001,472 timesteps. The run produced 18
model archives and 180 episode-level evaluation rows in 4,469.7 seconds
(74.50 minutes) of wall time with four CPU workers.

Quality checks found no duplicate episode keys, missing decision metrics,
non-finite decision metrics, failed policies or non-zero shaping terms. Reward
reconstruction errors were at most approximately `1.44e-6`, below the existing
`1e-3` CSV contract. All six source-model hashes remained unchanged.

An independent verifier used standard-library recomputation from the raw CSV
without importing the primary analysis functions. It reproduced the reference
gate, endpoint contrasts, six source hashes, and the hashes and timestep counts
of all 18 continued models. The complete automated suite passed all 77 tests.

## 3. Reference competence gate

The V4 protocol required each reference policy at 1M to satisfy

\[
\widehat p_{\mathrm{unhealthy}} \le 0.20,
\qquad
\overline v_x \ge 0.10\;\text{position units s}^{-1}.
\]

| Training seed | Unhealthy termination rate | Mean forward velocity | Joint result |
|---:|---:|---:|---|
| 41101 | 0.90 | 1.166 | Fail |
| 41102 | 0.60 | 1.075 | Fail |

Both policies passed the forward-velocity requirement and failed the health
requirement. The threshold was frozen before the extension and must not be
weakened after seeing these results. A formal reference condition is therefore
not yet validated.

## 4. Candidate adjudication at the primary 1M endpoint

For each candidate coefficient `w`, the candidate and same-training-seed
reference trajectories were rescored using the same proxy:

\[
R_w(\tau)=\sum_t\left(
r_t^{\mathrm{forward}}+r_t^{\mathrm{survive}}+r_t^{\mathrm{contact}}
-w\lVert a_t\rVert_2^2\right).
\]

For `w = 0.21875`, the matched proxy advantage was positive in both development
seeds: `+324.26` and `+99.47`. In those same paired comparisons:

| Seed | Change in net progress | Change in path efficiency | Change in torso-tilt RMS | Change in unhealthy termination rate |
|---:|---:|---:|---:|---:|
| 41101 | -3.78 | -0.189 | +1.463 rad | -0.70 |
| 41102 | -8.06 | -0.283 | +1.003 rad | -0.40 |

Under the frozen screen, `w = 0.21875` is therefore a strong development
candidate: matched proxy performance is higher while locomotion effectiveness
and posture stability are worse in both seeds. The candidate is simultaneously
less likely to terminate as unhealthy. This is a disaggregated multi-objective
trade-off. Without a validated scalar true reward or a predeclared priority
ordering over all diagnostics, it cannot be called uniformly lower overall
performance.

For `w = 0.125`, the matched proxy differences were `-282.12` and `-354.65`.
Although several diagnostics were worse, this condition failed the high-or-
similar-proxy gate at 1M. It remains a negative construct check rather than the
primary candidate.

## 5. Optimisation trajectory

The checkpoint pattern is non-monotonic. The `0.21875` condition qualified at
300k, did not qualify at 500k, had proxy gain without a consistently harmed
domain at 750k, and qualified again at 1M. The evidence therefore supports an
endpoint development candidate, not continuous or monotonic amplification
throughout training. Checkpoints must not be counted as independent
replications.

At 300k the `0.21875` interpretation depended primarily on absolute lateral
drift, for which longer exposure was a plausible alternative explanation. At
1M the common harmed domains are forward progress/path efficiency and torso
tilt, so the endpoint candidate no longer depends solely on that disputed
absolute-drift measure. Lateral diagnostics remain reported rather than
silently discarded.

## 6. Continuation limitation

The continuation preserved policy parameters, value parameters, optimiser
state and recorded timestep count, but not the exact live MuJoCo state or full
pseudorandom stream at the 300k save. It restarted each continuation with the
original training-seed identifier. Consequently, the run is a reproducible
policy-continuation experiment, not a bitwise reconstruction of uninterrupted
0-to-1M training.

## 7. Claim decision

The present evidence may support the following development statement:

> Within the tested Ant-v5/PPO development setting, `ctrl_cost_weight =
> 0.21875` was a reproducible candidate at the 1M endpoint: both trained
> policies achieved higher return under the same candidate proxy than their
> paired `0.5` references, while forward locomotion effectiveness and torso
> posture stability were worse.

It does not yet support claims of confirmed reward hacking, a scalar true
reward, globally low performance, a unique critical coefficient, monotonic
overoptimisation, a globally optimal coefficient, or generalisation beyond the
tested simulator, algorithm, budget and seed set.

## 8. Blocking issues and next gate

Formal protocol freeze remains blocked because:

1. both reference policies failed the predeclared health component of the
   competence gate;
2. the two training seeds are development data selected and inspected during
   candidate discovery, not held-out confirmation;
3. the final held-out condition matrix and replication count remain unfrozen;
4. the course requirement described as an `accuracy matrix` remains undefined
   for this continuous-control task.

The preferred next stage-one test is a separately frozen, fresh,
uninterrupted, reference-only 1M development replication. Its purpose is to
distinguish persistent baseline-config weakness from continuation discontinuity
or two-seed variation. If that reference-only test also fails, a separate
baseline-configuration pilot may examine documented normalisation or other
baseline settings. It must not be mixed with the present V4 evidence.

No shaping run should begin until stage-one reference competence and held-out
confirmation logic are resolved.

## 9. Accuracy-matrix clarification

A conventional accuracy or confusion matrix requires ground-truth classes and
predicted classes. The current PPO task outputs continuous actions and
continuous diagnostics, so those classes do not presently exist. The exact
question for the teaching assistant is:

> Do you mean a normalised confusion matrix? If yes, what are the ground-truth
> classes and predicted classes in this reinforcement-learning project?

Artificial classes must not be introduced solely to satisfy the appearance of
a classification deliverable.

## 10. Evidence locations

- frozen V4 config: `configs/stage1_development_budget_extension_v4_20260814.json`;
- raw 1M evaluation rows: `artifacts/exploration/stage1_budget_extension_1m_v4_20260814/logs/evaluation_metrics.csv`;
- primary analysis: `artifacts/analysis/stage1_budget_extension_1m_v4_20260814_attempt2`;
- independent result: `independent_verification.json` in the primary analysis directory;
- machine-readable V5 decision: `configs/stage1_post_extension_gate_v5_20260814.json`.
- Chinese PDF addendum: `output/pdf/ProxyGap_Stage1_1M_Extension_Adjudication_20260814_CN.pdf`.
- final hash inventory: `artifacts/analysis/stage1_budget_extension_1m_v4_20260814_attempt2/FINAL_SHA256_MANIFEST_20260814.csv`.
