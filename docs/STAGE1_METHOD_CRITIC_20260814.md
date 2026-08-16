# Stage-One Independent-Style Method Critic

**Date:** 2026-08-14<br>
**Scope:** development exploration and proposed held-out confirmation only<br>
**Scientific status:** unresolved; no formal or shaping run authorised

## 1. Evidence inventory

The review used the versioned development configuration and protocol, 840
harmonised evaluation episodes, 84 policy/checkpoint cells, the two-seed dense
development sweep, reward reconstruction checks, half/nominal/double-margin
sensitivity, 20,000 paired evaluation-seed bootstrap replicates, checkpoint
maps, six complete real-time videos and the interrupted-run tensor comparison.
Proposal_G6 was treated as the prescribed design; the revision directory was
treated as the actual design. These sources were not silently reconciled.

## 2. Reconstruction of the two-hour exploration

Before exploration, the main proposal was to preserve all historical files and
implement `development weight selection -> held-out proxy stress test ->
shaping`. The user then clarified that stage one must instead test whether a
reasonable coefficient range can create reproducible high-proxy/poor-diagnostic
behaviour, while stage two later adds a distinct constraint rather than merely
returning the coefficient to 0.5.

The earlier exploration proceeded as follows:

1. A 100k screen covered five coefficients and two development seeds; `0.25`
   was provisionally selected.
2. A 300k comparison of `0.5` and `0.25` did not satisfy the cross-seed domain
   rule.
3. An expanded 300k run added `0.375` and `0.125`; `0.125` produced an
   exploratory signal.
4. That four-policy task was interrupted after 50k/100k checkpoints when a
   pause request was misread as a process-stop request. A fresh output root was
   used for the restart; all four policies later reached 300k.
5. The unresolved interval between `0.125` and `0.25`, complete real-time video,
   explicit practical margins and episode-level uncertainty were not finished
   during the two-hour window.

The current run closed those engineering gaps by adding `0.15625`, `0.1875`
and `0.21875`, harmonising all old models with the corrected metric code and
rendering six fixed, complete trajectories. It deliberately did not start
held-out formal training or shaping.

## 3. Retain, modify, reject or defer

| Decision | Earlier element | Reason |
|---|---|---|
| Retain | Ant-v5, PPO and `ctrl_cost_weight` as the only manipulated variable | Directly matches the bounded mechanism question and avoids a factorial confound. |
| Retain | Candidate and reference rescored under the same candidate `R_w` | Raw returns from different reward formulae are not commensurable. |
| Retain | Training seed as the replication unit; evaluation episodes nested within policy | Prevents pseudo-replication. |
| Retain | 300k endpoint plus six 50k-spaced checkpoints | Endpoint answers the primary question; checkpoints describe optimisation dynamics. |
| Retain | Reward decomposition and disaggregated behavioural diagnostics | No defensible scalar true-performance utility has been established. |
| Modify | Coarse weight grid | The pre-run dense amendment evenly subdivided `[0.125,0.25]`; it revealed non-monotonic candidate status. |
| Modify | Any positive diagnostic delta counted as harm | Replaced by transparent practical margins and 0.5x/1x/2x sensitivity. |
| Modify | Two action indicators counted as separate harms | Saturation and roughness form one command-quality domain and both must cross their margins. |
| Reject for stage one | Select a best coefficient under a fixed `R_0.5` and then stress it | Answers reward-tuning reliability, not the present existence-and-localisation hypothesis. Preserve as historical design material only. |
| Reject | Optimise the reward coefficient by PPO gradient descent | PPO optimises policy parameters for fixed reward coefficients; the coefficient is an experimental variable, not a policy-network parameter. |
| Reject | Collapse diagnostics into an invented `true_performance` score | The weights of that score would introduce an unvalidated utility function and could recreate the same misspecification problem. |
| Defer | All shaping conditions and parameter selection | Stage two is a separate mitigation experiment and must not be tuned on unfinished stage-one evidence. |
| Defer | Higher-than-reference coefficients | They are not required for the targeted under-penalisation mechanism, but their omission must remain a scope limitation and Proposal_G6 deviation. |

## 4. Findings by severity

### Blockers before protocol freeze

**B1. Practical margins have no external safety validation.** The margins stop
tiny numerical changes being called harms, but they are study decisions rather
than Ant-v5 or hardware safety standards. Candidate membership changes from
`0.21875, 0.125` at 1x to `0.21875, 0.15625, 0.125` at 0.5x and none at 2x.
The group or supervisor must accept, revise or externally justify these margins
before formal outcomes are generated.

**B2. The development response is non-monotonic.** The endpoint sequence is
`no, no, yes, no, no, yes` as the reduced coefficient decreases. This rejects
a simple monotonic-threshold account. A formal design may test the same discrete
status pattern, but it cannot claim a unique threshold, discontinuity or phase
transition.

**B3. The provisional four-condition formal rule is now incomplete.** It omits
conditions required to re-evaluate the observed exit and re-entry. Six
conditions (`0.5`, `0.25`, `0.21875`, `0.1875`, `0.15625`, `0.125`) preserve
the development pattern. Smaller designs are permissible only after their lost
comparisons are recorded.

**B4. Five training seeds do not support conventional directional inference.**
Under a fair-sign reference, the one-sided probability of at least 4/5 same
direction is 0.1875. Five seeds are acceptable for a resource-limited,
descriptive student study if every seed and limitation is shown. If a stronger
directional gate is required, eight seeds with at least 7/8 gives 0.03515625
one-sided, still without solving construct validity or multiplicity.

**B5. The actual range conflicts with Proposal_G6.** Proposal_G6 calls for a
small range centred on the reference, whereas the actual range is one-sided
from `0.5` to `0.125`. The actual design is scientifically coherent as a
targeted control-cost under-penalisation study, but the group and supervisor
must accept the changed scope before formal freeze.

### Mandatory claim corrections

**C1. Do not call the outcome true performance.** The project has external
diagnostics but no labelled true reward or validated utility. Permitted wording
is `proxy-diagnostic divergence under the tested Ant-v5/PPO conditions`.

**C2. Gymnasium health is not broad robot health.** The environment checks a
finite state and torso height in `[0.2,1.0]`. Fixed videos show inverted poses
that remain in this height interval. Lateral drift, torso tilt and command
quality must remain separate outcomes.

**C3. Action saturation is command-bound occupancy.** It is not measured motor,
torque, thermal or electrical saturation. Cumulative squared action is a
control-effort proxy, not energy consumption.

**C4. A non-candidate is not evidence of equivalence.** Failing the positive
gate at `0.1875` or `0.15625` does not prove no divergence. The study may report
the topology of the fixed decision rule, not confirmed absence between positive
points.

### Acceptable with stated limitations

- Two development seeds are sufficient for screening and runtime estimation,
  but not confirmation.
- Twenty evaluation seeds in the proposed formal design improve within-policy
  measurement; they do not increase the independent training-run count.
- Same-numbered training seeds are useful matched blocks, but reward-dependent
  trajectories mean they are not exact common-random-number duplicates.
- The common rescore is a disclosed comparison ruler, not a privileged true
  objective. The full cross-rescore matrix appropriately exposes ranking
  sensitivity.
- Six checkpoints are useful for temporal diagnosis, provided they are not
  counted as six independent experiments.
- Complete fixed-seed videos are useful qualitative evidence after numerical
  selection. They cannot replace the quantitative gate.

## 5. Data and provenance adjudication

The first harmonised re-evaluation directory copied target checkpoint labels
into the actual-timestep field. It is retained but excluded. Version 2 reads
`model.num_timesteps`; all 69 other compared fields across 480 rows were
unchanged. The correct actual endpoint is 301,056 steps for every condition.

The interrupted directory contains 17 files and eight partial model
checkpoints. The restart directory contains 49 files and 24 checkpoints and
completed all four policies. ZIP hashes differ, but all eight shared policy
tensor hashes, keys and values are identical, with maximum absolute parameter
difference zero. Therefore there is no evidence that two complete scientific
results disappeared; the interruption lost wall-clock progress beyond 100k,
not a valid 300k endpoint.

## 6. Development findings and uncertainty

At the 300k endpoint, `0.21875` had positive matched proxy advantages of
713.93 and 333.25 and crossed the lateral-control margin in both development
seeds. `0.125` had positive advantages of 449.56 and 199.38 and crossed the
same path-efficiency constituent plus both command-quality constituents in both
seeds. These are development candidates, not formal findings.

The paired evaluation-seed bootstrap shows that the `0.125` action-saturation
and roughness harms are stable across the ten tested perturbations for both
trained policies. By contrast, both `0.21875` lateral-drift intervals cross the
0.50 decision boundary, and the `0.125`, seed-41102 proxy interval crosses
zero. Bootstrap rows remain nested within two trained policies and cannot be
reported as training-seed confidence intervals.

## 7. Adversarial checks

- **Observation that would contradict the preferred explanation:** held-out
  candidate policies fail to show jointly positive proxy advantage and the
  same diagnostic harm, or the pattern reverses under frozen margins.
- **Alternative explanation:** PPO stochasticity produced two unusual policies
  rather than a coefficient-linked mechanism.
- **Alternative explanation:** the practical margins select the narrative;
  the 2x sensitivity already removes every candidate.
- **Alternative explanation:** survival duration and early termination alter
  return, progress and velocity together.
- **Single-seed risk:** `0.125` seed 41102 nearly stops and inverts, whereas seed
  41101 retains forward locomotion; the visual mechanism is not uniform.

## 8. Gate decision

- Engineering validation: ready for final test run.
- Development analysis: complete and auditable.
- Scientific conclusion: unresolved.
- Stage-one protocol freeze: **not ready**.
- Formal training: **blocked** until the four decisions in
  `configs/stage1_formal_confirmation_proposal_v1_20260814.json` are approved.
- Stage-two shaping: deferred and outside this review.
