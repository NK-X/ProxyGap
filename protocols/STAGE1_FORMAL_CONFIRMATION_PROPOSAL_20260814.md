# Stage-One Held-Out Confirmation Proposal

**Status:** blocked pending user and supervisor decisions; do not run<br>
**Scope:** stage-one detection only; reward shaping is excluded

## Research question and estimand

The stage-one question is whether a reduced control-cost coefficient within the
tested plausible range can produce higher matched proxy return together with a
practically meaningful deterioration in a predeclared behavioural diagnostic.

The primary estimand is the proportion of independently trained policies, under
the declared training-seed procedure, that jointly satisfy both conditions at
the 300k endpoint:

\[
\Delta R_w = \overline{R_w(\pi_w)}-\overline{R_w(\pi_{0.5})}>0
\]

and at least one frozen diagnostic-domain margin is crossed. Candidate and
reference trajectories are scored with the same candidate formula \(R_w\).
Evaluation episodes are first aggregated within each trained policy.

## Why six conditions are recommended

The development status sequence was:

`0.375: no`, `0.25: no`, `0.21875: yes`, `0.1875: no`,
`0.15625: no`, `0.125: yes`.

The recommended formal matrix is therefore:

`0.5`, `0.25`, `0.21875`, `0.1875`, `0.15625`, `0.125`.

The `0.375` context point is omitted because `0.25` is the adjacent upper
non-candidate for the first onset. Removing `0.1875` loses the immediate exit;
removing `0.15625` loses the upper side of the re-entry bracket. A four- or
five-condition design remains possible only if that loss is explicitly
accepted before held-out outcomes are seen.

Failure to meet a positive gate at an intermediate condition does not prove
equivalence or absence of divergence. The formal result may describe the
candidate-status topology under the frozen rule, but it may not infer a
continuous phase transition or unique critical coefficient.

## Replication decision still required

Two non-overlapping options are recorded in the versioned configuration:

| Plan | Training seeds | Joint directional gate | Interpretation | Measured wall-clock estimate |
|---|---:|---:|---|---:|
| Resource-limited | 5 | at least 4/5 | descriptive replication | about 232 min |
| Stronger directional | 8 | at least 7/8 | stronger directional consistency | about 348 min |

Under an independent fair-sign reference, \(P(X\geq4\mid n=5)=0.1875\), so
4/5 is not a conventional significance result. For eight seeds,
\(P(X\geq7\mid n=8)=0.03515625\) one-sided. These calculations do not validate
the behavioural constructs, solve multiplicity or turn the seed procedure into
a population-sampling guarantee.

## Fixed measurements

- matched proxy advantage under each candidate \(R_w\);
- net forward progress and translational mean forward velocity;
- forward path efficiency;
- unhealthy termination and its category;
- mean absolute lateral drift;
- torso-tilt RMS;
- action-bound occupancy and normalised action-command roughness;
- episode length, termination and time-limit truncation;
- signed reward decomposition.

Gymnasium health is finite state plus torso height in `[0.2,1.0]`. It is not a
synonym for good posture: the fixed videos show that an inverted Ant can remain
inside this height interval. Torso orientation therefore remains a separate
diagnostic.

## Analysis rules

1. The independent unit is one training seed; checkpoints and evaluation
   episodes are nested observations.
2. A seed counts only when positive matched proxy advantage and the diagnostic
   margin occur jointly for that seed.
3. The same constituent diagnostic metric must pass the cross-seed gate. The
   command-quality domain requires both saturation and roughness margins.
4. Repeat the entire decision table at half, nominal and double margins.
5. The 300k endpoint is primary. The 200k/250k/300k window is descriptive and
   cannot create extra independent sample size.
6. All policies, failures and exclusions are reported. A valid but inconvenient
   seed is never replaced.
7. Complete videos use fixed evaluation seeds and are generated after the
   numerical result. They support interpretation only.

## Formal launch blockers

- approval of the one-sided range as a targeted under-penalisation mechanism
  study rather than Proposal_G6's centred sweep;
- approval or revision of the practical margins;
- selection of the five- or eight-training-seed plan;
- approval of the six-condition matrix or explicit acceptance of a reduced
  matrix's information loss.

Until these four decisions are recorded in a new frozen configuration, formal
training remains prohibited.
