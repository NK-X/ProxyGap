# Stage-One Development Budget Extension V4

**Status:** authorised development extension; scientifically unresolved. Formal
held-out training and all reward shaping remain prohibited.

## 1. Decision and purpose

The existing `w = 0.5`, `w = 0.21875` and `w = 0.125` development policies for
training seeds 41101 and 41102 will be extended from the saved 300k checkpoint
to a target of one million timesteps. The policy architecture, optimiser,
reward components, environment, seed identifiers, evaluation seeds and absence
of normalisation are unchanged.

This is a budget-sufficiency test, not a new condition search. It asks whether:

1. the weak 300k reference becomes minimally interpretable with more training;
2. the `0.21875` proxy-diagnostic divergence persists under greater optimisation;
3. the different `0.125` path-efficiency and command-quality mechanism persists.

No coefficient will be reselected after inspecting this extension.

## 2. Frozen matrix

| Dimension | Frozen value |
|---|---|
| Environment | Gymnasium `Ant-v5` |
| Algorithm | Stable-Baselines3 PPO, CPU only |
| Control-cost weights | `0.5`, `0.21875`, `0.125` |
| Development training seeds | `41101`, `41102` |
| New checkpoint labels | `500k`, `750k`, `1M` |
| Evaluation seeds | `51101` to `51110`, paired across policies |
| Evaluation policy | Deterministic |
| Episode limit | 1,000 simulator steps |
| Shaping | All shaping weights exactly zero |
| Normalisation | Disabled, as in the source policies |

The training seed, not a checkpoint or evaluation episode, is the independent
replication unit. The two training seeds remain development evidence and cannot
serve as held-out confirmation.

## 3. Frozen PPO implementation

The actor and critic each retain two 64-unit Tanh hidden layers. The observation
dimension is 105 and the action dimension is eight. PPO retains `n_steps=2048`,
`batch_size=64`, ten epochs, learning rate `3e-4`, `gamma=0.99`,
`gae_lambda=0.95`, clip range `0.2`, no entropy bonus, Adam with zero weight
decay, and no state-dependent exploration. Architecture optimisation is outside
this gate because changing it together with budget would confound the cause of
any improvement.

## 4. Checkpoint-continuation deviation

Each source archive records 301,056 actual timesteps because PPO updates in
2,048-step rollouts. Loading the archive preserves policy/value parameters,
optimiser state and the recorded timestep count. It does **not** preserve the
live MuJoCo state or complete pseudorandom-number streams at the instant of the
300k save. Continuation therefore begins from a new environment reset and an
explicit restart of the original training-seed identifier.

Consequently, this run estimates what happens when each saved development
policy receives further optimisation under a reproducible continuation rule. It
is not claimed to be bitwise-equivalent to an uninterrupted 0-to-1M run. This
deviation is acceptable for the present development decision but must not be
hidden in a formal methods section.

## 5. Reference competence gate

At 1M, each `w = 0.5` training seed must satisfy both:

\[
\widehat{p}_{\mathrm{unhealthy}} \leq 0.20,
\qquad
\overline{v}_{x} \geq 0.10\ \text{position units s}^{-1}.
\]

These are transparent project decision thresholds for interpretability. They
are neither literature-derived Ant success constants nor physical robot safety
limits. If either reference seed fails, formal confirmation remains blocked and
a separately frozen baseline-configuration pilot is required. This extension
must not be retroactively reinterpreted by changing the threshold.

## 6. Proxy and diagnostic analysis

For a candidate coefficient `w`, both candidate and same-seed reference
trajectories will be rescored under the same formula:

\[
R_w = \sum_t
\left(r^{\mathrm{forward}}_t + r^{\mathrm{survive}}_t
+ r^{\mathrm{contact}}_t - w\lVert a_t\rVert_2^2\right).
\]

The result remains a matched proxy comparison, not true performance. External
diagnostics stay disaggregated. Absolute lateral drift will be reported beside
exposure-normalised lateral displacement and lateral-path fraction because the
corridor-constrained interpretation is not yet frozen. The analysis will show
every training seed and aggregate the ten evaluation episodes within each
policy.

The 1M checkpoint is the primary extension endpoint. The 500k and 750k
checkpoints describe trajectory over optimisation time but do not add
independent replications. Failure of the earlier divergence to persist is a
valid negative development result and will not trigger a post-hoc coefficient,
margin or seed change.

## 7. Stopping, failure and exclusions

The planned stop is completion and evaluation of all six policies at 1M. A
failed or non-finite policy is retained and reported; it is not replaced with a
new seed. An interrupted partial task is preserved for forensic review because
another restart would add another environment/RNG discontinuity. Source-model
hashes are checked before and after continuation, and source archives are never
overwritten.

## 8. Unresolved accuracy-matrix request

The teaching-assistant description of percentages and error rates is consistent
with a normalised confusion matrix, but the present continuous-control study has
no predeclared ground-truth and predicted classes. The requirement therefore
remains unresolved. The exact clarification required is:

> Do you mean a normalised confusion matrix? If yes, what are the ground-truth
> classes and predicted classes in this reinforcement-learning project?

No artificial classes will be invented merely to produce a familiar matrix.

## 9. Gate after completion

Completion of this run does not automatically authorise a formal experiment. A
post-extension adjudication must separately decide:

1. whether both reference policies pass the competence gate;
2. whether matched proxy advantage and diagnostic harm persist by training seed;
3. whether absolute drift survives the exposure-normalised alternative checks;
4. whether the evidence justifies freezing a held-out condition matrix;
5. whether five or eight new formal training seeds are feasible.

Stage-two shaping remains outside the present scope.
