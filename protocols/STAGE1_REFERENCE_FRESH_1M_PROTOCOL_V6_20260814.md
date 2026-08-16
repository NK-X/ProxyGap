# Stage-One Fresh Reference Replication Protocol V6

**Status:** user-authorised development diagnostic; frozen before outcome
inspection; not a formal held-out experiment; reward shaping prohibited.

## 1. Decision addressed

The previous V4 continuation produced a promising stage-one
proxy-diagnostic candidate at `ctrl_cost_weight = 0.21875`, but both default
`0.5` reference policies failed the predeclared health component of the
reference-competence gate. The two policies had been continued from saved 300k
models, without restoration of the exact MuJoCo state or complete random-number
stream. Those facts leave two materially different explanations:

1. the unchanged reference configuration is persistently unreliable at the
   present 1M budget; or
2. the two observed failures reflect continuation discontinuity and/or
   variation in only two development training seeds.

V6 tests this decision before any candidate confirmation. It does not retune
the reward, select a candidate coefficient, or evaluate shaping.

## 2. Research question and scope

> Can the unchanged default Ant-v5/PPO reference configuration produce
> minimally competent locomotion in fresh, uninterrupted 0-to-1M training
> runs?

The result is conditional on one simulator (`Ant-v5`), one PPO implementation,
the recorded architecture and hyperparameters, CPU execution, a one-million
timestep budget and the five declared training seeds. It is not evidence of a
globally optimal PPO configuration or general continuous-control reliability.

## 3. Frozen matrix

| Element | Frozen value |
|---|---|
| Condition | Reference only, `ctrl_cost_weight = 0.5` |
| Training seeds | `41201`-`41205` |
| Reserved formal seeds | `42001`-`42008`, untouched |
| Training budget | 1,000,000 nominal timesteps per policy |
| Checkpoints | 250k, 500k, 750k and 1M |
| Evaluation seeds | `51201`-`51220`, paired across policies and checkpoints |
| Evaluation horizon | At most 1,000 steps per episode |
| Action selection | Deterministic during evaluation |
| Device | CPU only |
| Parallel workers | At most four; one independently trained policy per task |
| Normalisation | Disabled |
| Shaping | All shaping weights exactly zero |

PPO remains unchanged: `MlpPolicy`, two 64-unit `Tanh` hidden layers for the
actor and critic, `n_steps = 2048`, `batch_size = 64`, `n_epochs = 10`,
learning rate `3e-4`, `gamma = 0.99`, `gae_lambda = 0.95`, clip range `0.2`,
entropy coefficient `0`, value coefficient `0.5`, maximum gradient norm `0.5`,
advantage normalisation enabled and state-dependent exploration disabled.

## 4. Replication hierarchy

The independently trained policy, indexed by training seed, is the replication
unit. The 20 evaluation episodes are repeated observations nested within one
fixed policy. They improve estimation of that policy's termination rate and
mean velocity but do not turn five policies into 100 independent training
replications. The four checkpoints are repeated measurements of the same
evolving policy and are not independent replications.

The five development seeds are new and disjoint from the earlier `41101` and
`41102` development seeds and the reserved `42001`-series formal seeds.

## 5. Frozen reference-competence gate

At the primary 1M endpoint, one policy passes only if both conditions hold:

\[
\widehat p_{\mathrm{unhealthy}} \leq 0.20,
\qquad
\overline v_x \geq 0.10\ \text{position units s}^{-1}.
\]

With 20 evaluation episodes, the empirical unhealthy-termination rate has a
resolution of `0.05`; therefore, at most four unhealthy terminations are
permitted for a passing policy. The health flag is limited to Ant-v5's finite
state and torso-height rule. It is not comprehensive robot health or physical
safety. Mean forward velocity is displacement divided by simulated episode
duration, not leg cadence.

The configuration-level decision is prospective and descriptive:

- **supported:** four or five of the five policies pass jointly;
- **inconclusive:** two or three policies pass jointly;
- **failed:** zero or one policy passes jointly.

This gate is an operational interpretability screen, not a literature-derived
success threshold, a confidence interval, or a null-hypothesis significance
test. The exact seed-level values and all failures will be shown.

## 6. Analysis and claim rules

The 1M checkpoint is the sole primary endpoint. Intermediate checkpoints show
learning trajectories but cannot replace a failed endpoint or be selected
post hoc. Episode results will be aggregated within policy before the five
policies are compared.

The primary output is the configuration-level competence classification. All
reward components, forward progress, episode length, termination categories,
action diagnostics, lateral drift and torso orientation remain in the raw CSV
for audit. No scalar `true_reward` or `true_performance` is introduced.

If the configuration is supported, the next action is to freeze a separate
held-out stage-one candidate-confirmation protocol. Candidate training does not
start automatically. If the result is inconclusive or failed, the next action
is a separately frozen reference-configuration pilot; architecture,
normalisation and optimisation settings must not be changed after inspecting
V6 without recording a new design decision.

V6 cannot by itself establish reward misspecification, reward hacking, a
critical coefficient, an optimal coefficient, or real-robot safety.

## 7. Stopping, failure and exclusion rules

The planned scientific stop occurs after all five policies reach and are
evaluated at the nominal 1M checkpoint. The run may stop early only for an
explicit user interruption, insufficient storage, an unrecoverable I/O error,
or a non-finite training failure. Windows sleep prevention is requested during
parallel execution.

Every failed, interrupted or non-finite policy is retained. A failed seed is
not replaced. An incomplete policy directory is not silently resumed because
that would make its training history differ from the declared uninterrupted
design. Any rerun requires a recorded adjudication and a versioned output path.

## 8. Verification

Before the long run, a separate 4,096-step smoke test must verify model saving,
checkpoint evaluation, reward decomposition, zero shaping and CSV schemas. The
complete automated test suite must pass.

After training, primary analysis must verify the expected 20 models, 20 runtime
rows and 400 evaluation rows, unique policy/checkpoint/evaluation keys, complete
paired evaluation seeds and finite gate metrics. A separate verifier must
recompute the five policy gates directly from the raw CSV without importing the
primary analysis module, reload every model, compare timestep metadata and
produce SHA-256 hashes.

## 9. Provenance

The machine-readable source of this protocol is
`configs/stage1_reference_fresh_1m_v6_20260814.json`. Its parent decision is
`configs/stage1_post_extension_gate_v5_20260814.json`. The historical V4/V5
evidence remains unchanged and is not merged with V6 outcomes.
