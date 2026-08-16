# Stage-One Pre-Formal Revision Gate V3

**Status:** scientifically unresolved; formal held-out training and reward
shaping remain blocked.

## 1. Evidence adjudication

The bidirectional development grid contained nine tested coefficients from
`0.125` to `0.75`, two development training seeds, six checkpoints and ten
paired evaluation episodes per checkpoint. The combined dataset contains 1,080
episode rows with no missing cells, duplicate episode keys, non-finite decision
metrics or reward-reconciliation failures above `1e-3`.

At 300k, `w=0.21875` passed the frozen development screen. Under matched
rescoring with `R_0.21875`, both trained policies had strictly higher proxy
return than their same-seed `w=0.5` references. Mean absolute lateral drift was
at least 0.5 position unit higher in both seeds. The result was present at all
three late checkpoints and was unchanged by proxy margins of 0%, 2.5% and 5%.
It passed half and nominal diagnostic margins, but not double margins.

Neither `w=0.625` nor `w=0.75` qualified. This is a bounded negative result for
the tested upper values, not evidence that no larger coefficient can produce a
different failure mode.

## 2. Why formal confirmation cannot start yet

The development screen is an operational proxy-diagnostic divergence, not yet
an unambiguous high-reward/low-overall-performance result.

First, the `w=0.5` reference was majority-unhealthy in both development
policies at 300k: unhealthy-termination rates were 0.9 and 0.7. Its mean episode
lengths were 331.4 and 485.9 steps. The local Gymnasium registration reports an
`Ant-v5` reward threshold of 6000, whereas the two reference-policy mean returns
were approximately 243 and 476. The registered threshold is contextual rather
than a frozen project success criterion, but the scale and termination evidence
show that 300k is not a demonstrated competent reference.

Second, the `0.21875` candidate ran for all 1,000 steps in both development
policies. Absolute lateral drift increased, but final lateral displacement per
unit absolute forward progress and lateral-path fraction did not both worsen
across seeds. Longer exposure and greater travel therefore remain plausible
alternative explanations unless the intended task explicitly requires staying
near the initial x-axis corridor.

Third, two development training seeds are candidate-selection evidence, not
held-out replication. Evaluation episodes are nested within each trained policy
and do not increase the independent sample size.

## 3. PPO implementation adjudication

The saved policy has 105 observation inputs, eight bounded action outputs and
22,481 trainable parameters. Stable-Baselines3 resolves the architecture to
separate two-layer policy and value networks with 64 units per layer and Tanh
activation. The optimiser is Adam with learning rate `3e-4`, betas `(0.9,
0.999)`, epsilon `1e-5` and zero weight decay. The run also uses `n_steps=2048`,
batch size 64, ten optimisation epochs, `gamma=0.99`, `gae_lambda=0.95`, clip
range 0.2, no entropy bonus and no state-dependent exploration.

This is defensible as a standard PPO architecture: Schulman et al. (2017) used
two 64-unit Tanh layers for their MuJoCo experiments. It is not established as
the unique or optimal Ant-v5 architecture. Their benchmark used one million
timesteps, and the current RL Baselines3 Zoo Ant/MuJoCo baseline also declares a
one-million-step scale with observation/reward normalisation. Consequently, the
current 300k unnormalised setting is suitable for development exploration but
is not yet a validated competent formal baseline.

## 4. Recommended next gate: budget extension only

Before any held-out run, extend the existing development policies for
`w=0.5`, `0.21875` and `0.125` from approximately 300k to one million target
timesteps. Keep the network, optimiser, reward, environment, seeds and lack of
normalisation unchanged. This isolates training budget from architecture and
pre-processing changes. Evaluate at 500k, 750k and 1M with the existing paired
evaluation seeds.

The extension remains development evidence. It has three purposes:

1. determine whether the weak reference is primarily under-trained;
2. determine whether the `0.21875` divergence survives greater optimisation;
3. retain `0.125` as a construct check because its development result involved
   path efficiency and command quality rather than absolute drift alone.

The recommended minimum reference-competence rule, requiring approval before
the extension, is that each reference training seed has an unhealthy-
termination rate no greater than 0.2 and mean forward velocity of at least
0.1 position unit per second at 1M. These are transparent simulation-level
operational thresholds, not physical safety limits or literature-derived Ant
success constants. The registered reward threshold of 6000 remains contextual
and is not substituted for the project intent.

If the reference fails this rule, formal training remains blocked. A separate
baseline-configuration pilot may then compare the current setup with a
normalised or otherwise documented PPO baseline; architecture or normalisation
must not be changed in the same run as the budget extension.

## 5. Provisional held-out design

Subject to the competence extension, the smallest defensible formal matrix is:

- `w=0.5`: reference comparator;
- `w=0.21875`: primary local development candidate, provided corridor adherence
  is accepted as part of the intended task;
- `w=0.125`: secondary mechanism/construct candidate.

Five new training seeds would support resource-limited descriptive replication;
eight would provide stronger directional evidence. Neither option converts the
seed-generating procedure into a universal population. The primary endpoint
would be 1M, with earlier checkpoints treated as dependent repeated measures.
Each candidate and reference would be rescored under the same candidate
`R_w`, and every seed-level effect would be reported.

## 6. Video rule

After numerical analysis, select the training seed and evaluation seed closest
to the median final-checkpoint net progress **within the reference condition**.
Use that same training seed and evaluation seed for every condition and
checkpoint. Exact and floating-point ties use the lower numeric seed. Render
one frame per simulator step at 20 fps (`dt=0.05 s`) until termination or the
1,000-step limit. Videos are qualitative audit evidence and cannot override the
numerical gate.

The development implementation has been verified with matched videos. The
reference trajectory contains 328 frames and lasts 16.4 s before unhealthy
termination; the `0.21875` trajectory contains 1,000 frames and lasts 50.0 s.

## 7. Decisions required for protocol freeze

1. Is the hidden intended task corridor-constrained forward locomotion, so that
   absolute y-axis deviation is independently undesirable?
2. Are the proposed reference-competence thresholds acceptable?
3. Should the one-million-step development extension be run before any formal
   seed is opened?
4. Should the formal matrix contain both `0.21875` and `0.125`, or only one?
5. Is the final replication plan five descriptive seeds or eight stronger
   directional seeds?

Until these decisions are frozen in a new executable configuration, the project
status is **scientifically unresolved**, not protocol-freeze-ready.

## References

Agarwal, R. et al. (2021) 'Deep reinforcement learning at the edge of the
statistical precipice', *Advances in Neural Information Processing Systems*, 34.

Farama Foundation (2026) *Ant - Gymnasium documentation*. Available at:
https://gymnasium.farama.org/environments/mujoco/ant/ (Accessed: 14 August
2026).

Henderson, P. et al. (2018) 'Deep reinforcement learning that matters',
*Proceedings of the AAAI Conference on Artificial Intelligence*, 32(1).

Pan, A., Bhatia, K. and Steinhardt, J. (2022) 'The effects of reward
misspecification: mapping and mitigating misaligned models', *International
Conference on Learning Representations*.

Raffin, A. (2026) *RL Baselines3 Zoo PPO hyperparameters*. Available at:
https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml
(Accessed: 14 August 2026).

Schulman, J. et al. (2017) 'Proximal policy optimization algorithms',
arXiv:1707.06347.

Skalse, J. et al. (2022) 'Defining and characterizing reward hacking',
*Advances in Neural Information Processing Systems*, 35.
