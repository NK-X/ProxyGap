# RQ1 Matched Default-Reward Baseline Protocol

**Status:** frozen, resource-limited formal descriptive comparison

**Config:** `configs/rq1_matched_baseline_v1_20260820.json`
**Primary unit:** independently trained policy (`n = 3` paired training seeds)

## Question and comparison

This experiment asks whether one prospectively frozen Stage-1 task-oriented
reward package improves selected flat-ground locomotion diagnostics relative to
the default Ant-v5 reward under a matched PPO implementation.

The comparison changes the optimiser-facing reward package only. Both
conditions use the same Ant-v5 simulator, diagnostic wrapper, 113-dimensional
observation, eight-dimensional action, PPO network, optimiser, one-million-step
budget, checkpoints, training-seed identifiers and paired evaluation seeds.
The previous applied action is appended to both observations because the shaped
condition contains an action-rate term. Consequently, the comparator uses the
default Ant-v5 reward exactly but is not the unmodified 105-dimensional
Gymnasium interface. This deviation is disclosed rather than hidden.

## Frozen shaped package

`S1_STAGE1_SHAPED` reproduces the complete `reward-v1-foot-landing` package
selected before the held-out seeds were run:

- bounded tracking of a `1.0 m/s` forward target;
- lateral-velocity penalty;
- cosine torso-orientation penalty;
- action-rate penalty;
- bounded root vertical-velocity and roll/pitch angular-speed penalties;
- grounded distal-foot lateral- and vertical-velocity penalties.

The later pitch-balance term is excluded because development evidence showed
that it improved its own balance score while increasing action roughness and
landing frequency. The experiment estimates the package effect; it cannot
attribute any result to an individual reward term.

## Replication and evaluation

- Training seeds: `62401`, `62402`, `62403`.
- Evaluation seeds: `72401` to `72410` for every trained policy and checkpoint.
- Endpoint: one million interactions per policy.
- Primary checkpoint: one million interactions.
- Intermediate checkpoints: descriptive learning dynamics only.
- Evaluation: deterministic policy, at most 1,000 steps per episode.

Evaluation episodes are nested measurements. They are averaged within policy
before conditions are compared. They do not increase the independent sample
size beyond three policies per condition.

## Outcomes and decision rule

Primary diagnostic domains are target-speed error, direction error, path
efficiency, action roughness and unhealthy-termination rate. Secondary
diagnostics include torso tilt, action saturation, airborne exposure, distal
support, duty fraction and a common default-reward rescore.

The resource-limited descriptive gate requires:

1. no higher unhealthy-termination rate for the shaped policy in at least two
   of three paired training seeds; and
2. improvement in at least three of four quality outcomes (target-speed error,
   direction error, path efficiency and action roughness), where each improved
   outcome must favour shaping in at least two of three paired seeds.

This gate is a project decision rule, not a significance test. With three
pairs, the minimum exact two-sided sign-test p-value is `0.25`, even if all
three effects point in one direction. Every policy-level value and paired
effect must therefore be shown.

## Validity boundaries

The experiment does not validate a biologically natural gait. It does not test
unseen terrain, new robot morphologies, physical hardware or different PPO
implementations. A failure of the shaped package is retained as a result; no
seed, coefficient or checkpoint may be replaced because an outcome is
inconvenient.

The exact source, configuration, runtime environment, checkpoints, raw episode
rows, policy-level summaries, paired effects, failures and SHA-256 manifest are
stored under the versioned output root. A smoke run validates engineering only
and cannot answer RQ1.
