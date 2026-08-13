# Experiment Status

## Completed evidence

### Formal v1

- Status: completed retrospective exploratory study.
- Environment and algorithm: Ant-v5 with PPO.
- Coefficient sweep: `0.5`, `0.25`, `0.125` and `0.0625`.
- Core comparison: three training seeds for the reference, reduced-cost and
  historical shaped conditions.
- Checkpoints: 50k, 100k, 150k, 200k, 250k and 300k.
- Evaluation: ten paired deterministic episodes per checkpoint.

Formal v1 supports a descriptive multi-objective trade-off. It does not prove
uniformly worse locomotion or formal reward hacking.

## Historical shaping

The old shaped condition duplicated the forward-reward component at
`ctrl_cost_weight=0.0625`. It is retained to preserve provenance, but it is not
the intended bounded, behaviour-targeted mitigation experiment.

## Not completed in this repository

- A prospectively frozen diagnosis using fresh held-out training seeds.
- A bounded mitigation intervention applied at a reproducibly divergent weight.
- Physical-robot, ROS, vision or real-energy evaluation.

## Interpretation warning

The private team repository contains the pre-revision implementation. New
method-development work is deliberately isolated and is not silently merged
with formal-v1 evidence.
