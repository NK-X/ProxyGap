# Project Overview

## Research question

Can a plausible reduction in the Ant-v5 control-cost penalty allow PPO to
obtain a favourable proxy reward while separately measured locomotion
properties deteriorate?

## System under study

- Environment: Gymnasium MuJoCo `Ant-v5`.
- Agent: Stable-Baselines3 PPO with an MLP actor-critic policy.
- Observation: 105 continuous state values.
- Action: eight bounded joint-control values.
- Compute: local Windows laptop, CPU-only.

## Experimental idea

The Ant reward combines forward movement, healthy survival, contact cost and a
penalty proportional to squared action magnitude. The exploratory study changes
only `ctrl_cost_weight` while keeping the remaining environment and PPO settings
fixed.

PPO receives the condition-specific scalar reward. Evaluation additionally
records forward progress, squared action, unhealthy termination, lateral drift,
torso orientation and episode length. These diagnostics remain disaggregated;
the project does not define a universal scalar `true_performance` score.

## Two intended experiments

1. **Diagnosis:** test whether a plausible reward-weight change produces a
   reproducible proxy-diagnostic divergence.
2. **Mitigation:** if a divergence is established, keep the detected
   `ctrl_cost_weight` fixed and add a different, bounded behavioural shaping
   signal targeting the observed failure.

The second experiment must not simply restore the original control-cost weight.
It is also stopped when the first experiment does not identify a reproducible
failure.

## Claim boundary

This project is a controlled simulation case study. It does not establish
performance on a physical robot, direct electrical energy consumption, a
globally optimal reward coefficient or a universal theory of reward hacking.
