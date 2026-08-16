# Literature Evidence Register V1

## Purpose

This register states exactly what each external source can and cannot support
in the ProxyGap study. It prevents citations from being used as substitutes
for project-specific empirical validation.

## Primary and authoritative sources

| Source | Evidence used in this project | Does not establish |
|---|---|---|
| Gymnasium, *Ant-v5 documentation* | The official task is to coordinate four legs to move in the positive horizontal direction; the default action has eight torque controls; the default observation has 105 values; the reward combines healthy and forward rewards minus control and contact costs. | That a trained PPO policy will be upright, smooth, direct or suitable for hardware; that the project's intended-behaviour thresholds are official Ant requirements. |
| Schulman et al. (2017), *Proximal Policy Optimization Algorithms* | PPO is a policy-gradient method that alternates sampled interaction and multiple optimisation epochs using a clipped surrogate objective. | That the current Stable-Baselines3 hyperparameters or 2-by-64 network are optimal for Ant-v5. |
| Pan, Bhatia and Steinhardt (2022), *The Effects of Reward Misspecification: Mapping and Mitigating Misaligned Models* | Reward misspecification can be studied by separating the reward used for optimisation from an independently defined evaluation objective. | That the present Ant-v5 observations already prove reward hacking or identify a scalar true reward. |
| Skalse et al. (2022), *Defining and Characterizing Reward Hacking* | Formal reward-hacking claims require a relationship between reward functions or policy orderings, not merely an unusual-looking video. | That domain-level proxy-diagnostic inversions in this development sample satisfy every formal definition of hackability. |
| Achiam et al. (2017), *Constrained Policy Optimization* | Reward maximisation and expected constraint satisfaction can be formulated separately in a constrained Markov decision process. | That the deterministic action projection implemented here is CPO, or that it inherits CPO's guarantees. |
| Raffin, Kober and Stulp (2022), *Smooth Exploration for Robotic Reinforcement Learning* | Step-wise exploration can create jerky robot actions; exploration design is a plausible alternative mechanism for action roughness. | That gSDE is required in this project or that reward misspecification is the only possible cause of rough motion. |
| Tan et al. (2018), *Sim-to-Real: Learning Agile Locomotion for Quadruped Robots* | Practical quadruped learning may require actuator modelling, latency treatment, perturbations and careful observation design when real transfer is intended. | Any need to add those elements to the present flat-ground simulation-only study, or any real-robot generalisation. |
| Aractingi et al. (2023), *Controlling the Solo12 quadruped robot with deep reinforcement learning* | Applied quadruped control evaluates and/or rewards command tracking, body orientation, joint effort and action smoothness rather than relying on forward motion alone. | Universal numerical weights or safety limits for Gymnasium Ant-v5. |

## Project-specific decisions requiring local evidence

The following values are operational study choices and must be justified by
local feasibility, sensitivity or development evidence rather than by the
citations above:

- the 1.0 m/s target and its 0.8-1.2 m/s tolerance;
- 15 degrees torso-tilt RMS;
- 5 degrees net-displacement direction error;
- 0.90 forward path efficiency;
- 0.04 normalised action roughness;
- 1% action-component saturation;
- cosine posture shaping with weight 0.1;
- lateral-offset shaping weights 0.05 and 0.1; and
- the normalised action-slew bound 1.1.

The bounded development matrices may reject or nominate these candidates.
They cannot turn them into universal robot-control standards.

## References and stable links

- Achiam, J., Held, D., Tamar, A. and Abbeel, P. (2017) 'Constrained policy
  optimization', *Proceedings of Machine Learning Research*, 70, pp. 22-31.
  https://proceedings.mlr.press/v70/achiam17a.html
- Aractingi, M. et al. (2023) 'Controlling the Solo12 quadruped robot with deep
  reinforcement learning', *Scientific Reports*, 13, 11945.
  https://doi.org/10.1038/s41598-023-38259-7
- Farama Foundation (2026) *Ant-v5*. Available at:
  https://gymnasium.farama.org/environments/mujoco/ant/
- Pan, A., Bhatia, K. and Steinhardt, J. (2022) *The effects of reward
  misspecification: mapping and mitigating misaligned models*. arXiv:2201.03544.
  https://arxiv.org/abs/2201.03544
- Raffin, A., Kober, J. and Stulp, F. (2022) 'Smooth exploration for robotic
  reinforcement learning', *Proceedings of Machine Learning Research*, 164,
  pp. 1634-1644. https://proceedings.mlr.press/v164/raffin22a.html
- Schulman, J. et al. (2017) *Proximal policy optimization algorithms*.
  arXiv:1707.06347. https://arxiv.org/abs/1707.06347
- Skalse, J. et al. (2022) *Defining and characterizing reward hacking*.
  arXiv:2209.13085. https://arxiv.org/abs/2209.13085
- Tan, J. et al. (2018) *Sim-to-real: learning agile locomotion for quadruped
  robots*. arXiv:1804.10332. https://arxiv.org/abs/1804.10332
