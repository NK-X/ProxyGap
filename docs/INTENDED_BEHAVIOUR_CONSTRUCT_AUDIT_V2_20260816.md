# Intended Behaviour Construct Audit of V2

## Status

**Status:** development construct audit; V2 remains immutable. This document
does not retrospectively change any V2 compliance label. It records a defect to
resolve before a held-out protocol is frozen.

## Finding

The V2 label `action smoothness` is narrower than the intended phrase
`smooth locomotion`. Its metric,

\[
Q_a=\frac{1}{32(T-1)}\sum_{t=2}^{T}\lVert a_t-a_{t-1}\rVert_2^2,
\]

measures adjacent policy-output changes. It does not measure vertical body
motion, angular oscillation, aerial phases, foot slip, impact or gait phase.

This distinction is empirically material. Under the nominal V2 thresholds, all
six hybrid-guardrail conditions passed the action-output domain in every
evaluation episode, yet none passed path directness and the fixed replay of
`Rt0p1_Rvy0__K1p1`, training seed 41301, contained repeated take-offs, 52.9%
no-floor-contact steps and large raw contact-force peaks. Therefore, V2 cannot
use `action smoothness` as a synonym for body-level locomotion smoothness.

Evidence:

- `artifacts/dev/hg_r3_obsfix_v1/analysis/intent_sensitivity/condition_domain_compliance.csv`;
- `artifacts/dev/hg_r3_obsfix_v1/analysis/jump_contact_gait/jump_contact_gait_summary.json`;
- `artifacts/dev/hg_r3_obsfix_v1/analysis/contact_gait_matrix/endpoint_contact_gait_matrix.csv`.

## Required terminology correction for subsequent work

Use the following three constructs separately:

1. **Policy-output rate:** adjacent proposed or applied action changes, measured
   by \(Q_a\).
2. **Body-level locomotion smoothness:** vertical and angular body motion,
   flight time, take-off events, foot slip and impact-related diagnostics while
   retaining command tracking.
3. **Specified gait:** an explicitly declared foot-contact phase structure such
   as crawl, trot, pace or bound.

Passing one construct does not imply passing either of the others.

## Candidate V3 measurements

The next development protocol should report, without yet combining them into a
single score:

- RMS and maximum root vertical velocity;
- RMS roll/pitch angular velocity;
- no-floor-contact step fraction;
- prominent take-off count under a predeclared event definition;
- raw floor-force peak and upper quantile, labelled as MuJoCo diagnostics rather
  than hardware-calibrated force limits;
- foot-contact duty factors and contact-pattern occupancy;
- policy-output first difference and, if implemented, second difference;
- command velocity, path efficiency, direction error and termination alongside
  every smoothness diagnostic.

Numerical V3 acceptance thresholds are **not frozen by this audit**. They must
be justified before held-out evaluation using task intent, primary literature,
development distributions and sensitivity analysis. The existing exploratory
`hopping-dominant` flag is not a validated gait classifier.

## Gait decision boundary

The current project intent requires stable command-following locomotion but does
not name a biological gait. Simultaneous front-leg action is therefore not, by
itself, a protocol violation. If the project requires a crawl, trot or another
contact schedule, that gait must be declared before training and encoded through
phase/contact objectives, a reference controller or imitation data. It must not
be inferred retrospectively from whichever learned policy looks most natural.

## Literature basis and transfer limit

Gymnasium Ant-v5 defines torque actions and a reward containing healthy,
forward-velocity, control and clipped contact terms; it does not define an
action-rate or gait-phase term (Farama Foundation, 2026). Raffin, Kober and
Stulp (2022) distinguish jerky step-wise exploration from smooth robotic
exploration. Aractingi et al. (2023) separately use command-velocity tracking,
first- and second-order action-difference penalties, foot-slip terms and a
position/PD control interface for Solo12. These sources justify separating the
constructs; they do not validate the present Ant thresholds or prove that one
intervention will solve every failure mode.

## References

Aractingi, M., Desbiez, A., Ferrari, R., Le Moal, C., Ivaldi, S. and Mouret,
J.-B. (2023) 'Controlling the Solo12 quadruped robot with deep reinforcement
learning', *Scientific Reports*, 13, 11945. doi:
10.1038/s41598-023-38259-7.

Farama Foundation (2026) 'Ant - Gymnasium documentation'. Available at:
https://gymnasium.farama.org/environments/mujoco/ant/ (Accessed: 16 August
2026).

Raffin, A., Kober, J. and Stulp, F. (2022) 'Smooth exploration for robotic
reinforcement learning', *Proceedings of Machine Learning Research*, 164,
pp. 1634-1644.
