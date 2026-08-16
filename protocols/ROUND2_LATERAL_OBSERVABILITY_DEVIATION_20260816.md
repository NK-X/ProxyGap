# Round-two Lateral Observability Deviation

## Detection

Detected during a static implementation audit on 16 August 2026 while the
round-two development matrix was still running and before its outcome metrics
were inspected.

## Intended design

The approved straight-line tracking formulation was a lateral-velocity error
term with command \(v_y^\star=0\), for example

\[
r_{y,t}=-\lambda_y(v_{y,t}-v_y^\star)^2
\]

or a bounded transformation of the same observable error. The default Ant-v5
observation contains torso lateral velocity.

## Executed round-two implementation

The frozen round-two source uses

\[
r_{y,t}=-\lambda_y
\tanh\left(\frac{|y_t-y_0|}{s_y}\right).
\]

Default Ant-v5 excludes absolute torso \(x\) and \(y\) position from the
105-dimensional observation. The project appends only the preceding applied
eight-dimensional action, yielding 113 dimensions. Consequently, the policy
does not directly observe the cumulative offset used by this reward term.
It can respond to lateral velocity and other correlated state, but it cannot
directly condition its action on the signed accumulated displacement.

## Impact

- Engineering and data-quality evidence from the run remains valid.
- The action-slew comparison and posture-only comparator remain descriptive
  development evidence.
- Conditions with non-zero lateral-offset shaping cannot nominate the final
  reward package for the approved lateral-velocity design.
- Any weak result could reflect partial observability rather than an
  ineffective lateral-control objective.
- Any apparent improvement would still be evidence for the executed
  offset-based reward, not for the approved velocity-tracking formulation.

## Corrective rule

The active round-two run must not be altered or restarted. After complete QA,
one engineering correction may be versioned because the trigger is a
prospective implementation-conformance defect discovered before outcome
inspection, not performance-based reward tuning.

The correction must:

1. implement a bounded lateral-velocity penalty based on observed \(v_y\);
2. retain posture weight 0.1 and the same candidate weights 0, 0.05 and 0.1;
3. retain the 1.1 action-slew candidate, PPO settings, budgets and development
   seeds;
4. add reward-decomposition, sign, boundedness and observation-access tests;
5. preserve the complete round-two outputs as non-confirmatory evidence; and
6. leave held-out training seeds untouched.

This correction does not authorise a third outcome-driven tuning round.

## Scope

The issue and correction concern default flat-ground Ant-v5 simulation only.
They do not add terrain, friction, external-force or real-hardware claims.
