# ProxyGap Current Development Synthesis

**Date:** 16 August 2026<br>
**Status:** development evidence; not a held-out formal result

## Updated direction

The project now studies whether the default Ant-v5 scalar reward can select a
PPO policy that scores well while failing a predeclared, project-specific
flat-ground locomotion intention. If a reproducible proxy-behaviour gap is
identified, a bounded development stage tests whether reward shaping, external
constraints or a training mechanism reduces the gap without sacrificing the
forward task.

The historical control-cost coefficient sweep remains exploratory evidence. It
is no longer the main experiment because comparing several incomplete reward
variants does not first establish whether the default proxy represents the
intended behaviour.

## What is measured

The platform task, optimiser-facing reward and researcher-facing intention are
kept separate. The intention is a vector rather than an invented scalar true
reward: target-speed tracking, upright posture, no unhealthy termination or
sustained inversion, limited direction error, direct paths, smooth actions and
limited action saturation over a 1,000-step episode.

## Development findings

1. A deterministic replay of one guardrail policy contained 15 prominent
   take-offs, 52.9% steps without floor contact and a maximum vertical root
   velocity of 2.540 m/s. This is mechanism evidence, not a population estimate.
2. Replacing unbounded forward velocity with 1 m/s target tracking improved
   survival and reduced action roughness at 300k steps. Extending the two target
   conditions to 1M steps improved forward speed, but repeated take-off and
   flight remained.
3. Under ordinary PPO exploration, bounded body-dynamics shaping reduced mean
   take-offs from 21.1 to 3.77 and reduced the no-floor-contact fraction from
   0.526 to 0.465 across the development policies, with a modest reduction in
   mean forward speed from 0.961 to 0.931 m/s.
4. The exact tested gSDE setting failed: both gSDE cells showed near-zero
   forward speed and unhealthy termination in all evaluation episodes. This
   rejects that configuration, not gSDE in general.
5. The external action-slew projection made applied actions smoother, but it
   intervened on approximately 98.4-98.7% of steps while the proposed policy
   actions remained rough. It is therefore a strong guardrail, not evidence
   that the policy itself learned smooth control.

## Open problems

1. **Natural gait:** current diagnostics capture stability, direction, action
   quality and contact behaviour, but no gait family or footfall phase target
   has been predeclared. Biological or natural-gait claims are therefore out of
   scope unless a separate contact-phase protocol is frozen.
2. **Policy versus guardrail:** a controller can appear smooth because an
   external projection edits almost every proposed action. Formal mitigation
   must distinguish learned policy quality from constraint-layer performance.

## Next gate

Before formal training, the team should decide whether the project claims only
general stable locomotion or a specific gait family. It should then select at
most one coherent mitigation candidate, freeze the reward, constraints, PPO
configuration, metrics, exclusion rules and video-selection rule, and evaluate
the frozen design with training seeds not used during development.

All claims remain limited to default flat-ground Ant-v5, PPO and CPU-based
simulation. No claim is made about physical robot safety, terrain robustness or
biological locomotion.
