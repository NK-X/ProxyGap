# ProxyGap Team Progress Update (16 August 2026)

## What the project studies now

The project now studies whether the default Ant-v5 scalar reward is an adequate
proxy for a predeclared flat-ground locomotion intention, and whether bounded
reward shaping or an external control mechanism can reduce any observed gap.

The earlier `ctrl_cost_weight` sweep remains exploratory background rather than
the central formal design.

## Human intention used for development

The current operational contract evaluates a full 1,000-step episode and asks
the Ant to track 1 m/s forward motion, remain broadly upright, avoid unhealthy
termination and sustained inversion, limit direction and path error, and avoid
rough or saturated actions. These thresholds are project-defined development
criteria, not universal robot-safety limits.

"Natural gait" is not yet a valid outcome because no crawl, trot, pace or bound
contact pattern has been frozen.

## Main development findings

### Posture and external action constraint

Cosine orientation shaping reduced some inversion failures. The external
action-slew projection guaranteed smoother applied actions, but approximately
98% of evaluation steps required intervention in the tested hybrid candidates.
The policy itself therefore remained rough.

### Residual hopping mechanism

One exact replay contained 15 prominent take-offs in 50 seconds, 52.9% of steps
without floor contact and a maximum vertical torso velocity of 2.540 m/s. The
replay matched the logged trajectory to floating-point precision. This is a
mechanism diagnostic, not a formal prevalence estimate.

### Target tracking and learned action smoothness

Replacing linear forward reward with bounded 1 m/s target tracking and adding a
normalised action-rate penalty were tested without an external limiter. At the
1M-step development endpoint:

| Condition | Mean speed | Path efficiency | Action roughness |
|---|---:|---:|---:|
| Target tracking, no action-rate penalty | 0.844 m/s | 0.809 | 0.0139 |
| Target tracking, action-rate weight 0.2 | 0.918 m/s | 0.857 | 0.00985 |

The policy learned smoother commands, but both conditions retained roughly 52%
no-floor-contact time and remained hopping-dominant under the exploratory
contact diagnostic.

### Body dynamics and exploration

A frozen 2x2 matrix tested a bounded vertical/angular body penalty and gSDE
exploration. Under ordinary Gaussian exploration, body shaping reduced mean
take-offs from 21.1 to 3.77 and reduced mean no-floor-contact fraction from
0.526 to 0.465, while mean forward velocity changed from 0.961 to 0.931 m/s.

The tested gSDE setting failed in all three development training seeds: policies
terminated early and did not track forward motion. This rejects that exact
configuration; it does not show that gSDE is generally ineffective.

## Two unresolved problems

1. **Natural gait:** current metrics can assess direction, posture, action
   smoothness and body dynamics, but not whether the four feet follow a declared
   biological gait.
2. **Policy versus controller:** an external guardrail may improve executed
   motion without teaching PPO to generate good actions. Proposed and applied
   actions must remain separately reported.

## Next gate

No formal held-out claim is authorised yet. The next scientific decision is
whether to nominate the ordinary-exploration body-dynamics candidate for one
final bounded development check, or to report it as partial mitigation evidence
and retain the remaining gait limitation. Only after this decision are the
reward, constraints, PPO settings, metrics and exclusion rules frozen for
untouched training seeds.

## Evidence locations

- Protocols: `protocols/`
- Executable configurations: `configs/`
- Lightweight analysis outputs: `results/development_20260816/`
- Development audit reports: `reports/`
- Editable group presentation: `presentations/`
