# V2 Research Direction: Intended Gait, Misspecification and Mitigation

## Scope

V2 remains a reward-misspecification and reward-shaping study in flat-ground
Gymnasium Ant-v5 with PPO. It does not target a real robot or claim biological
fidelity.

## Intended behaviour hierarchy

The intended behaviour will be frozen in three layers rather than collapsed
immediately into one invented true-reward scalar.

### 1. Core task

- track a declared positive forward-velocity command;
- avoid material lateral motion and yaw when the command is straight;
- complete the fixed evaluation horizon unless a safety condition is violated.

### 2. Safety constraints

- prevent sustained inversion or clearly unhealthy torso states;
- retain the declared action range and simulator joint limits;
- distinguish an external intervention from behaviour learned by the policy.

### 3. Gait-quality preferences

- coordinated four-leg contact timing;
- bounded vertical body motion and roll/pitch rotation;
- limited unintended flight or repeated take-off;
- bounded action changes and control effort;
- stable path and posture across the evaluation horizon.

## Operationalisation gate

Before another mitigation search, the project must:

1. map the four Ant-v5 legs and contact bodies to stable semantic labels;
2. extract per-step contact indicators or forces with tested conventions;
3. predeclare whether a free coordinated gait or a specific trot-like template
   is being evaluated;
4. define cadence, duty factor, diagonal or adjacent-leg phase relations,
   contact regularity and flight fraction;
5. define metric direction, units, episode aggregation and practical thresholds;
6. test sensitivity to reasonable thresholds without selecting them from held-out
   outcomes;
7. preserve task, safety and gait diagnostics separately in the main analysis.

Reducing vertical velocity or angular velocity alone does not establish a
coordinated gait. Conversely, a reference gait should not be imposed merely
because it is visually familiar; its task relevance and measurability must be
stated.

## Reward, constraints and evaluation

The same intended behaviour may be represented differently by function:

| Function | Appropriate role |
|---|---|
| Forward command tracking | Core task reward |
| Sustained inversion or hard action limits | External constraint or termination rule |
| Contact coordination, posture and smoothness | Bounded shaping candidates |
| All task, safety and gait measures | Independent evaluation diagnostics |

Not every diagnostic should be inserted into the optimised reward. Independent
diagnostics are required to detect whether an intervention improves its visible
objective while shifting failure to an omitted behaviour.

## Experimental sequence

```text
freeze intended gait
        -> audit default reward and existing policies
        -> attribute the largest reproducible gap
        -> compare a bounded intervention set
        -> reject candidates that transfer failure across domains
        -> freeze one candidate
        -> held-out paired training-seed confirmation
```

Development may use only a limited, versioned number of iterations. Raw
development outputs remain separate from held-out evidence, and negative
results remain part of the audit trail.

## Authorised claim boundary

Until the operationalisation gate is passed, the project may describe measured
posture, direction, contact and smoothness behaviour. It may not claim natural
gait, biological realism, physical safety or successful complete mitigation.
