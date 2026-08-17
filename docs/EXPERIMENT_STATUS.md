# Experiment Status

> Version notice: V2 is canonical. Use `current/README.md` for new work and
> `legacy/weight_sweep_v1/README.md` for the V1 coefficient study.

## Status map

| Workstream | Current status |
|---|---|
| Formal-v1 coefficient sweep | Completed retrospective exploratory evidence |
| Default-reward construct audit | Development evidence; held-out confirmation not run |
| Orientation/lateral/action-slew candidates | Development evidence; complete intent gate not passed |
| Target-speed and action-rate experiment | Development mechanism evidence; 1M extension complete |
| Body-dynamics and gSDE matrix | Development mechanism evidence; matrix complete |
| Body-dynamics ordinary-exploration replication | Public protocol/code only; local outputs outside this release |
| Natural or specified gait evaluation | Scientifically unresolved; next V2 gate |
| Final mitigation candidate | Not frozen |
| Held-out formal comparison | Not authorised |
| Real-robot deployment | Out of scope |

## Historical formal-v1

Formal-v1 used an exploratory control-cost coefficient sweep and remains
preserved for provenance. It supports a descriptive multi-objective trade-off,
not a universal or confirmatory reward-hacking claim. Its historical shaped
condition duplicated forward reward and is not the current mitigation design.

## Current direction

```text
default reward construct audit
        -> bounded mechanism development
        -> candidate freeze
        -> untouched held-out training seeds
        -> confirmatory comparison
```

The public repository contains design and development material for the first
two components. A passed engineering test, a local run or a visually improved
video cannot advance a candidate to formal status.

## Current evidence boundary

All 16 August result summaries were produced with development seeds that
informed the design. They may support mechanism selection, negative results and
methodological reflection. They must not be described as independent formal
confirmation.

See `current/RESEARCH_DIRECTION_V2.md` and
`docs/TEAM_PROGRESS_UPDATE_20260816.md`. The bounded body-replication protocol
and its release boundary are recorded in
`docs/FUTURE_TESTING_DIRECTION_20260817.md`; its presence does not publish or
imply a result.
