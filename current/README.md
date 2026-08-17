# V2 Current Research Direction

This directory is the canonical starting point for new ProxyGap work. It is a
navigation layer: implementation remains in `src/`, `scripts/`, `configs/`,
`protocols/` and `tests/` so existing commands and provenance remain valid.

## Research focus

V2 asks whether the default Ant-v5 reward adequately represents a predeclared
flat-ground quadrupedal locomotion intention, and whether a bounded shaping or
constraint intervention can reduce any confirmed mismatch.

The study does not aim to prove that Ant-v5 reproduces a real animal. The
current operational target is a **stable, coordinated, task-appropriate
quadrupedal gait**. A trot-like contact template is a candidate, not yet a
frozen fact.

## Required order

1. Freeze intended task, safety and gait-quality constructs.
2. Map each construct to diagnostics independently of the reward.
3. Audit existing baseline policies against those diagnostics.
4. Attribute material gaps before choosing an intervention.
5. Run a bounded development comparison with fixed seeds and budgets.
6. Freeze one candidate and its evaluation rules.
7. Use untouched held-out training seeds for formal comparison.

## Current authoritative files

- `current/RESEARCH_DIRECTION_V2.md`: current scientific design.
- `STATUS.md`: project-wide status and next gate.
- `docs/INTENDED_BEHAVIOUR_CONTRACT_V2_20260816.md`: existing behaviour contract
  to be revised, not silently treated as final gait specification.
- `docs/EXPERIMENT_STATUS.md`: detailed evidence status.
- `handoff/START_HERE.md`: transfer and reproduction order.

## Not yet frozen

- leg-name and foot-contact mapping;
- gait family or contact-phase template;
- duty-factor, cadence and permitted flight-phase ranges;
- thresholds and aggregation for gait coordination;
- final shaping formula and coefficients;
- final external constraints;
- held-out formal protocol.

No new long training should be labelled formal until these items are resolved
and versioned.
