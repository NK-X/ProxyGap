# ProxyGap Status

Last updated: 17 August 2026

## Canonical version

**V2 is the current research direction.** Its immediate gate is to define a
measurable, task-appropriate quadrupedal gait before selecting another shaping
or constraint candidate.

| Workstream | Public status | Next gate |
|---|---|---|
| Intended task and safety requirements | Conceptually defined | Freeze measurable thresholds and aggregation |
| Specified gait/contact pattern | Scientifically unresolved | Map Ant-v5 legs and contacts; define coordination metrics |
| Default-reward construct audit | Development evidence exists | Re-evaluate under the frozen V2 behaviour contract |
| Reward-shaping mechanisms | Development only | Permit only bounded candidates derived from diagnosed gaps |
| External constraints | Development only | Separate safety enforcement from soft preferences |
| Body-dynamics replication protocol | Public protocol and code available; local outputs are outside this release | Do not infer a public result from the protocol alone |
| Final mitigation candidate | Not frozen | Pass a predeclared development gate |
| Held-out formal comparison | Not authorised | Preserve reserved training seeds |
| Real hardware, terrain and disturbances | Out of scope | None for this project |

## Latest reward iterations

Two user-directed development versions are now recorded:

- `reward-v1-foot-landing`: four-foot grounded Vy/Vz shaping;
- `reward-v2-pitch-balance`: the same package plus event-level signed-pitch
  time balance.

The pitch version improved its declared balance score but increased landing
frequency and action roughness, so it remains development evidence rather than
a frozen final mitigation candidate. See
[`docs/REWARD_ITERATION_HISTORY_20260817.md`](docs/REWARD_ITERATION_HISTORY_20260817.md).

## V1 status

The earlier control-cost coefficient sweep is preserved as **legacy
retrospective exploratory evidence**. It may motivate V2 but cannot be merged
with V2 development runs as if both came from one preregistered design.

## Public-release boundary

The public repository contains code, protocols, configurations, tests,
lightweight reviewed summaries and sanitised indexes. Newly generated models,
logs, full videos, recovery records and unreviewed result tables are not part
of this release.
