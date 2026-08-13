# Repository Guide

This page annotates the files most team members will encounter.

## Core source

| File | What it is | Use it for |
|---|---|---|
| `src/proxygap/ant_wrapper.py` | Gymnasium wrapper around Ant-v5 | Reward decomposition, shaping components and per-step diagnostic logging |
| `src/proxygap/metrics.py` | Metric definitions and episode accumulator | Reconstructing reward and calculating locomotion diagnostics |
| `src/proxygap/experiment.py` | PPO training and deterministic evaluation helpers | Checkpointing, models and evaluation CSV generation |
| `src/proxygap/protocol.py` | Configuration gate | Preventing incomplete prospective configurations from being treated as frozen |
| `src/proxygap/__init__.py` | Public package interface | Importing the supported project functions |

## Runnable scripts

| File | What it does | Status |
|---|---|---|
| `inspect_ant_reference.py` | Prints Ant-v5 defaults and interface details | Safe inspection |
| `smoke_train_benchmark.py` | Runs a tiny PPO training-speed check | Engineering test only |
| `smoke_render_video.py` | Checks local rendering and video generation | Engineering test only |
| `run_coefficient_pilot.py` | Runs a short coefficient pilot | Parameter-development evidence only |
| `run_formal.py` | Trains versioned formal-v1 conditions | Historical formal-v1 runner |
| `run_shaping_pilot.py` | Tests historical shaping settings | Exploratory; not current mitigation design |
| `validate_formal_outputs.py` | Checks required outputs and schemas | Integrity check |
| `rescore_formal_v1.py` | Applies the fixed 0.5 common rescore | Retrospective derived analysis |
| `analyse_formal_results.py` | Produces combined tables and figures | Retrospective result generation |
| `render_formal_videos.py` | Renders prespecified representative trajectories | Qualitative audit support |
| `validate_prospective_protocol.py` | Reports unresolved protocol blockers | Must pass before prospective runs |

## Configurations

- `reference_ant_v5.json`: documented environment and PPO reference settings.
- `formal_v1_*.json`: immutable records of the historical runs.
- `pilot_*.json`: development decisions, not final-result configurations.
- `prospective_v2_revision_gate_20260810.json`: blocked draft; do not run as a
  formal experiment.

## Tests

`tests/test_ant_wrapper.py` checks reset/step behaviour, reward reconciliation,
metric calculations, termination categories, seeding, CSV fields and protocol
validation. Passing tests establish implementation consistency, not construct
validity or a scientific finding.
