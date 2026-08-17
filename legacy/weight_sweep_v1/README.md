# V1 Legacy: Control-Cost Weight Sweep

## Status

V1 is preserved as retrospective exploratory evidence. It is not the current
formal design and must not be pooled with V2 development or future held-out
runs.

## Original question

V1 varied `ctrl_cost_weight` in Ant-v5 PPO and compared condition-specific
reward, a fixed 0.5 common rescore and separate behavioural diagnostics. It
examined whether reducing a reward component could produce proxy-behaviour
divergence.

## What it can support

- descriptive evidence that reward weights change learned behaviour;
- multi-objective trade-off and seed-sensitivity discussion;
- motivation for a stronger intended-behaviour audit;
- testing and provenance of reward decomposition and metric logging.

## What it cannot support

- a globally optimal reward weight;
- a universal reward-hacking or reward-misspecification theorem;
- a biological or natural-gait claim;
- a held-out V2 mitigation result;
- pooling with later runs as one preregistered experiment.

## Preserved V1 files

- `configs/formal_v1_coefficients_20260808.json`
- `configs/formal_v1_core_replication_20260808.json`
- `configs/formal_v1_shaped_20260808.json`
- `protocols/formal_v1_retrospective_analysis_20260810.md`
- `scripts/run_formal.py`
- `scripts/rescore_formal_v1.py`
- `scripts/analyse_formal_results.py`
- `scripts/render_formal_videos.py`
- `scripts/validate_formal_outputs.py`

The files remain at their original paths so hashes, imports and historical
commands are not broken. Git history and the version tag preserve the exact
pre-V2 public snapshot.
