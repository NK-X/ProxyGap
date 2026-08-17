# ProxyGap: Reward Misspecification and Reward Shaping in Ant-v5

ProxyGap is a student research project using Gymnasium `Ant-v5` and
Stable-Baselines3 PPO. It studies whether the reward optimised by an agent
reliably represents a predeclared, task-appropriate quadruped locomotion
intention, and whether bounded reward shaping or external constraints can
reduce an observed proxy-behaviour gap.

## Start here

This repository contains two clearly separated research versions. They must
not be analysed as one frozen experiment.

| Version | Role | Entry point |
|---|---|---|
| **V2 current** | Intended-gait specification, default-reward audit and bounded mitigation development | [`current/README.md`](current/README.md) |
| **V1 legacy** | Historical `ctrl_cost_weight` sweep and retrospective exploratory evidence | [`legacy/weight_sweep_v1/README.md`](legacy/weight_sweep_v1/README.md) |

For a project handover, begin with [`handoff/START_HERE.md`](handoff/START_HERE.md)
and [`STATUS.md`](STATUS.md). The change from V1 to V2 is documented in
[`CHANGELOG.md`](CHANGELOG.md).

The two user-directed reward iterations trained on 17 August are recorded in
[`docs/REWARD_ITERATION_HISTORY_20260817.md`](docs/REWARD_ITERATION_HISTORY_20260817.md),
with a machine-readable, path-sanitised manifest in
[`results/development_20260817/reward_iterations/version_manifest.json`](results/development_20260817/reward_iterations/version_manifest.json).

## Current research direction

V2 no longer treats a one-directional `ctrl_cost_weight` sweep as the main
experiment. Its intended sequence is:

1. **Specify intended behaviour.** Convert the task, safety and gait-quality
   requirements into measurable quantities without claiming that Ant-v5 is a
   biologically faithful animal model.
2. **Audit the default reward.** Test whether independently trained policies
   ranked by the documented Ant-v5 proxy are also acceptable under the frozen
   behavioural specification.
3. **Develop bounded mitigation.** Compare a small number of predeclared reward
   shaping and external-constraint mechanisms.
4. **Freeze and confirm.** Use untouched training seeds only after the intended
   behaviour, metrics, reward, constraints and exclusion rules are frozen.

The phrase **natural gait** is not currently an authorised result claim. The
next design gate is to operationalise a stable, coordinated, task-appropriate
quadrupedal gait using contact sequence, posture, direction, smoothness and
task diagnostics. See
[`current/RESEARCH_DIRECTION_V2.md`](current/RESEARCH_DIRECTION_V2.md).

## Scientific status

- V1 is retained as retrospective exploratory evidence, not confirmatory
  proof of universal reward hacking.
- Result summaries dated 16 August 2026 are development evidence.
- Future-test protocols and code may be public before execution, but new
  models, logs, videos and unreviewed result tables remain local.
- No held-out V2 formal comparison has been authorised.
- No real-robot, terrain, disturbance or biological-gait claim is in scope.

## Repository layout

| Path | Purpose |
|---|---|
| `current/` | Canonical V2 direction and decision gates |
| `legacy/` | V1 status and provenance map; no raw evidence is rewritten |
| `handoff/` | Transfer guide, data dictionary, run registry and file manifest |
| `src/proxygap/` | Ant-v5 wrappers, reward decomposition, metrics and experiment logic |
| `scripts/` | Training, evaluation, rendering, analysis and QA entry points |
| `configs/` | Immutable historical records and versioned development configurations |
| `protocols/` | Predeclared protocols, adjudications and deviation records |
| `tests/` | Engineering, metric and schema regression tests |
| `docs/` | Detailed research notes and reproducibility guidance |
| `reports/` | Reviewed development reports with explicit claim boundaries |
| `results/` | Lightweight public summaries and sanitised video indexes only |
| `presentations/` | Editable English and Chinese team updates |

Existing executable paths remain in place so that historical commands and
hashes continue to work. The `current/` and `legacy/` directories are
navigation and governance layers, not duplicated source trees.

## Public-data boundary

Git intentionally excludes trained model archives, compressed step logs,
complete MP4 panels, recovery folders, environments and machine-specific
paths. These materials belong in a separately verified handover bundle. The
repository records their schemas, identifiers and provenance without
presenting evaluation episodes or videos as independent replications.

## Installation on Windows

```powershell
conda env create -f environment.yml
conda activate proxygap-ant
python -m pip install -e .
python -m pytest tests
```

CUDA is not required. Passing tests establishes implementation consistency; it
does not establish construct validity or a scientific finding.

## Research-integrity boundaries

- Training seeds create independently trained policies and are the replication
  units; evaluation episodes are nested repeated measurements.
- Raw returns from different reward formulae are not automatically comparable.
- Videos are prespecified qualitative audit evidence, not additional samples.
- `common_rescored_return` is a comparator, not true human performance.
- Raw and generated data are immutable; corrections create a new version and a
  deviation record.
- Conclusions are limited to the tested flat-ground Ant-v5, PPO configuration
  and training budget.
