# ProxyGap: Reward Misspecification in Quadruped Locomotion

ProxyGap is a student research project investigating whether a PPO-controlled
quadruped can obtain a higher numerical training reward while exhibiting an
undesirable locomotion trade-off. The simulation uses Gymnasium MuJoCo
`Ant-v5`; the manipulated reward parameter is `ctrl_cost_weight`.

## Important status

This repository is a **pre-revision source snapshot** shared for team review.
It contains the implementation used for the completed exploratory `formal-v1`
study. It is not the current development branch and must not be presented as a
finished proof of reward hacking.

- `formal-v1`: completed, retrospective and exploratory.
- Historical shaping: forward-reward reweighting, not the intended bounded
  mitigation intervention.
- Prospective v2: draft and blocked; do not run it as a formal experiment.
- Training seed is the independent replication unit. Checkpoints and evaluation
  episodes are repeated observations, not extra independent samples.

See [Project Overview](docs/PROJECT_OVERVIEW.md),
[Experiment Status](docs/EXPERIMENT_STATUS.md), and
[Repository Guide](docs/REPOSITORY_GUIDE.md) before interpreting the code.

## Project structure

| Path | Purpose |
|---|---|
| `src/proxygap/` | Environment wrapper, metrics, PPO training and protocol validation |
| `scripts/` | Runnable inspection, smoke-test, training, evaluation and analysis commands |
| `configs/` | Versioned environment and experiment settings |
| `tests/` | Automated engineering checks |
| `docs/` | Metric definitions and team-facing methodological notes |
| `protocols/` | Formal-v1 retrospective record and blocked prospective draft |
| `results/` | Explanation of which generated outputs may be shared separately |

## Installation on Windows

Install Miniforge or Miniconda, open PowerShell in this repository, then run:

```powershell
conda env create -f environment.yml
conda activate proxygap-ant
python -m pip install -e .
python -m pytest tests
```

CUDA is not required. The baseline is CPU-only.

## Minimum verification

```powershell
python scripts/inspect_ant_reference.py
python scripts/smoke_train_benchmark.py
```

These commands verify installation and engineering feasibility only. A passed
smoke test is not scientific evidence for reward misspecification.

## Reproducing experiments

Read [Reproducibility](docs/REPRODUCIBILITY.md) before running any training.
Generated models, raw logs, figures and videos are intentionally excluded from
Git because they are large or derived artifacts.

## Scientific interpretation

Different values of `ctrl_cost_weight` define different numerical reward
functions. Raw condition-specific returns must therefore not be ranked as if
they were measurements on one common scale. The repository also computes a
fixed-weight common rescore, but this is a benchmark comparator rather than a
ground-truth measure of locomotion quality.

The strongest defensible formal-v1 interpretation is a simulation-specific,
multi-objective trade-off across reward, forward progress, action magnitude,
termination, lateral drift and torso orientation.

## Team workflow

Create a branch for each change and open a pull request. Do not edit raw result
files or silently alter locked configuration files. See
[Contributing](CONTRIBUTING.md).
