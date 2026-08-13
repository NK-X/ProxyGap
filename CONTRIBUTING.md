# Contributing

## Before changing code

1. Read `docs/PROJECT_OVERVIEW.md` and `docs/EXPERIMENT_STATUS.md`.
2. Create a branch named `name/short-purpose`.
3. State whether the change affects engineering, measurement or research design.

## Required checks

```powershell
conda activate proxygap-ant
python -m pytest tests
```

Training-code changes also require a small smoke test. Smoke outputs must be
stored outside Git and labelled as engineering evidence, not experimental
results.

## Research rules

- Do not alter raw logs or generated data manually.
- Do not compare raw returns from different reward formulas as a common scale.
- Do not treat evaluation episodes or checkpoints as independent replications.
- Do not call `common_rescored_return` true performance.
- Do not promote a draft or pilot configuration to formal status by changing
  only its status label.
- Record changes to seeds, weights, metrics, exclusions or stopping rules before
  interpreting the affected results.

## Pull requests

Describe the purpose, files changed, tests run, expected scientific effect and
remaining limitations. Keep code and research-design changes distinguishable.
