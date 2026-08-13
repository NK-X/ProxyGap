# Reproducibility

## 1. Create the environment

```powershell
conda env create -f environment.yml
conda activate proxygap-ant
python -m pip install -e .
```

## 2. Run automated checks

```powershell
python -m pytest tests
python scripts/inspect_ant_reference.py
```

## 3. Run a smoke test

```powershell
$env:MUJOCO_GL = "disable"
python scripts/smoke_train_benchmark.py
```

Do not interpret this run scientifically. Its purpose is to confirm that the
environment, PPO and filesystem output pipeline operate locally.

## 4. Validate configurations

```powershell
python scripts/validate_prospective_protocol.py `
  --config configs/prospective_v2_revision_gate_20260810.json
```

A blocked result is expected for the included prospective draft. Do not bypass
the gate by editing only a status string.

## 5. Generated outputs

Models, CSV logs, figures and videos are written beneath `artifacts/`, which is
excluded from Git. When sharing a result package, include its configuration,
package versions, seeds, source commit SHA and a SHA-256 manifest.

## 6. Rendering

Headless training uses `MUJOCO_GL=disable`. Video rendering requires a supported
rendering backend such as `glfw` on the host system.
