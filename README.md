# ProxyGap Ant-v5/PPO Canonical Implementation

## Revision gate (10 August 2026)

Formal v1 is a completed retrospective exploratory study. Prospective v2 is **not frozen and must not be run yet**. The controlling records are:

- `protocols/formal_v1_retrospective_analysis_20260810.md`;
- `protocols/prospective_v2_protocol_draft_20260810.md`;
- `docs/METRIC_DEFINITIONS_V2_20260810.md`;
- `configs/prospective_v2_revision_gate_20260810.json`.

Different `ctrl_cost_weight` conditions optimise different numerical reward definitions. Reports must distinguish `condition_objective_return` from the fixed-weight `common_rescored_return`; neither is a scalar true-performance measure. The historical `fall` field is a legacy alias. Prospective reporting uses low-z collapse, high-z excursion, non-finite termination and TimeLimit truncation.

Run the revision-gate validator before any timed pilot. A blocked result is currently expected because scientific parameters still require approval:

```powershell
python scripts/validate_prospective_protocol.py `
  --config configs/prospective_v2_revision_gate_20260810.json
```

Do not bypass a blocked status by changing the configuration status string.

This directory contains the canonical Ant-v5/PPO implementation for the ProxyGap study.
The legacy 2D grid/DQN project is retained separately as historical material.

The formal v1 experiment is complete:

- The main coefficient sweep tested control-cost weights `0.5`, `0.25`, `0.125` and `0.0625`.
- The historical exploratory forward-reweighting condition used control-cost weight `0.0625` with forward-progress weight `1.0`; it is not Proposal-conformant mitigation.
- Core reference, divergent and shaped comparisons use three training seeds; the intermediate coefficient conditions use one seed.
- Every condition uses six fixed checkpoints from 50k to 300k and ten paired deterministic evaluation episodes per checkpoint.
- Reward components and disaggregated diagnostics are logged separately from the observed proxy return.
- Pilot and formal evidence remain in separate artifact directories.

Recommended PowerShell session:

```powershell
conda activate D:\ProxyGap\envs\proxygap-ant
Set-Location D:\ProxyGap\proxygap_ant
$env:MPLCONFIGDIR = "D:\ProxyGap\matplotlib_cache"
$env:MUJOCO_GL = "disable"
python -m pytest tests
python scripts\validate_formal_outputs.py --config configs\formal_v1_coefficients_20260808.json
python scripts\validate_formal_outputs.py --config configs\formal_v1_shaped_20260808.json
python scripts\validate_formal_outputs.py --config configs\formal_v1_core_replication_20260808.json
python scripts\analyse_formal_results.py
```

Formal result tables, six report-ready figures and the concise results note are in:

```text
D:\ProxyGap\proxygap_ant\artifacts\formal\combined_v1_20260809
```

To reproduce the three paired 300k trajectory videos, use a rendering-capable backend:

```powershell
$env:MUJOCO_GL = "glfw"
python scripts\render_formal_videos.py
```

The versioned formal configurations are retained in `configs`. The runner supports
`--resume` and skips only conditions whose models, evaluation CSV and runtime CSV
are complete. Raw formal outputs should not be edited; regenerate the combined
tables and figures with `scripts\analyse_formal_results.py`.

The canonical research design, parameter-lock history and interpretation limits are
recorded in `D:\ProxyGap\PROJECT_CONTEXT.md`. The formal evidence is exploratory:
three training seeds are not sufficient for strong inferential or general robotics
claims, and Ant-v5 control effort is not a direct physical-energy measurement.
