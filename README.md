# ProxyGap: Reward Misspecification and Reward Shaping in Ant-v5

ProxyGap is a student research project using Gymnasium `Ant-v5` and
Stable-Baselines3 PPO to study whether a scalar reinforcement-learning reward
reliably represents intended quadruped locomotion on the default flat-ground
simulation task.

## Updated research direction

The project no longer treats a one-directional `ctrl_cost_weight` sweep as the
main experiment. That sweep remains historical exploratory evidence. The
current study has two linked stages:

1. **Default-reward construct audit.** Test whether the documented Ant-v5
   reward ranks independently trained policies consistently with a
   predeclared, project-specific locomotion intention.
2. **Bounded mitigation development.** Test whether targeted reward shaping,
   an external control constraint, or a training mechanism reduces the
   observed proxy-behaviour gap without materially damaging forward task
   performance.

The intended behaviour is not represented as an invented scalar "true
reward". It is evaluated through separate task, posture, direction, path,
action and termination diagnostics over a 1,000-step episode. See
[Research Direction](docs/RESEARCH_DIRECTION_20260816.md) and
[Intended Behaviour Contract](docs/INTENDED_BEHAVIOUR_CONTRACT_V2_20260816.md).

## Current scientific status

All results added on 16 August 2026 are **development evidence**. They were
used to diagnose mechanisms and refine the design; they are not held-out
formal confirmation.

- The default reward can coexist with repeated take-off, substantial flight
  time and high raw MuJoCo contact-force diagnostics.
- Orientation and lateral shaping reduced some failures but did not satisfy
  the complete intended-behaviour gate.
- An external action-slew projection constrained applied actions, while the
  PPO policy continued to propose rough actions on most steps.
- A 1 m/s target-tracking reward and an action-rate penalty improved command
  tracking and policy-output smoothness after a 1M-step development extension,
  but body-level hopping remained.
- A bounded body-dynamics penalty reduced several hopping diagnostics under
  ordinary PPO exploration. The tested gSDE setting failed in this exact
  configuration and is rejected as a development candidate, not as a general
  method.

Two questions remain open:

1. **Specified gait:** the project does not yet define or validate a crawl,
   trot, pace or bound contact-phase pattern. "Natural gait" is therefore not
   a supported outcome claim.
2. **Learned control versus guardrail dependence:** smoother applied actions do
   not prove that PPO learned a smooth policy when an external controller
   intervenes on most steps.

See [Team Progress Update](docs/TEAM_PROGRESS_UPDATE_20260816.md) for the full
plain-language summary.

The next bounded local test is declared in
[Future Testing Direction](docs/FUTURE_TESTING_DIRECTION_20260817.md). Its
protocol, configuration and executable code are public, while newly generated
models, logs, videos and result tables remain local until a separate evidence
review authorises a later release.

## Repository structure

| Path | Purpose |
|---|---|
| `src/proxygap/` | Ant-v5 wrappers, metrics, evaluation and experiment logic |
| `scripts/` | Training, evaluation, rendering, analysis and QA entry points |
| `configs/` | Versioned historical and development configurations |
| `protocols/` | Predeclared protocols, adjudications and deviation records |
| `tests/` | Automated engineering and schema tests |
| `docs/` | Research direction, metric contracts and reproducibility notes |
| `reports/` | Development audit reports with explicit claim boundaries |
| `results/development_20260816/` | Lightweight summaries, figures and video indexes |
| `presentations/` | Editable English and Chinese team-update presentations |

Large model checkpoints, compressed step logs and MP4 files are intentionally
excluded from Git. Their indexes are retained so that the corresponding local
evidence can be located and regenerated without presenting videos as
independent replications.

Future experiment plans, versioned configurations and implementation changes
are committed before or alongside execution. New experiment outputs are not
automatically published.

## Installation on Windows

Install Miniforge or Miniconda, open PowerShell in this repository, then run:

```powershell
conda env create -f environment.yml
conda activate proxygap-ant
python -m pip install -e .
python -m pytest tests
```

CUDA is not required. The project is designed for CPU execution.

## Minimum engineering verification

```powershell
python scripts/inspect_ant_reference.py
python scripts/smoke_train_benchmark.py
python -m pytest tests
```

A successful smoke test demonstrates that the pipeline runs. It does not
validate reward misspecification or mitigation.

## Research integrity boundaries

- Training seeds create independently trained policies and are the replication
  units. Evaluation episodes are paired measurements of fixed policies.
- Videos provide qualitative audit evidence only and must be selected by a
  prespecified rule.
- Raw returns from different reward functions are not automatically comparable.
- MuJoCo contact-force diagnostics are not calibrated physical safety units.
- Conclusions are limited to default flat-ground Ant-v5 with PPO and the tested
  budgets; no real-robot, terrain or external-disturbance claim is made.

## Historical material

`formal-v1` and the earlier coefficient sweep are preserved for provenance.
They can motivate the revised research question, but they must not be merged
with the new development evidence as if all runs belonged to one frozen formal
experiment.
