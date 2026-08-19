# ProxyGap: Reward Misspecification and Reward Shaping in Ant-v5

ProxyGap is a student research project using Gymnasium `Ant-v5` and
Stable-Baselines3 PPO. It studies whether the reward optimised by an agent
reliably represents a predeclared, task-appropriate quadruped locomotion
intention, and whether bounded reward shaping or external constraints can
reduce an observed proxy-behaviour gap.

## Final known-map outcome (20 August 2026)

The final audited system combines an archived bidirectional low-level expert,
the PAIR0 contact contract and a known-map waypoint planner. Two route
contracts completed **6/6 formal episodes** on one frozen continuous
heightfield, with **0 falls and 0 duration-corrected sustained-slip events**.
Time-priority, balanced and energy-priority preferences were applied only
after arrival and safety gates passed.

This is a known-map, candidate-bank result—not unseen-map generalisation,
biological gait validation, battery-energy optimisation or a mathematical
global optimum. Start with the
[overnight delivery report](docs/OVERNIGHT_OPTIMISATION_AND_DELIVERY_REPORT_20260820_CN.md),
[final scientific report](docs/V4_PAIR0_MULTIOBJECTIVE_FULL_MAP_FINAL_REPORT_20260820_CN.md),
[editable presentation draft](deliverables/ProxyGap_Final_Presentation_Draft_20260820.pptx)
and [editable report draft](deliverables/ProxyGap_Final_Report_Draft_20260820.docx).

## Start here

This repository retains two historically separated reward-research versions
and a later terrain-navigation integration. They must not be analysed as one
frozen experiment.

| Version | Role | Entry point |
|---|---|---|
| **V2 current** | Intended-gait specification, default-reward audit and bounded mitigation development | [`current/README.md`](current/README.md) |
| **Stage 2 / Project V3** | Terrain/contact diagnosis, slope and turn tests, known-map planning and multi-objective completion | [`docs/V4_PAIR0_MULTIOBJECTIVE_FULL_MAP_FINAL_REPORT_20260820_CN.md`](docs/V4_PAIR0_MULTIOBJECTIVE_FULL_MAP_FINAL_REPORT_20260820_CN.md) |
| **V1 legacy** | Historical `ctrl_cost_weight` sweep and retrospective exploratory evidence | [`legacy/weight_sweep_v1/README.md`](legacy/weight_sweep_v1/README.md) |

For a project handover, begin with [`handoff/START_HERE.md`](handoff/START_HERE.md)
and [`STATUS.md`](STATUS.md). The change from V1 to V2 is documented in
[`CHANGELOG.md`](CHANGELOG.md).

The two user-directed reward iterations trained on 17 August are recorded in
[`docs/REWARD_ITERATION_HISTORY_20260817.md`](docs/REWARD_ITERATION_HISTORY_20260817.md),
with a machine-readable, path-sanitised manifest in
[`results/development_20260817/reward_iterations/version_manifest.json`](results/development_20260817/reward_iterations/version_manifest.json).

The later pre-pitch planar translation transition is recorded in
[`docs/PLANAR_TRANSLATION_TRANSITION_20260818.md`](docs/PLANAR_TRANSLATION_TRANSITION_20260818.md),
with a path-sanitised manifest in
[`results/development_20260818/planar_translation_transition/version_manifest.json`](results/development_20260818/planar_translation_transition/version_manifest.json).

The subsequent local curved-gait development (translation plus torso yaw,
without route-position learning) is recorded in
[`docs/CURVED_GAIT_TRAINING_20260818.md`](docs/CURVED_GAIT_TRAINING_20260818.md),
with a lightweight result record in
[`results/development_20260818/curved_gait/selection_summary.json`](results/development_20260818/curved_gait/selection_summary.json).

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
- The final integration completed 6/6 formal reset-seed episodes on one known
  frozen map; it is not an independent training-seed or unseen-map study.
- The selected routes are preferred within 15 evaluated feasible candidates;
  positive mechanical work is a simulation proxy rather than battery energy.
- No real-robot or biological-gait claim is authorised. Representative final
  episodes still contain complete control intervals with no foot contact.

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

The final release deliberately includes a narrow, hash-audited set of the
checkpoint, formal traces and MP4 evidence needed to reproduce the reported
known-map result. Smoke, failed, intermediate, cache, recovery and
machine-specific files remain excluded. Evaluation episodes are nested reset
measurements and must not be described as independent training replications.

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
- Stage-1 conclusions are limited to the tested Ant-v5/PPO configurations;
  final Stage-2 conclusions are limited to the frozen known map, controller,
  candidate bank and reset seeds.

## License

ProxyGap is open-source software released under the [MIT License](LICENSE).
