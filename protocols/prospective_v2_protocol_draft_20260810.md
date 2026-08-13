# ProxyGap Prospective v2 Protocol

**Draft date:** 10 August 2026  
**Status:** revision gate open; not frozen; do not run the timed pilot  
**Controlling proposal:** Proposal_G6, SHA-256 `4E78D9803FDB7F7A8176702E4BB20FC382E3322D5B3DE1E8414FBCF1BCE67AD9`  
**Machine-readable record:** `configs/prospective_v2_revision_gate_20260810.json`

## 1. Separation from formal v1

Prospective v2 is a new development and held-out evaluation study. Formal-v1 models, pilot rows, evaluation seeds and condition-selection evidence are historical inputs only. No historical input will be pooled with v2 outcomes. The formal-v1 analysis is controlled by `formal_v1_retrospective_analysis_20260810.md`.

This draft cannot be called preregistered because the broader project and `0.0625` condition were previously observed. Once the remaining decisions are approved, the frozen v2 file, resolved configuration, source manifest, timestamp and SHA-256 will be recorded before the first timed-pilot output is generated.

## 2. Research questions

### Primary diagnostic question

Under held-out optimiser and reset seeds, how does training with `ctrl_cost_weight=0.0625` rather than the default `0.5` comparator affect final-checkpoint net forward progress, a common counterfactual return scored at `ctrl_cost_weight=0.5`, and the prespecified protected diagnostic vector?

### Intervention question

Does one frozen bounded effort/orientation intervention improve net forward progress relative to unshaped `0.0625`, and what changes occur in every protected diagnostic?

Unless a margin-based analysis route is approved before the timed pilot, the intervention question remains descriptive. The words *effective mitigation*, *non-inferior* and *without harm* will then be prohibited.

## 3. Conditions

The mandatory core is:

1. `C0`: default benchmark comparator, `ctrl_cost_weight=0.5`;
2. `C1`: selected reduced-cost condition, `ctrl_cost_weight=0.0625`, unshaped;
3. `C2`: one bounded combined effort/orientation intervention at `ctrl_cost_weight=0.0625`.

The proposal requires one mitigated condition and does not require component attribution. The recommended resource-aware route is therefore `combined_only_no_component_attribution`. If component attribution is required, add:

4. `C2E`: effort-only intervention;
5. `C2O`: orientation-only intervention;
6. retain `C2` as the combined intervention.

The attribution choice remains a user decision. No report may infer an effort-specific or orientation-specific mechanism from the combined-only route.

## 4. Intervention formula

The supported bounded terms are:

```text
effort_penalty_t
  = -lambda_effort * tanh(sum_j(a_t,j^2) / scale_effort)

orientation_penalty_t
  = -lambda_orientation * tanh(torso_tilt_rad / scale_orientation)
```

The combined condition adds both terms to the unshaped `0.0625` objective. Forward-reward reweighting is excluded from v2.

The current recommendations are `scale_effort=2.0`, `scale_orientation=0.5 rad` and a per-component cap of `0.25 reward units/step`. The proposed values are transparent design choices based on the fixed action and angle domains, not established scientific constants. Approval is required before protocol freeze.

## 5. Outcomes

The primary diagnostic is `net_forward_progress` at the final target. The reward measures are:

- `condition_objective_return`, which describes the reward actually optimised by each policy;
- `common_rescored_return`, which applies the fixed default control-cost weight `0.5` to every recorded evaluation trajectory.

The protected vector is:

- low-z collapse rate;
- high-z excursion rate;
- non-finite termination rate;
- mean and maximum absolute lateral offset;
- torso-tilt RMS and 95th percentile;
- mean squared action per step;
- action saturation rate at `|a_j| >= 0.95`;
- episode length.

Cumulative squared action, progress per step, final lateral offset, cumulative lateral path, torso-tilt mean/standard deviation/maximum, survival reward and all base reward components remain explanatory diagnostics.

The exact formula, unit, direction and aggregation for every field are controlled by `docs/METRIC_DEFINITIONS_V2_20260810.md`.

## 6. Seeds and pairing

The partitions are frozen and mutually disjoint:

| Role | Seeds |
|---|---|
| Timed-pilot training | `20260811`, `20260812` |
| Timed-pilot evaluation | `60260810`-`60260814` |
| Held-out training | `20260821`-`20260825` |
| Held-out evaluation | `70260810`-`70260819` |

Training seeds are reused across conditions within a stage. Evaluation reset seeds are reused across conditions and checkpoints within a stage as common random numbers. Timed-pilot rows are excluded from the held-out dataset.

## 7. Timed-pilot design

The timed pilot is blocked until the revision gate closes. Once authorised, every condition uses 50,000 target steps, checkpoint targets at 25,000 and 50,000, two paired development seeds, five deterministic evaluation episodes per checkpoint, the complete resolved PPO configuration and CPU-only execution.

| Scope | Conditions | Independent policies | Checkpoints | Evaluation episodes | Estimated active time | Estimated model storage |
|---|---:|---:|---:|---:|---:|---:|
| Combined-only | 3 | 6 | 12 | 60 | 10.2 min | 3.5 MB |
| Attribution ablation | 5 | 10 | 20 | 100 | 17.0 min | 5.9 MB |

Estimates use `537.495 actual steps/s`, expected rollout completion at approximately 51,200 steps per policy and `0.702076 s/evaluation episode`. The figures are planning estimates rather than guarantees.

The pilot evaluates implementation integrity, reward and shaping scale, tanh saturation, diagnostic finiteness, denominator sensitivity, catastrophic failures, replay, storage, runtime and the stability of within-policy episode means. It cannot establish coefficient thresholds, mitigation efficacy, no-harm, formal reward hacking or generalisation.

## 8. Held-out design

The primary endpoint is the 300k target. Targets at 50k, 100k, 150k, 200k, 250k and 300k are dependent secondary observations. `actual_model_timesteps` is reported alongside every target.

The combined-only route contains 15 independent policies, 90 model checkpoints and 900 deterministic evaluation episodes. Its measured planning estimate is 150.6 active CPU minutes and 26.5 MB of model checkpoints. The five-condition attribution route contains 25 policies, 150 checkpoints and 1,500 episodes, with an estimate of 250.9 minutes and 44.1 MB.

Five training seeds are a resource-aware student-project target, not proof of broad robustness. Every seed-level result and a leave-one-seed-out sensitivity summary will be reported.

## 9. Analysis

Evaluation episodes are averaged within each `condition x training_seed x checkpoint` policy before any across-seed summary. Paired differences are calculated within training seed. Episodes and checkpoints never increase the independent seed count.

The primary endpoint is fixed at 300k. Checkpoint profiles answer a secondary temporal-emergence question. No best checkpoint or seed may replace the primary endpoint.

The report will show condition-objective return, common-rescored return and the complete diagnostic vector separately. No scalar `true_performance` score will be constructed. A candidate that improves progress but crosses a protected harm boundary is not eligible for a no-displacement claim.

Two analysis routes are possible:

1. **Descriptive-only:** report paired effects and uncertainty without declaring successful mitigation or no-harm. Practical margins are not required, but claim strength is explicitly restricted.
2. **Margin-based mitigation:** freeze a smallest meaningful progress improvement and a harm margin for every protected metric before the timed pilot. The candidate must satisfy the complete rule on held-out seeds.

The analysis route and any margins remain unresolved user decisions.

## 10. Recording and reproducibility

Every run must save the resolved configuration, package lock, hardware record, source hashes, model checkpoint, Stable-Baselines3 Monitor file, runtime row, episode-level evaluation CSV and one compressed UTF-8 step CSV per evaluation episode. Failed, interrupted and resumed attempts remain in separate auditable records.

The complete PPO configuration is:

```text
MlpPolicy; pi=[64,64]; vf=[64,64]; Tanh; Adam
n_steps=2048; batch_size=64; n_epochs=10; learning_rate=3e-4
gamma=0.99; gae_lambda=0.95; clip_range=0.2
ent_coef=0; vf_coef=0.5; max_grad_norm=0.5
normalize_advantage=True; device=cpu
```

All parameters are read from and persisted in the resolved configuration. Code defaults may support historical v1 loading but cannot substitute for a complete schema-v2 configuration.

## 11. Video rule

For each condition, select the training seed closest to the median final-checkpoint policy mean progress, then select the evaluation seed closest to that policy's median final-checkpoint episode progress. Lower numeric seeds break exact ties. Reuse the selected evaluation seed at 50k, 150k and 300k. Videos remain illustrations rather than independent evidence.

## 12. Stopping, exclusion and recovery

Stop a run for audit if:

- in-memory reward reconciliation exceeds `1e-8`, or CSV reconstruction exceeds `1e-4`;
- required rows, models, monitors, step logs or resolved configuration fields are missing;
- required finite metrics, observations, actions or parameters contain non-finite values;
- environment, architecture, budget or evaluation seeds differ across conditions without a registered deviation;
- checkpoint overshoot reaches one full 2,048-step rollout;
- deterministic replay fails on the recorded machine and environment;
- free disk capacity falls below 20 GB;
- a partial run cannot be distinguished from a completed run.

A non-finite state is retained as a catastrophic termination record and triggers technical review. Low-z or high-z termination does not by itself justify deleting a seed; it is a scientific outcome.

Formal training may not stop, rerun or exclude a seed because results are negative, mixed or inconvenient. Only traceable technical corruption permits a rerun, and the failed record must remain. A time-cap fallback must trigger before condition outcomes are inspected and must reduce the claim to a descriptive case study.

## 13. Current freeze gate

The following blockers remain open:

1. approve either descriptive-only or margin-based analysis;
2. if margin-based, approve the progress and protected harm margins;
3. approve `effort_distance_min`;
4. approve the intervention scales and caps;
5. approve combined-only or attribution-ablation scope.

Until these decisions are recorded in the machine-readable configuration, the status remains `revision_gate_open_do_not_run`. Neither the 50k timed pilot nor held-out training is authorised.
