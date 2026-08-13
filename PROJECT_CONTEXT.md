# ProxyGap — Canonical Project Context

> **Status:** Canonical handover record for all subsequent implementation, experimentation, analysis, reporting and presentation work.  
> **Last consolidated:** 8 August 2026  
> **Rule:** When any earlier prompt, file or chat conflicts with this document, this document takes precedence unless the user explicitly approves a later revision.

## 1. Canonical project identity

**Working group name:** ProxyGap  
**Group slogan:** *Mind the Gap Between Reward and Reality.*

**Canonical academic title:**  
*When Higher Reward Means Worse Locomotion: Diagnosing and Mitigating Reward Misspecification in a PPO-Controlled Quadruped*

**Course context:** *Deep Reinforcement Learning and Robotic Automation*, Cambridge summer programme. The project must remain recognisably based on deep reinforcement learning and robotic automation. A purely two-dimensional grid task was rejected because it would not adequately represent the chosen course theme or continuous robotic locomotion.

**Core research aim:**  
To determine whether a plausible misspecification of the control-cost term in a PPO-controlled simulated quadruped allows proxy return to improve while locomotion quality deteriorates, and to evaluate whether a bounded reward-shaping intervention mitigates the divergence without transferring optimisation pressure to another undesirable behaviour.

**Scope statement suitable for academic outputs:**  
*In a controlled quadruped simulation, the study will demonstrate how reward misspecification can emerge under extended optimisation and will evaluate whether a bounded, scale-normalised shaping intervention can reduce the resulting proxy–performance divergence across a disaggregated set of locomotion measures.*

## 2. Research questions and hypotheses

### Primary research question

Under what control-cost weighting and stage of PPO training does episodic proxy return cease to provide reliable evidence of desirable quadruped locomotion?

### Secondary research question

Can a bounded, scale-normalised reward-shaping intervention reduce the observed divergence without creating a new failure in an insufficiently constrained dimension?

### Working hypotheses

1. Plausible under-penalisation of control effort will initially improve forward motion and proxy return, but sufficiently strong optimisation may favour inefficient or unstable locomotion.
2. Longer training may amplify exploitation of an imperfect proxy: proxy return may continue to rise while control effort, stability, fall rate or lateral drift deteriorates.
3. A bounded shaping intervention may reduce the original failure; however, success must be judged empirically because the intervention may transfer the failure to inactivity, lateral motion, jumping or another behaviour.
4. A higher scalar reward is not, by itself, evidence of better task performance.

These hypotheses are directional but not predetermined conclusions. Negative or mixed findings remain valid if the method and evidence are reported honestly.

## 3. Conceptual framework

### Reward misspecification

PPO optimises the numerical reward supplied during training; it does not infer the designer's verbal intention. A locomotion reward commonly combines forward motion, survival or healthy-state reward, control cost and contact cost. If one term is underweighted, a policy can satisfy the scalar objective through behaviour that is fast but unstable, energetically inefficient or mechanically implausible.

The project concerns **designer-specified proxy failure exploited by optimisation**. The coefficient choice originates from human specification, while the learned exploit is discovered by the agent. This is not a study of simulator bugs, sensor corruption or an autonomous algorithmic “mistake” independent of reward design.

### Goodhart-style concern

When a proxy measure becomes the optimisation target, it may cease to remain a reliable measure of the broader objective. This does not imply that reward design is impossible. It implies that the scalar training objective must be audited against separately reported task attributes and behavioural evidence.

### Training reward versus external evaluation

Speed, control effort and stability can appear both in reward design and in evaluation, but their roles differ:

- During training, selected quantities are weighted and collapsed into the single scalar that PPO optimises.
- During evaluation, quantities are reported separately, and where practical are calculated differently or held out from the training reward.
- The diagnostic scorecard is not described as an objectively “true reward”. It provides evidence about trade-offs and side effects hidden by the scalar return.

No chosen metric is inherently identical to the designer's full intention. The strength of the evaluation lies in transparent, disaggregated evidence rather than another arbitrarily weighted aggregate.

## 4. Canonical experimental platform

- **Simulator/environment:** Gymnasium MuJoCo `Ant-v5`.
- **Learning algorithm:** Proximal Policy Optimisation (PPO), implemented through Stable-Baselines3 unless a documented compatibility reason requires otherwise.
- **Robot:** simulated quadruped only; no physical quadruped is required for the Cambridge project.
- **Compute baseline:** local Windows laptop, CPU-only. CUDA is unavailable and must not be assumed.
- **ROS, vision and local perception deployment:** outside the core Cambridge experiment. These components relate to the later university robotics-team selection, but are not required to establish reward misspecification in Ant-v5. They may be described as future integration work, not as completed Cambridge-project contributions.
- **Isaac Sim/Isaac Lab:** not part of the baseline because of local hardware requirements and project-time risk.
- **AI Studio or other cloud GPU services:** optional only; local CPU feasibility has already been demonstrated, so cloud availability must not become a dependency.

## 5. Experimental design — fixed decisions

### 5.1 Independent variable

Manipulate **one reward coefficient only** in the primary study: `ctrl_cost_weight` (the control-effort penalty). Keep the forward-reward coefficient and all other environment and PPO settings fixed.

This one-dimensional design was chosen to reduce confounding and support causal interpretation. A two-dimensional reward sweep is not part of the core project.

### 5.2 Experimental conditions

The intended minimal design contains five conditions:

1. the Ant-v5 reference reward condition;
2. three progressively reduced control-cost penalties within a plausible range around the reference configuration;
3. one shaped/mitigated condition applied to a setting that has demonstrated reliable proxy–performance divergence during the pilot.

The misspecified condition must not be an obviously absurd or zero-cost straw man. The coefficient range should remain plausible enough that the resulting failure could arise during ordinary reward tuning.

### 5.3 Controlled variables

Across conditions, retain the same:

- PPO policy architecture;
- optimiser and learning hyperparameters;
- total training budget;
- environment version;
- observation and action spaces;
- evaluation episodes and initialisation policy;
- primary random seed;
- checkpoint schedule;
- logging definitions.

Any unavoidable difference must be documented before results are interpreted.

### 5.4 Training checkpoints

Save and evaluate policies at approximately 25%, 50%, 75% and 100% of the total training budget. Do not evaluate only the final policy. The checkpoints are necessary to test whether extended optimisation amplifies divergence or produces a behavioural phase transition.

### 5.5 Random seeds

A fixed primary seed is sufficient for the time-constrained core comparison. Additional seeds are a robustness extension rather than the primary deliverable. A single seed cannot justify broad statistical generalisation, and that limitation must be stated. If compute permits, add seeds only after the full core pipeline and all required figures have been validated.

## 6. Reward shaping — fixed principles and open formula

The mitigation will use bounded, scale-normalised terms addressing excessive control effort and instability. Its purpose is not to maximise an alternative arbitrary score, but to test whether a targeted intervention reduces the diagnosed failure.

The exact shaping equation and coefficients are **not yet finalised**. They must be locked only after a pilot identifies a reproducible divergent condition. The shaping design must follow these constraints:

- preserve the same PPO implementation and training budget;
- avoid unbounded terms whose scale dominates the base reward;
- normalise quantities with substantially different numerical ranges;
- modify only terms justified by the diagnosed behaviour;
- inspect whether the mitigation transfers failure to inactivity, lateral drift, jumping or another dimension;
- distinguish practical engineering shaping from potential-based shaping.

Potential-based shaping, `F(s,a,s') = gamma * Phi(s') - Phi(s)`, provides a theoretical policy-invariance condition. The proposed practical multi-term shaping is not automatically policy invariant and must not be described as possessing that guarantee unless the final implementation satisfies the theorem's assumptions.

## 7. Evaluation protocol

### 7.1 Training signal

Record episodic **proxy return**, including its component terms where available. Proxy return is what PPO optimises.

### 7.2 Diagnostic outcome measures

Report the following separately rather than collapsing them into a second scalar “true performance” score:

1. **Net forward progress** or an equivalent forward-locomotion measure.
2. **Control effort per unit distance**, used as an energy proxy rather than a direct physical energy measurement.
3. **Fall rate** or unhealthy termination rate under a clearly fixed definition.
4. **Lateral drift**.
5. **Torso-orientation variability** or another reproducible stability measure.
6. **Episode length/survival information** where necessary to interpret fall and movement results.
7. **Behaviour video** at representative checkpoints to audit violent leg motion, jumping, unstable sprinting, inactivity or other qualitative failures.

Metric formulae, units, sign conventions and aggregation rules must be written down before the formal experiment. Evaluation should use held-out episodes and deterministic actions where appropriate. Training and evaluation logs must not be mixed.

### 7.3 Interpretation rule

A condition constitutes evidence of proxy–performance divergence only when proxy return improves while one or more predeclared diagnostic outcomes deteriorate to a meaningful degree, supported by behavioural inspection. A visually unusual gait alone is insufficient, and a metric difference without comparable evaluation conditions is insufficient.

### 7.4 Required comparisons

- reward coefficient versus final proxy return;
- reward coefficient versus each diagnostic metric;
- training checkpoint versus proxy return and each diagnostic metric;
- baseline/reference versus divergent condition;
- divergent condition versus shaped condition;
- videos or trajectory summaries supporting quantitative interpretation.

## 8. Parameters that remain to be locked

Do not silently invent these values. Determine them through a short, documented pilot, then record them in the experiment configuration and update this context:

- exact four control-cost coefficient values (reference plus three reductions);
- exact total timesteps for formal training;
- PPO rollout and batch settings if the Stable-Baselines3 defaults are changed;
- exact primary seed value;
- number of evaluation episodes per checkpoint;
- operational thresholds for “fall” and any instability statistic;
- exact shaping formula and coefficients;
- whether one or more extra seeds fit the time budget.

Pilot choices must be made before formal results are generated. Once locked, do not tune them after inspecting final results without labelling the later run as exploratory.

## 9. Feasibility, time and deliverables

### 9.1 Available project time

The conservative project window contains eight full working days: Week 1 Tuesday–Friday and Week 2 Monday–Thursday. Optional weekend compute may be used, but the plan must not depend on every group member being available during the weekend. The final two working days are reserved for collective integration, checking and finalisation.

### 9.2 Timeline

- **Days 1–2:** freeze scope, metric definitions, environment wrapper and logging; validate reference PPO run.
- **Days 3–4:** conduct coefficient pilot/sweep and checkpoint training; identify a reproducible divergent condition.
- **Days 5–6:** implement and train the shaping condition; complete held-out evaluation and behavioural audit.
- **Days 7–8:** all members finalise analysis, report, presentation, figures, limitations and reproducibility evidence.

### 9.3 Minimum viable research deliverables

- reproducible source code and locked configuration files;
- environment and dependency record;
- raw training and evaluation CSV files;
- saved policies at planned checkpoints;
- learning curves and coefficient/metric comparison figures;
- representative behaviour videos;
- experiment report in academic English;
- presentation slides and a corresponding speech script;
- limitations and reproducibility statement.

The technical pipeline should be validated in advance. A complete technical draft may be generated rapidly once the pipeline is fixed, but final scientific credibility still requires the user's understanding, interpretation and revision.

## 10. Current local deployment state

The following work was completed and verified on 7 August 2026:

- **Project root:** `D:\ProxyGap`
- **Legacy project copy:** `D:\ProxyGap\drl_reward_misspec_robotics`
- **Conda environment:** `D:\ProxyGap\envs\proxygap-ant`
- **Conda package cache:** `D:\ProxyGap\conda_pkgs`
- **pip cache:** `D:\ProxyGap\pip_cache`
- **MuJoCo/cache directory:** `D:\ProxyGap\mujoco_cache`
- **Matplotlib cache:** `D:\ProxyGap\matplotlib_cache`
- **Benchmark logs:** `D:\ProxyGap\logs`
- **Smoke-test models:** `D:\ProxyGap\models`

Verified environment:

- Python 3.11
- PyTorch 2.13.0+cpu
- Gymnasium 1.3.0
- Stable-Baselines3 2.9.0
- MuJoCo 3.11.0
- NumPy 2.4.6
- pandas 3.0.5
- Matplotlib 3.11.1
- pytest 9.1.1
- CUDA unavailable, as expected

Verified Ant-v5 interface:

- observation shape: `(105,)`
- action shape: `(8,)`
- environment creation, reset and step succeeded.

Verified PPO CPU smoke tests:

- 1,024 timesteps: approximately 1,358.8 steps/s;
- 8,192 timesteps: approximately 1,340.36 steps/s;
- benchmark logs: `D:\ProxyGap\logs\ant_v5_ppo_smoke_benchmark.csv` and `D:\ProxyGap\logs\ant_v5_ppo_speed_benchmark.csv`;
- smoke model: `D:\ProxyGap\models\ppo_ant_v5_smoke.zip`.

These measurements establish basic local feasibility but do not yet establish full-experiment duration, because formal evaluation, checkpointing, video rendering and repeated conditions add overhead.

Recommended PowerShell session settings:

```powershell
conda activate D:\ProxyGap\envs\proxygap-ant
Set-Location D:\ProxyGap
$env:MPLCONFIGDIR = "D:\ProxyGap\matplotlib_cache"
$env:MUJOCO_GL = "disable"
```

Use a rendering-capable MuJoCo setting rather than `disable` only when videos are generated; confirm the appropriate Windows backend during the video smoke test.

## 11. Legacy material — retained but superseded

The directory `D:\ProxyGap\drl_reward_misspec_robotics` contains an earlier 2D grid-world/DQN prototype. That prototype remains useful as coding practice and historical evidence, but it is **not the canonical Cambridge experiment**.

The following earlier decisions are explicitly superseded:

- custom 2D grid environment as the main robotic task;
- DQN as the primary algorithm;
- aligned/misspecified/shaped grid-cell reward modes as the formal experiment;
- prohibition of MuJoCo;
- use of a single aggregate `true_performance` scalar as the principal external criterion;
- ROS, Gazebo or visual-local-deployment work as a required part of the Cambridge reward-misspecification experiment.

Do not delete the legacy directory without explicit user approval. New Ant-v5/PPO implementation should be placed in a clearly named canonical source directory under `D:\ProxyGap`, or the legacy directory should be renamed only after approval.

## 12. Relationship to the university robotics-team selection

The broader personal plan seeks one body of work that supports three goals:

1. completion of the Cambridge DRL and robotic-automation project;
2. evidence for a computer-science transfer interview;
3. preparation for a university robotics-team quadruped challenge.

Relevant dates previously provided:

- Cambridge online examination: 4 August 2026;
- Cambridge in-person course begins: 10 August 2026, followed by an approximately two-week project;
- programme-transfer deadline/interview preparation: 28 August 2026;
- quadruped add-on challenge deadline: 12 September 2026;
- vision-group assessment deadline: 7 October 2026.

The Ant-v5/PPO study provides conceptual and technical preparation for quadruped control, reward design and experimental evaluation. It does **not** by itself complete the robotics-team vision tasks, local perception deployment or ROS integration. Those later tasks may reuse Python, experimental practice and quadruped concepts, but should be managed as a subsequent workstream rather than being inserted into the eight-day Cambridge core.

## 13. User background and collaboration requirements

- Current preparation includes four machine-learning lectures, two supervision sessions and Chapter 14 of Qiu Xipeng's *Neural Networks and Deep Learning*; a practice book is also available.
- The user is an early undergraduate and requires conceptual explanations bridging the gap between current knowledge and PPO/MuJoCo implementation.
- The assistant should first establish that every technical step works, then explain each step clearly enough for the user to understand, challenge and revise it.
- Progress must not be presented as the user's independent understanding until the user has reviewed the reasoning and results.
- Explanations should combine plain Chinese with precise English terminology where useful. Formal academic artifacts must use academic British English.

## 14. Academic writing rules

All formal proposal, report, slide and script text must:

- use academic British English;
- avoid first-person `I` as the grammatical subject;
- avoid vague verbs such as `make` when a precise verb is available;
- avoid overstating novelty, causality, generality or real-world transfer;
- distinguish simulation evidence from physical-robot evidence;
- distinguish control effort from directly measured energy;
- distinguish proxy return from disaggregated diagnostic measures;
- report limitations, negative findings and exploratory changes transparently;
- follow the supplied Harvard referencing guidance consistently.

The contribution should be framed as a controlled, reproducible case study rather than a novel PPO algorithm or the first use of reward shaping in locomotion.

## 15. Team allocation recorded in the proposal

There are eight unique members. The current A/B/C allocations are provisional and may overlap:

- **Group A — research design and reward specification:** Mingqian Chai, Yunxi An, Miaoxi Song, Wenjie Guo.
- **Group B — implementation and training:** Yunxi An, Chuhan Shang, Yuyuan Sun, Qixiang Huang.
- **Group C — evaluation and analysis:** Yiyang Chen, Mingqian Chai, Chuhan Shang, Miaoxi Song.
- **Final two working days:** all members complete integration, checking, report and presentation finalisation.

Concise proposal wording:

*Group A will lead reward design and literature review; Group B will implement the Ant-v5/PPO pipeline and conduct training; Group C will develop evaluation metrics and analyse results. Days 1–4 will cover environment validation, baseline training and the reward-coefficient sweep. Days 5–6 will focus on reward shaping and evaluation. All members will use Days 7–8 to finalise the report, presentation and supporting evidence.*

## 16. Proposal and supporting artifacts

Artifacts created during project scoping remain in the earlier Codex workspace and should be treated as planning evidence:

- completed proposal draft: `C:\Users\18522\Documents\Codex\2026-07-31\referenced-chatgpt-conversation-this-is-an\AI+ Project Proposal - 2026 RL - completed.docx`;
- final eight-day Gantt chart: `C:\Users\18522\Documents\Codex\2026-07-31\referenced-chatgpt-conversation-this-is-an\quadruped_reward_misspecification_gantt_v4.png`.

The Word file may still contain the older timeline paragraph and earlier reference formatting. Before final submission, replace the timeline with the concise wording above and apply the reference corrections listed below.

## 17. Core reading and reference set

Use targeted reading rather than attempting to read every paper line by line. A separate annotated reading guide should be produced only when the user says **“产出阅读资料”**.

Core conceptual sequence:

1. DeepMind, *Specification Gaming* — behavioural intuition for agents exploiting reward rules.
2. *AI Safety Gridworlds* — separation of visible reward and a performance function.
3. Pan et al. — effects of reward misspecification, optimisation power/training time and behavioural transitions.
4. Skalse et al. — formal definition and characterisation of reward hacking.
5. Amodei et al., *Concrete Problems in AI Safety* — wrong objectives, reward hacking and negative side effects.
6. Ng et al. — policy invariance and potential-based reward shaping.
7. Qiu Xipeng, Chapter 14 — MDPs, policy gradient, REINFORCE and actor–critic foundations.
8. Schulman et al. — PPO and the clipped objective.
9. GAE paper — optional after actor–critic foundations; mathematically more demanding.

Harvard-style references already checked for the proposal:

1. Farama Foundation (2026) *Ant*. Available at: https://gymnasium.farama.org/environments/mujoco/ant/ (Accessed: 7 August 2026).
2. Ng, A.Y. et al. (1999) ‘Policy invariance under reward transformations: theory and application to reward shaping.’ *Proceedings of the Sixteenth International Conference on Machine Learning (ICML 1999)*. Bled, Slovenia, 27–30 June. San Francisco, CA: Morgan Kaufmann, pp. 278–287.
3. Pan, A. et al. (2022) ‘The effects of reward misspecification: mapping and mitigating misaligned models.’ *Tenth International Conference on Learning Representations*. Virtual, 25–29 April. Available at: https://openreview.net/forum?id=JYtwGwIL7ye (Accessed: 7 August 2026).
4. Schulman, J. et al. (2017) ‘Proximal policy optimization algorithms.’ *arXiv*, arXiv:1707.06347. Available at: https://arxiv.org/abs/1707.06347 (Accessed: 7 August 2026).
5. Skalse, J. et al. (2022) ‘Defining and characterizing reward hacking.’ *arXiv*, arXiv:2209.13085. Available at: https://arxiv.org/abs/2209.13085 (Accessed: 7 August 2026).
6. Todorov, E. et al. (2012) ‘MuJoCo: a physics engine for model-based control.’ *2012 IEEE/RSJ International Conference on Intelligent Robots and Systems*. Vilamoura, Algarve, Portugal, 7–12 October. Piscataway, NJ: IEEE, pp. 5026–5033. doi: 10.1109/IROS.2012.6386109.
7. Optional reproducibility reference: Henderson, P. et al. (2018) ‘Deep reinforcement learning that matters.’ *Proceedings of the AAAI Conference on Artificial Intelligence*, 32(1), pp. 3207–3214. doi: 10.1609/aaai.v32i1.11694.

Use `et al.` consistently in in-text citations for works with three or more authors under the selected Harvard guide.

## 18. Immediate next actions for the ProxyGap implementation task

Proceed in this order:

1. Read this complete file before modifying project code.
2. Audit `D:\ProxyGap` and preserve all existing environments, benchmark logs, models and legacy files.
3. Create a clean canonical Ant-v5/PPO project structure distinct from the legacy DQN grid project.
4. Implement an environment wrapper or logging layer that records each reward component and every predeclared diagnostic metric.
5. Add automated tests for reward decomposition, metric calculation, termination handling, seeding and CSV schema.
6. Run a short rendering/video smoke test separately from the headless training test.
7. Conduct a pilot to lock coefficient levels, formal timesteps, evaluation episode count, primary seed and shaping definition.
8. Write the locked decisions back into this document and versioned configuration files before formal training.
9. Estimate end-to-end runtime including evaluation, checkpointing and rendering, not merely raw training steps.
10. Present the pilot design and runtime estimate to the user for comprehension and approval before initiating the complete formal experiment.

## 19. Non-negotiable safeguards

- Preserve all existing user files and environments; do not delete or overwrite legacy material without explicit approval.
- Do not describe smoke-test outputs as research findings.
- Do not invent unapproved coefficient values, metric thresholds or shaping formulae.
- Do not tune the final method retrospectively without labelling the change exploratory.
- Do not equate simulation success with real-robot deployment or sim-to-real transfer.
- Do not allow the old 2D/DQN prompt to override the Ant-v5/PPO design.
- Keep raw data, derived data, figures, models and configuration files clearly separated and reproducible.
- Record package versions, seeds, commands, dates and configuration identifiers for every formal run.


## 20. Deployment and pilot evidence added on 8 August 2026

This section records implementation and pilot evidence only. It does not convert smoke-test or pilot outputs into formal research findings.

Completed canonical deployment facts:

- Created canonical Ant-v5/PPO implementation directory: `D:\ProxyGap\proxygap_ant`.
- Preserved the legacy 2D/DQN directory: `D:\ProxyGap\drl_reward_misspec_robotics`.
- Implemented a `ProxyGapAntWrapper` for Gymnasium `Ant-v5` that records reward decomposition and disaggregated diagnostic metrics.
- Implemented 25/50/75/100% checkpoint support through the canonical training helper.
- Implemented deterministic held-out evaluation logging with separate CSV output from training runtime logs.
- Added automated tests for reward coefficient application, metric calculation, seeding, checkpoint targets and CSV schema coverage.
- Verified automated tests on 8 August 2026: `9 passed`.
- Verified headless canonical PPO smoke benchmark: `D:\ProxyGap\proxygap_ant\artifacts\logs\canonical_smoke_train_benchmark.csv`.
- Verified short MuJoCo rendering smoke test and saved: `D:\ProxyGap\proxygap_ant\artifacts\videos\ant_v5_render_smoke.gif`.

Pilot evidence saved on 8 August 2026:

- `D:\ProxyGap\proxygap_ant\artifacts\pilot\coefficient_pilot_20260808_local_small`
- `D:\ProxyGap\proxygap_ant\artifacts\pilot\coefficient_pilot_20260808_20k`
- `D:\ProxyGap\proxygap_ant\artifacts\pilot\learning_sanity_reference_50k`
- `D:\ProxyGap\proxygap_ant\artifacts\pilot\coefficient_pilot_20260808_50k_fullppo`
- Summary note: `D:\ProxyGap\proxygap_ant\artifacts\pilot\PILOT_SUMMARY_20260808.md`
- Recommendation config requiring user approval: `D:\ProxyGap\proxygap_ant\configs\pilot_recommendation_20260808.json`

Interpretation of pilot evidence:

- The 4,096-step and 20,000-step coefficient pilots were useful for pipeline and runtime checks, but were not sufficient for scientific parameter locking because proxy return was dominated by survival reward and locomotion was weak.
- A 50,000-step reference-only sanity run with fuller PPO update settings showed movement and instability/fall signals, indicating that stronger PPO updates are required for meaningful pilot evidence.
- The most useful current coefficient evidence is `coefficient_pilot_20260808_50k_fullppo`, which tested `[0.5, 0.25, 0.125, 0.0625]` using `n_steps=2048`, `batch_size=64`, `n_epochs=10` and CPU.
- The 75% checkpoint of `ctrl_cost_weight=0.25` is the clearest current candidate for proxy-performance divergence, because proxy return and forward progress were high while lateral drift was also high. This remains pilot evidence and must be confirmed before the shaping formula is locked.
- The shaping condition remains unlocked. No shaping equation or shaping coefficient has been approved.

Runtime estimate from the 50k full-PPO pilot:

- measured pilot training speed: approximately 577.49 steps/s;
- measured pilot evaluation speed: approximately 0.704 seconds per evaluation episode;
- estimated 5-condition run at 100k timesteps/condition with 5 eval episodes/checkpoint: approximately 936.3 seconds;
- estimated 5-condition run at 200k timesteps/condition with 5 eval episodes/checkpoint: approximately 1802.1 seconds.

Recommended but not yet user-approved choices:

- coefficient set: `[0.5, 0.25, 0.125, 0.0625]`;
- primary seed: `20260808`;
- PPO settings: `n_steps=2048`, `batch_size=64`, `n_epochs=10`, CPU;
- evaluation episodes per checkpoint: `5`;
- formal training budget option A: `100000` timesteps per condition;
- formal training budget option B: `200000` timesteps per condition if time allows.

These recommendations require user approval before formal training. If approved, the next step is to create locked versioned configuration files and then run the formal reference/reduced-coefficient conditions before defining the shaping condition.

## 21. Formal v1 coefficient-sweep decisions locked on 8 August 2026

The user approved the following formal design before any formal training output was generated:

- formal coefficient-sweep config: `D:\ProxyGap\proxygap_ant\configs\formal_v1_coefficients_20260808.json`;
- environment and algorithm: Gymnasium `Ant-v5` with Stable-Baselines3 PPO on CPU;
- primary independent variable: `ctrl_cost_weight` only;
- coefficient conditions: reference `0.5` plus reduced values `0.25`, `0.125` and `0.0625`;
- primary training seed: `20260808`;
- paired evaluation seed base: `30260808`, reused across conditions and checkpoints;
- training budget: `300000` timesteps per condition;
- fixed checkpoint targets: `50000`, `100000`, `150000`, `200000`, `250000` and `300000` timesteps;
- held-out deterministic evaluation: `10` episodes per checkpoint, with a maximum of `1000` steps per episode;
- PPO settings: `n_steps=2048`, `batch_size=64`, `n_epochs=10`, learning rate `0.0003`, CPU device;
- primary endpoint: the predeclared `300000`-timestep checkpoint;
- secondary analysis: all six checkpoint trajectories, without post-hoc selection of a best checkpoint.

This change replaces the earlier proposed `200000`-timestep, four-checkpoint design. It does not add experimental conditions. Relative to the earlier five-condition estimate, it increases planned training and checkpoint evaluation by approximately 50 per cent. Local pilot throughput predicts approximately 46.8 minutes for all five eventual conditions, excluding interpretation, plotting and representative video selection.

The formal experiment remains phased:

1. run the locked reference and three reduced-control-cost conditions;
2. confirm the divergent reduced-control-cost setting using proxy return and the predeclared disaggregated diagnostics;
3. lock the shaping formula and coefficient in a separate versioned config before training the shaping condition;
4. after the main-seed scan, add seeds `20260809` and `20260810` as a clearly labelled robustness extension for the reference, selected divergent and shaped conditions only.

The extra seeds are replications, not additional control groups. The full coefficient sweep remains a single-main-seed exploratory study; the three-condition extension supports a more cautious assessment of whether the key reference-divergence-mitigation pattern repeats.

Implementation changes associated with this lock must preserve base proxy reward, shaping reward and observed proxy return as separately logged quantities. Pilot outputs remain non-formal and must not be merged with formal CSV files.

## 22. Formal coefficient-sweep completion and shaping pilot decision on 8 August 2026

The locked main-seed coefficient sweep completed for the reference, `0.25`, `0.125` and `0.0625` conditions. Its canonical output directory is:

`D:\ProxyGap\proxygap_ant\artifacts\formal\formal_v1_coefficients_20260808`

Data-quality evidence is saved in `logs/data_quality_report.json`. The validation passed all declared checks: 240 evaluation rows, 24 runtime rows, 24 model checkpoints, no duplicate episode keys, paired evaluation seeds, complete required metrics, bounded PPO rollout overshoot and exact observed/base/shaping reward reconciliation within floating-point tolerance.

At the predeclared 300000-timestep endpoint, the `0.0625` condition is the clearest main-seed proxy-performance divergence candidate. Its mean proxy return was approximately `737.59`, compared with `584.23` for the reference, while mean net forward progress was approximately `-0.75`, compared with `6.26` for the reference. Its mean survival-reward sum was `1000`, indicating that episode survival can sustain a comparatively high proxy return despite backward net motion. This is formal single-main-seed evidence, not yet a multi-seed final conclusion.

The shaping pilot is therefore locked to `ctrl_cost_weight=0.0625` and forward-progress shaping rather than lateral-drift shaping. The versioned pilot config is:

`D:\ProxyGap\proxygap_ant\configs\shaping_pilot_v1_20260808.json`

The pilot formula is:

`observed_reward = base_proxy_reward + forward_progress_shaping_weight * reward_forward`

The paired pilot candidates are `0.0`, `0.5`, `1.0` and `2.0`, trained for 100000 timesteps with checkpoints at 50000 and 100000, ten evaluation episodes, training seed `20260807`, evaluation seed base `40260808`, and the locked full-PPO settings. The smallest coefficient producing positive mean net forward progress at 100000 timesteps with fall rate no greater than 0.2 should be preferred. These pilot outputs remain parameter-selection evidence and must not be reported as formal results.

## 23. Shaping and robustness-extension parameters locked on 8 August 2026

The forward-progress shaping pilot completed with 80 paired evaluation episodes. At 100000 timesteps:

- the paired unshaped condition had mean net forward progress of approximately `-0.73` and fall rate `0.0`;
- shaping weight `0.5` had mean net forward progress of approximately `0.46` but fall rate `0.3`, so it failed the predeclared fall-rate guardrail;
- shaping weight `1.0` had mean net forward progress of approximately `6.40` and fall rate `0.0`;
- shaping weight `2.0` had mean net forward progress of approximately `1.48` and fall rate `0.0`.

Under the predeclared rule, `forward_progress_shaping_weight=1.0` is locked because it is the smallest tested coefficient satisfying both positive forward progress and fall rate no greater than `0.2`. This remains pilot-based parameter-selection evidence, not a formal result.

The formal shaping config is:

`D:\ProxyGap\proxygap_ant\configs\formal_v1_shaped_20260808.json`

It uses `ctrl_cost_weight=0.0625`, `forward_progress_shaping_weight=1.0`, primary seed `20260808`, the same paired evaluation seeds as the coefficient sweep, 300000 timesteps, six fixed checkpoints and ten evaluation episodes per checkpoint.

The approved robustness-extension config is:

`D:\ProxyGap\proxygap_ant\configs\formal_v1_core_replication_20260808.json`

It adds training seeds `20260809` and `20260810` for exactly three conditions: reference, unshaped `ctrl_cost_weight=0.0625`, and shaped `ctrl_cost_weight=0.0625` with forward shaping weight `1.0`. These are replications rather than additional experimental groups.

## 24. Formal v1 completion and validated descriptive results on 9 August 2026

All approved formal training, evaluation and core replications are complete. No smoke-test or pilot rows are included in the formal combined dataset.

Canonical formal output directories:

- coefficient sweep: `D:\ProxyGap\proxygap_ant\artifacts\formal\formal_v1_coefficients_20260808`;
- main-seed shaped condition: `D:\ProxyGap\proxygap_ant\artifacts\formal\formal_v1_shaped_20260808`;
- two-seed core replication: `D:\ProxyGap\proxygap_ant\artifacts\formal\formal_v1_core_replication_20260808`;
- combined tables, figures, note and videos: `D:\ProxyGap\proxygap_ant\artifacts\formal\combined_v1_20260809`.

Completion and data-quality evidence:

- 66 model checkpoints are present: 24 coefficient-sweep models, 6 main-seed shaped models and 36 replication models;
- 660 formal evaluation rows are present: 240, 60 and 360 rows respectively;
- the combined dataset has no duplicate `(condition_id, training_seed, target_timesteps, episode)` keys;
- every condition/seed has six declared checkpoints and every seed/checkpoint has ten evaluation episodes;
- core conditions use three training seeds, while `0.25` and `0.125` use the predeclared single main seed;
- all three formal validation reports passed reward decomposition, required-metric, checkpoint, model-count, seed-pairing and CSV-schema checks;
- combined observed/base/shaping reward reconciliation has maximum absolute error approximately `1.5e-11`;
- the automated test suite passes with `14 passed`;
- combined validation evidence is saved in `combined_v1_20260809\data\analysis_manifest.json`.

The 300000-timestep endpoint is summarised below. Values are descriptive means across training-seed means, with each training-seed mean based on ten paired evaluation episodes. Standard deviations are shown only where three training seeds are available.

| Condition | Training seeds | Proxy return | Net forward progress | Fall rate | Final lateral drift | Torso tilt SD |
|---|---:|---:|---:|---:|---:|---:|
| Reference (`0.5`) | 3 | `539.26 +/- 156.11` | `4.64 +/- 1.79` | `0.53 +/- 0.21` | `3.45 +/- 0.17` | `0.42 +/- 0.12` |
| Control cost `0.25` | 1 | `869.01` | `4.07` | `0.00` | `4.42` | `0.83` |
| Control cost `0.125` | 1 | `909.17` | `7.99` | `0.10` | `2.34` | `0.19` |
| Control cost `0.0625` | 3 | `787.52 +/- 45.11` | `0.62 +/- 1.51` | `0.00 +/- 0.00` | `0.92 +/- 0.99` | `0.08 +/- 0.08` |
| Shaped `0.0625` | 3 | `770.74 +/- 237.80` | `2.00 +/- 0.77` | `0.23 +/- 0.40` | `1.65 +/- 0.84` | `0.13 +/- 0.13` |

Locked descriptive interpretation:

- the unshaped `0.0625` condition is the clearest three-seed proxy-performance gap: mean proxy return exceeded the reference while mean net forward progress was substantially lower;
- forward-progress shaping increased mean net forward progress from approximately `0.62` to `2.00`, but did not restore the reference mean of approximately `4.64`; mitigation is therefore partial;
- shaping increased mean fall rate from `0.00` to approximately `0.23`, while the reference fall rate was approximately `0.53`; this is a separate safety trade-off rather than a single aggregate success or failure judgement;
- the `0.125` single-seed condition achieved both high proxy return and high forward progress, showing that reduced control cost does not uniformly create misspecification across every coefficient;
- seed variability is material, especially for proxy return and fall rate. These results support an exploratory simulation case study, not an inferentially strong or universal claim about PPO, locomotion or physical robots.

No aggregate hidden `true_performance` score has been introduced. Proxy return, net forward progress, control effort per unit distance, fall rate, lateral drift, torso orientation variability and videos remain separately reported diagnostics. Control effort is not described as direct energy consumption.

Six report-ready static figures are saved in `combined_v1_20260809\plots`. Three paired 300000-timestep trajectory GIFs use evaluation seed `30260808` and are saved in `combined_v1_20260809\videos`. Rendering validation confirmed non-blank 480-by-480 animations with 6, 55 and 21 frames and clear first-to-last pixel changes. The trajectory metric file is strict JSON; undefined effort per unit distance is represented as `null`.

One coefficient-sweep runtime row includes a Windows system-suspend interval and must not be used as an active-training-speed estimate. The affected model, actual timesteps and evaluation data passed all integrity checks. Sleep prevention and safe resume support were added before the remaining formal runs. Runtime comparisons should therefore use validated active runs or the earlier pilot estimate, not the contaminated wall-clock row.

The next workstream is report and presentation production in academic British English. It should use the combined tables and figures, state the single-seed status of the intermediate coefficients visibly, treat shaping as partial mitigation with a safety trade-off, and preserve the simulation-only limitations above.

## 25. Rigorous academic audit and protocol correction on 10 August 2026

This section records a retrospective audit. It does not change the formal v1 data, convert exploratory work into a preregistered study or authorise a new long experiment.

Canonical audit outputs:

- audit report: `D:\ProxyGap\proxygap_ant\artifacts\audit\rigorous_academic_work_20260810\RIGOROUS_ACADEMIC_AUDIT_20260810.md`;
- execution-ready protocol: `D:\ProxyGap\proxygap_ant\artifacts\audit\rigorous_academic_work_20260810\EXECUTION_READY_RESEARCH_PROTOCOL_v1_1_20260810.md`;
- deviation register: `D:\ProxyGap\proxygap_ant\artifacts\audit\rigorous_academic_work_20260810\DEVIATION_REGISTER_20260810.csv`;
- hardware and environment record: `D:\ProxyGap\proxygap_ant\artifacts\audit\rigorous_academic_work_20260810\environment-and-hardware-20260810.json`;
- exact package lock and source-hash manifest: the audit directory's `environment` folder and `source_manifest_sha256.csv`;
- reproducible evidence tables: `paired_effects_300k.csv`, `checkpoint_gaps.csv`, `runtime_audit.csv` and `resource_scenarios.csv`.

The audit preserves the formal v1 tables and introduces the following interpretation corrections:

1. The default `ctrl_cost_weight=0.5` condition is a benchmark comparator, not a validated aligned reward or ground-truth objective. A separate post-hoc sanity test on ten new evaluation seeds found that an untrained PPO policy achieved mean proxy return approximately `992.06` with mean net progress approximately `-0.07`, whereas the trained reference achieved approximately `617.17` proxy return and `6.56` progress. Survival reward and episode length must therefore be reported as part of the proxy mechanism. These post-hoc rows are not formal data.
2. The implemented shaped condition is `base_proxy_reward + 1.0 * reward_forward`. It doubles the effective forward-reward contribution and does not implement Proposal_G6's proposed bounded, scale-normalised effort and instability penalties. Formal v1 reporting must label it `exploratory forward-reward reweighting` or disclose the deviation explicitly. It can support a partial-progress-recovery observation, but it cannot validate the proposal-conformant mitigation construct.
3. For `0.0625 - reference` at the final checkpoint, all three paired proxy-return differences are positive and all three paired net-progress differences are negative. Falls, lateral drift and torso-tilt variation improve rather than worsen on average. The defensible conclusion is therefore a proxy-diagnostic trade-off centred on progress and control efficiency, not uniformly worse locomotion.
4. Forward reweighting improves progress relative to unshaped `0.0625` in all three paired seeds, but one seed has a `+0.7` fall-rate difference together with worse drift and tilt. Mitigation without failure displacement is not established.
5. The checkpoint evidence supports late-emerging progress divergence but not monotonic amplification. The mean proxy gap peaks before the final checkpoint and narrows by 300k.
6. Three training seeds do not support a conventional significance claim. With three same-direction paired signs, the smallest attainable exact two-sided sign-test p-value is `0.25`.
7. Runtime inspection identifies two, not one, severe wall-clock outliers: `8,368.689` s and `29,647.345` s. The data and models in both rows pass integrity checks, but their wall-clock durations are excluded only from active-throughput estimation. Across the remaining 64 chunks, measured throughput is approximately `537.50` actual steps/s; formal evaluation averages approximately `0.7021` s per episode. System suspension is consistent with at least part of this contamination, but a specific cause is not proven for both rows.

Fresh audit smoke evidence:

- permanent suite: `14 passed`;
- reduced-setting PPO smoke: 1,024 steps in `1.045` s (`979.87` steps/s), used only as a pipeline check;
- dynamic protocol smoke: eight checks passed, covering seed replay, fixed-transition reward isolation, shaping decomposition, reward formula, termination, truncation and checkpoint replay;
- the initial low-z termination construction failed because contact dynamics returned the torso to the healthy interval; the failed record is preserved and the corrected high-z construction passed.

If an additional timed coefficient pilot is required, the locked engineering design is four coefficients (`0.5`, `0.25`, `0.125`, `0.0625`), two paired training seeds (`20260811`, `20260812`), 50,000 target steps, 25k and 50k checkpoint targets, and five paired evaluation seeds (`60260810`-`60260814`). The estimated active time is approximately `13.6` minutes and model storage approximately `4.7` MB. This pilot is for timing, failure and variance reconnaissance, not for formal coefficient inference.

No additional long training is required for an explicitly exploratory report based on formal v1. A new proposal-conformant mitigation claim requires a versioned bounded effort/stability formula, pilot-only scale and coefficient selection, fresh held-out seeds, frozen success and protected-metric rules, and a new deviation-controlled study. The resource-aware recommendation is five paired seeds for three core conditions (approximately `150.6` active CPU minutes and `26.5` MB of model checkpoints); ten seeds provide a stronger distributional picture at approximately `301.1` minutes and `52.9` MB. These are measured planning estimates, not guaranteed runtimes.

The final assessment rubric, word limit, presentation duration and AI-assistance disclosure policy remain unresolved. Material AI assistance must be declared if required by the applicable course rules, and all generated academic text and numerical claims require student verification before submission.

## 26. Independent method-critic revision gate on 10 August 2026

The independent review `D:\nn_lecture\INDEPENDENT_METHOD_CRITIC_REVIEW_20260810.md` was read in full as UTF-8 and adjudicated against the exact Proposal_G6 PDF, canonical source code, formal CSV data and fresh tests. The canonical adjudication is:

`D:\ProxyGap\proxygap_ant\artifacts\audit\rigorous_academic_work_20260810\METHOD_CRITIC_ADJUDICATION_20260810.md`

The earlier combined protocol `EXECUTION_READY_RESEARCH_PROTOCOL_v1_1_20260810.md` is retained as a historical audit record but is superseded for prospective execution. The research records are now separated:

- retrospective formal v1: `D:\ProxyGap\proxygap_ant\protocols\formal_v1_retrospective_analysis_20260810.md`;
- prospective v2 draft: `D:\ProxyGap\proxygap_ant\protocols\prospective_v2_protocol_draft_20260810.md`;
- v2 metric contract: `D:\ProxyGap\proxygap_ant\docs\METRIC_DEFINITIONS_V2_20260810.md`;
- machine-readable revision gate: `D:\ProxyGap\proxygap_ant\configs\prospective_v2_revision_gate_20260810.json`.

The principal accepted correction is that condition-specific returns from different `ctrl_cost_weight` values are not a common numerical proxy scale. Formal v1 now distinguishes the reward each condition actually optimised from a counterfactual common rescore using `ctrl_cost_weight=0.5`. The original 660-row formal CSV remains unchanged. Reproducible rescoring outputs are generated by `scripts\rescore_formal_v1.py`.

At the 300k target, the paired `0.0625 - reference` differences are:

- condition-specific objective return: `+153.37`, `+157.40`, `+434.03` across seeds `20260808-10`;
- common-rescored return at fixed `w=0.5`: `-1552.11`, `-1362.06`, `-996.96`;
- net forward progress: `-7.02`, `-2.70`, `-2.34`;
- cumulative squared action: `+3562.12`, `+3004.61`, `+2961.53`;
- unhealthy termination rate: `-0.60`, `-0.30`, `-0.70`.

The positive condition-objective difference is partly mechanical because the reward definitions differ. The common-rescore, progress and action evidence supports a stronger but still bounded conclusion: reduced-cost policies use substantially larger squared actions and perform worse under the fixed default-weight rescore, while surviving longer and terminating less often. This is a multi-objective proxy-diagnostic trade-off, not formal reward hacking or uniformly worse locomotion.

Prospective code now supports and tests:

- fixed-weight common rescoring and reward reconstruction;
- condition-specific objective terminology;
- low-z collapse, high-z excursion, non-finite and other unhealthy termination categories;
- cumulative and per-step squared action, action saturation, denominator validity, maximum drift, cumulative lateral path, tilt RMS/p95/max and progress per step;
- bounded effort and torso-orientation intervention terms, kept separate from historical forward reweighting;
- complete PPO parameters from a resolved schema-v2 configuration;
- SB3 Monitor training logs and compressed per-step evaluation records;
- deterministic median-policy/median-episode video selection.

Fresh revision-gate validation completed without a 50k pilot or formal training:

- revised permanent suite: `23 passed`;
- 256-step PPO smoke: passed with one model, one Monitor file, one compressed evaluation-step record and one evaluation row;
- retrospective rescoring: `660` episode rows, `66` policy/checkpoint rows and `3` final paired rows;
- maximum absolute reward reconstruction error: approximately `1.82e-5`, below the CSV tolerance `1e-4`.

The timed pilot remains prohibited. The prospective v2 protocol is not frozen until the following are approved and written into the versioned config:

1. `descriptive_only` versus `margin_based_mitigation` analysis;
2. the positive `effort_distance_min` value;
3. intervention scales and per-step reward caps;
4. combined-only versus effort/orientation/combined attribution scope;
5. if margin-based analysis is selected, the smallest meaningful progress improvement and every protected harm margin.

The recommended resource-aware route is descriptive-only, `effort_distance_min=0.10` with sensitivity at `0.05/0.10/0.25/0.50`, a combined intervention without component-attribution claims, effort/orientation scales `2.0` and `0.5 rad`, and per-component caps of `0.25 reward units/step`. These are recommendations awaiting user approval, not locked research facts.

The final assessment rubric, presentation duration and AI-assistance policy also remain unavailable. They do not block code-level scientific development but continue to block final submission-compliance certification.
