# ProxyGap final presentation — 15-minute speaker script

Target delivery: 14 minutes 10 seconds, leaving approximately 50 seconds of contingency before Q&A.

Replace every speaker placeholder with the member's name in Pinyin. The wording is intentionally evidence-bounded; do not replace terms such as “known map”, “proxy” or “candidate bank” with stronger claims.

## Slide 1 — A two-stage PPO–Ant study progressed from flat walking to verified map completion

**Speaker:** [Speaker A Pinyin]

**Target time:** 0:40

The short answer to our main question is yes: the final system reached the goal. Across six formal episodes, it completed the same frozen terrain mission six times, with no falls and no duration-corrected sustained-slip events. The three videos represent time-priority, balanced and energy-priority preferences. However, this is a known-map result, not unseen-map generalisation, and “optimal” means preferred within a bank of fifteen evaluated, feasible candidates. Our project is therefore about both capability and evidence boundaries.

## Slide 2 — V2 solved locomotion diagnostics; V3 added terrain, safety and planning

**Speaker:** [Speaker A Pinyin]

**Target time:** 1:10

We organise the work into two research stages. The legacy V1 trials mainly revealed that a generic forward reward did not express our real intention. Stage 1, which we call Project V2, asked whether reward and constraint interventions could produce directed and more coordinated flat-ground locomotion. Stage 2, or Project V3, raised the complexity substantially: a continuous heightfield, local terrain observations, contact auditing, slopes, turning, route planning and human time–energy preferences. These presentation labels describe research complexity; they are not the suffixes of every checkpoint file in the repository. The central research question is how a locomotor progresses from movement to safe mission completion.

## Slide 3 — The baseline is locally reproduced Ant-v5 plus PPO

**Speaker:** [Speaker A Pinyin]

**Target time:** 1:40

The baseline needs three precise parts. First, Farama's Ant-v5 page defines the current environment and eight torque actions, but it is documentation, not a research paper. Second, Schulman and colleagues' PPO paper provides the algorithmic source. Third, our numerical comparator is a locally reproduced Ant-v5 plus PPO configuration with matched environment, budget, seed rules and evaluation protocol. This matters because published robots, observations, simulator versions and training budgets are not directly interchangeable. Fu and colleagues provide a methodological anchor for gait and mechanical-energy measurement, while Lee, Miki and Aractingi provide terrain and quadruped-control context. We do not subtract their published scores from ours. In Stage 2, the fair comparator is the Stage 1 controller lineage and the direct-goal baseline—not an unmodified Ant that has neither map nor waypoint information.

## Slide 4 — Reward was a training signal; task validity and safety were judged independently

**Speaker:** [Speaker B Pinyin]

**Target time:** 1:05

This slide shows our main methodological principle. PPO optimised a reward containing speed, direction, posture, action-rate, landing and support terms. We then measured independent diagnostics such as path efficiency, heading error, tilt, contact sequence and mechanical proxies. Finally, a separate mission gate defined success: enter within 1.5 metres of the goal, remain within 2 metres for two seconds, and avoid falls, torso-ground contact, sustained non-foot contact and sustained corrected-slip events. Time and energy were compared only after validity and safety passed. This separation is important because most of our shaping terms are not guaranteed to preserve the original optimal policy.

## Slide 5 — Stage 1 reshaped locomotion without changing the robot action space

**Speaker:** [Speaker B Pinyin]

**Target time:** 1:05

Stage 1 retained the Ant body and all eight torque actions. We added bounded target-speed and direction tracking, torso vertical and angular terms, signed-pitch diagnostics, action-rate and saturation measurements, and foot-landing and support diagnostics. An external action limiter was recorded separately as a controller intervention. This distinction prevents a common overclaim: smoother output from a limiter is not automatically a learned gait mechanism. The objective was to improve directed motion and expose side effects, not simply to maximise one larger reward number.

## Slide 6 — Selected diagnostics improved, but a natural gait was not scientifically verified

**Speaker:** [Speaker B Pinyin]

**Target time:** 1:15

Two development comparisons illustrate the trade-offs. In one paired setting, action roughness fell from 0.0139 to 0.00985, mean speed increased from 0.844 to 0.918 metres per second, and path efficiency rose from 0.809 to 0.857. In another, mean take-offs fell from 21.1 to 3.77 and the no-floor fraction decreased, although mean speed also fell slightly. These episodes support the statement that selected coordination and stability diagnostics improved. They do not establish a biologically natural gait, because duty factor, phase consistency and contact order were not frozen across speeds as primary validation measures.

## Slide 7 — Support, turning and global route choice were different failure layers

**Speaker:** [Speaker C Pinyin]

**Target time:** 1:10

Stage 2 revealed three separate problems. At the low level, the policy had 122 locomotion variables plus thirteen local terrain-preview variables. PAIR0 improved distal support on the heightfield. At the middle level, balanced left and right commands preserved slope safety but failed the signed turning gate. At the high level, the direct-goal PAIR0 policy had no complete map: after 600 seconds it achieved only 14.51 metres of best progress and never arrived. Improving contact support therefore did not automatically solve steering or route selection.

## Slide 8 — PAIR0 improved support and established tested slope boundaries

**Speaker:** [Speaker C Pinyin]

**Target time:** 1:10

PAIR0 introduced four explicit floor-to-distal-foot contact pairs while retaining the frozen robot geometry. In the held-out diagnostic, complete zero-foot control intervals fell from approximately 24.04 to 3.21 per cent, and mean supporting feet increased from 0.353 to 1.344. On the tested slope grid, uphill episodes passed contiguously through 12 degrees, while 16 degrees was the first tested failure. Downhill results were non-monotonic. Across all 55 slope episodes there were no falls, torso-ground contacts, sustained non-foot contacts or corrected sustained-slip events. We call 12 degrees a tested lower capability boundary, not a physical maximum, and we do not claim that PAIR0 “fixed MuJoCo”.

## Slide 9 — The successful solution was hierarchical, not another reward term

**Speaker:** [Speaker C Pinyin]

**Target time:** 1:15

A read-only checkpoint screen found that an archived canonical-frame V4 expert was the only candidate with both left and right response. V4 plus PAIR0 passed flat, plus and minus eight degrees, and plus twelve degrees safety screens. A naïve action blend between V4 and the final PAIR0 policy fell after 15.3 seconds, so it was rejected. The final architecture kept each responsibility explicit. The known-map planner read the frozen 1025 by 1025 terrain, selected a feasible route, and supplied a local waypoint with a three-metre lookahead. The frozen low-level expert then produced eight torques from local sensing. The PPO policy never received the full map.

## Slide 10 — Three preferences selected two routes from fifteen feasible candidates

**Speaker:** [Speaker D Pinyin]

**Target time:** 1:10

All fifteen plotted candidates first passed arrival and safety gates. We then ranked them with a declared normalised objective: the time term was completion time divided by the minimum observed time, and the energy term was positive mechanical-work proxy divided by its minimum. Time priority used weights 0.8 and 0.2; balanced used 0.5 and 0.5; energy priority used 0.2 and 0.8. Time and balanced selected the same contract, while energy selected a second. The correct claim is near-optimal within this evaluated bank. We did not evaluate a continuous Pareto surface, and the work proxy is not battery energy.

## Slide 11 — Six out of six formal episodes reached the goal

**Speaker:** [Speaker D Pinyin]

**Target time:** 1:50, including 20–25 seconds of video

The time and balanced route completed three out of three formal episodes, with a mean of 264.55 seconds, 55.65 kilojoules of positive-work proxy and 153.23 metres of actual path. The energy route also completed three out of three, with means of 259.37 seconds, 55.13 kilojoules and 152.14 metres. The energy route was also slightly faster in this formal mean, which demonstrates reset-seed variability rather than a universal dominance result. Every video was reproduced from its formal trace with zero state and five-substep contact mismatches, and every frame decoded successfully. The safety result is zero sustained corrected-slip events, not zero transient slip samples. Complete airborne control intervals still occupied about 9.25 to 10.02 per cent of the representative runs.

**Video cue:** Play approximately seven seconds each from the time-priority, balanced and energy-priority MP4 files. If playback fails, use the three final-frame images already present in the deck and state the exact replay result.

## Slide 12 — Method provenance and comparator literature

**Speaker:** [Speaker D Pinyin]

**Target time:** 0:20

These are the primary method and comparator sources. Full Harvard entries and the project evidence appendix are in the report. We will not read the list aloud; its purpose is to make the provenance visible and auditable.

## Slide 13 — The project established a reproducible known-map system

**Speaker:** [Speaker D Pinyin]

**Target time:** 0:30

The project establishes selected Stage 1 improvements, PAIR0 support under a frozen contact model, tested slope boundaries and six known-map completions. It does not establish a biologically natural gait, unseen-map generalisation, independent training-seed robustness, electrical energy or a global optimum. The key result is not that one reward solved everything. A sequence of auditable failures revealed the contact, steering and planning layers required for reliable completion. Thank you; we welcome your questions.

## Video files for offline playback

1. `artifacts/dev/v4_pair0_multiobjective_full_map_video_v1_20260820/time_priority/v4_pair0_time_priority_seed_690223864_full_map_relief_v1.mp4`
2. `artifacts/dev/v4_pair0_multiobjective_full_map_video_v1_20260820/balanced/v4_pair0_balanced_seed_1864999454_full_map_relief_v1.mp4`
3. `artifacts/dev/v4_pair0_multiobjective_full_map_video_v1_20260820/energy_priority/v4_pair0_energy_priority_seed_952993985_full_map_relief_v1.mp4`

## Final rehearsal checks

- Replace all speaker and group placeholders.
- Rehearse at least once with the exact offline MP4 files.
- Keep the main talk below 14:10 and reserve the remaining time for playback or handover delays.
- Ensure each external claim retains its same-slide source and complete reference entry.
- Do not say “natural gait achieved”, “globally optimal”, “no slip”, “battery energy”, or “generalises to unseen maps”.
