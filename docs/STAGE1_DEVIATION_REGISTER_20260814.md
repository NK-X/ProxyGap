# Stage-One Deviation Register

**Source specification:** `Proposal_G6.pdf`, dated 2026-08-07<br>
**Actual workspace:** isolated revision directory dated 2026-08-13/14<br>
**Scope:** development exploration only

| Item | Prescribed or proposed design | Actual stage-one design | Rationale | Impact and required action |
|---|---|---|---|---|
| Coefficient range | A small range centred on the `0.5` reference | One-sided `0.125` to `0.5` range | Tests the specific mechanism of under-penalised control effort | Cannot be described as centred or bidirectional. Obtain group/supervisor acceptance before formal freeze. |
| Conditions | Reference, three reduced penalties and one shaping condition | Reference plus six reduced coefficients in development; no shaping | Dense localisation of a possible divergence interval | More development cost; stage-two mitigation deliverable remains outstanding. |
| Checkpoints | 25%, 50%, 75% and 100% | 50k, 100k, 150k, 200k, 250k and 300k | Better late-window persistence check | Checkpoints remain dependent repeated measures, not replications. |
| Seeds | One fixed primary seed; additional seeds optional | Two development seeds; five held-out seeds reserved | Reduce single-run selection risk | Higher compute cost; five seeds still support descriptive replication more readily than conventional null-hypothesis inference. |
| Threshold claim | Identify any divergence threshold | Identify a discrete onset interval only | Seven tested coefficients do not identify a continuous change point | Prohibit claims of a unique critical coefficient, discontinuity or phase transition. |
| Proxy similarity | Similar or higher proxy performance | Strong screen requires strictly positive paired proxy advantage in both development seeds | No non-inferiority margin was frozen | Do not claim proxy equivalence or non-inferiority from this development run. |
| Timesteps | Nominal checkpoint budgets | PPO may overshoot each target to complete a 2,048-step rollout | Algorithmic batching | Record both target and actual timesteps; block comparison if actual counts differ by condition. |
| Formal condition selection | Reference, upper non-candidate, first candidate and next lower severity point | Development results show candidate entry, exit and re-entry | The earlier four-condition rule cannot preserve the non-monotonic pattern | Proposed six-condition matrix requires approval before formal freeze. |
| Health construct | Environment health may be read as robot health | Gym health uses finite state and torso height only; complete videos include inverted but height-healthy poses | Prevent construct overclaim | Keep torso orientation and other diagnostics separate; never call the health flag comprehensive safety. |
| Formal seed claim | Five seeds with a 4/5 gate | One-sided fair-sign tail probability is 0.1875 | Five seeds provide descriptive replication, not conventional directional significance | Choose five descriptive seeds or eight seeds with a 7/8 directional gate before freeze. |

No entry in this register changes the historical content of `Proposal_G6.pdf`.
No formal held-out experiment or shaping experiment has been started under this
revision.

## Post-bidirectional update

| Item | Earlier stage-one position | Evidence after the upper-side run | Adjudication |
|---|---|---|---|
| Coefficient direction | One-sided reduced-weight development | Added `0.625` and `0.75` before inspecting their outcomes; neither qualified | Development grid is now bidirectional over tested points. The negative upper result is retained without extending the range. |
| Proxy gate | Strict positive proxy gain only | A 5% relative non-inferiority rule was frozen before the upper run, with 0%, 2.5% and 5% sensitivity | `0.21875` and `0.125` have strict positive gain, so their classification does not depend on the relaxed proxy margin. |
| Reference competence | Audit listed but no criterion or result | At 300k, reference unhealthy-termination rates are 0.9 and 0.7, with mean episode lengths 331.4 and 485.9 | Formal launch is blocked. A one-million-step, unchanged-configuration development extension is proposed before architecture tuning. |
| Candidate interpretation | `0.21875` was the nearest qualifying lower value | Its absolute lateral drift worsens in both seeds, but normalised lateral checks do not both worsen and the candidate runs longer | Retain the development nomination; do not yet claim low overall performance. Freeze corridor intent or use `0.125` as an additional construct candidate. |
| Training budget | 300k endpoint | PPO MuJoCo precedent and RL Zoo operate at a one-million-step scale; current reference remains weak | Treat 300k as development exploration, not a validated formal budget. |
| Video comparison | Complete real-time videos after numerical selection | Per-condition median selection had a floating-point tie issue and did not guarantee matched cases | Tie handling was fixed and tested. Select the case from the reference only, then reuse the same training and evaluation seeds across conditions. |
| Formal matrix | Six lower-side transition-preserving conditions proposed | Upper-side run is negative; the principal local candidate has a construct ambiguity; `0.125` shows a different diagnostic pattern | Supersede the executable intent of formal proposal v1 with the blocked V3 revision gate. Preserve v1 as historical material. |

## Post-budget-extension update

| Item | Frozen V4 position | Evidence after continuation to 1M | Adjudication |
|---|---|---|---|
| Continuation semantics | Resume six saved 301,056-step policies without changing architecture, optimiser, reward or normalisation | Policy/value and optimiser state were restored, but live MuJoCo state and the complete RNG stream were not recoverable | Describe this as reproducible policy continuation, not bitwise-equivalent uninterrupted training. A fresh uninterrupted reference-only test is the preferred next gate. |
| Reference competence | Each `w=0.5` seed must have unhealthy termination rate `<=0.20` and mean forward velocity `>=0.10` at 1M | Seeds 41101 and 41102 had rates `0.90` and `0.60`; both passed the velocity threshold | The frozen joint gate failed. Do not weaken it retrospectively; formal launch remains blocked. |
| Primary candidate mechanism | `w=0.21875` nominated at 300k mainly through absolute lateral drift | At 1M it has positive matched proxy advantage in both seeds together with lower net progress/path efficiency and higher torso-tilt RMS | Candidate status is strengthened and no longer rests only on absolute drift, but it remains development evidence. |
| Overall performance language | No scalar true performance was defined | The candidate is worse on locomotion effectiveness and posture but better on unhealthy termination | Report a proxy-diagnostic, multi-objective trade-off; prohibit “uniformly worse overall performance”. |
| Optimisation pressure | Check 500k, 750k and 1M for persistence | The qualifying pattern disappears and reappears across checkpoints | Prohibit continuous, stable-throughout and monotonic-amplification claims. |
| `w=0.125` role | Retained as an alternative construct candidate | Matched proxy return is lower than reference in both seeds at 1M | Retain as a negative construct check; do not carry it forward as the primary high-proxy candidate without a new rationale. |
| Accuracy matrix | Teaching assistant reportedly expects percentages/error rates | No ground-truth/predicted classes exist in the frozen continuous-control protocol | Seek exact clarification; do not invent a confusion matrix or recast diagnostics as classification post hoc. |
| Stage-two shaping | Explicitly outside the V4 gate | No shaping weights were non-zero and no shaping run was started | Continue to prohibit shaping until stage-one logic is frozen. |

## Fresh-reference V6 pre-run update

| Item | V5 evidence or earlier plan | Frozen V6 action | Impact and boundary |
|---|---|---|---|
| Reference initialisation | Two reference policies were continued from 301,056-step checkpoints; exact MuJoCo state and complete RNG stream were not restored | Train five new `w=0.5` policies uninterrupted from initialisation to 1M | Separates fresh-baseline behaviour from the continuation limitation; does not erase or rerun V4 |
| Replication count | Two inspected development training seeds | Five new development training seeds, `41201`-`41205` | Improves descriptive assessment of run-to-run variation; remains development evidence rather than conventional significance testing |
| Evaluation precision | Ten episodes per policy/checkpoint | Twenty paired evaluation seeds, `51201`-`51220` | Health-rate resolution becomes `0.05`; episodes remain nested observations and are not counted as 100 independent policies |
| Configuration decision | Both earlier policies failed; no five-policy rule existed | `4-5/5` supported, `2-3/5` inconclusive, `0-1/5` failed | Rule frozen prospectively as an operational gate, not a literature-derived success threshold |
| Candidate and shaping scope | `0.21875` retained as a development candidate; shaping blocked | Reference only; no candidate or shaping policy is trained | Resolves reference competence before testing the stage-one hypothesis; stage two remains outside scope |
| Formal seeds | Five or eight held-out seeds had been discussed | Reserve `42001`-`42008`; do not use them in V6 | Prevents this diagnostic from consuming a future held-out confirmation set |
| PPO configuration | Standard 2-by-64 `Tanh` PPO was considered a reasonable, unoptimised baseline | Keep architecture, optimiser, reward and disabled normalisation unchanged | V6 diagnoses the current baseline; any later optimisation requires a separate frozen pilot |

V6 was authorised by the user on 14 August 2026. Its protocol and
machine-readable configuration were frozen before the five-policy outcomes were
generated. No result is recorded in this section until execution and independent
verification are complete.

### V6 execution-attempt note

The first V6 execution attempt was deliberately stopped before its first 250k
checkpoint after early throughput showed that the initially assigned two-hour
outer command timeout could terminate the queued fifth policy late in the run.
No model checkpoint or evaluation row had been produced and no scientific
outcome was inspected. All partial monitor logs are preserved under the
versioned `attempt1_interrupted` directory with hashes and are excluded from
analysis. The identical frozen scientific configuration is restarted from
initialisation; only the outer engineering timeout changes from two to six
hours.

The second V6 attempt was terminated when the Codex long-running tool host
became stale after approximately two hours, despite the nested command having a
six-hour timeout. Four policies had reached approximately 666k-674k episode
steps and eight model checkpoints existed, but no policy had completed and no
runtime or evaluation CSV had been written. No checkpoint outcome was inspected.
The full directory is preserved as `attempt2_interrupted_host_timeout` and is
excluded from analysis. Attempt 3 uses the identical frozen scientific config
from fresh initialisation; only process hosting changes to a detached hidden
Windows process so that tool-host rotation cannot terminate training.

## Fresh-reference V7 post-run adjudication

| Item | Frozen V6 position | Observed evidence | Adjudication |
|---|---|---|---|
| Completion | Five fresh `w=0.5` policies to nominal 1M, four checkpoints and 20 evaluations per checkpoint | Attempt 3 completed 20 models and 400 evaluation rows; every final model records 1,001,472 actual timesteps | Complete engineering execution under the frozen scientific configuration |
| Forward competence | Mean forward velocity must be at least 0.10 for each policy | All five policies passed; means range from 0.806 to 1.141 position units/s | Forward locomotion alone is not the blocking component |
| Health competence | Unhealthy termination rate must be at most 0.20 for the same policy | Rates were 0.00, 0.60, 0.60, 0.00 and 0.65 | Seeds 41202, 41203 and 41205 fail the frozen health component; thresholds are unchanged and no seed is replaced |
| Configuration rule | `4-5/5` supported, `2-3/5` inconclusive, `0-1/5` failed | Two of five policies pass both criteria | Reference configuration is `inconclusive`; do not reinterpret 20 evaluation episodes as 100 independent policies |
| Stage-one hypothesis | Reference competence must be resolved before candidate confirmation | V6 trained the reference only and did not test another coefficient | Reward misspecification is neither confirmed nor refuted by V6 |
| Next gate | If inconclusive or failed, freeze a separate reference-configuration pilot | Substantial seed-dependent health variation remains under one fixed configuration | A minimal one-factor pilot is required; do not jointly tune architecture, normalisation and optimiser |
| Formal and shaping scope | Candidate, formal and shaping launches prohibited | Reserved formal seeds 42001-42008 were not used; all shaping terms remained zero | Prohibitions remain in force |

Independent recomputation reproduced the five policy decisions and the 2/5
classification from the raw CSV. All 20 model hashes and timestep counts were
verified. The authoritative post-run files are
`configs/stage1_reference_fresh_1m_outcome_v7_20260814.json` and
`protocols/STAGE1_REFERENCE_FRESH_1M_ADJUDICATION_V7_20260814.md`.

## V8 reference-construct diagnostic

| Item | V7 interpretation | V8 evidence | Revised decision |
|---|---|---|---|
| Three gate failures | Possible seed-sensitive reference training | All 37 failures were rapid high-z excursions; none were low-z or non-finite | Retain as genuine simulator events, not ordinary falls or harmless static height offsets |
| Two gate passes | Passed the frozen velocity-and-Gym-health criteria | Seed 41204 spent 44.795% of recorded steps at tilt >=90 degrees and 42.780% below height 0.3; 9/20 episodes were majority-inverted | V7 arithmetic remains correct, but the gate is construct-insufficient and cannot certify human-intended competence |
| Gym health | Finite state and torso height `[0.2,1.0]` | An inverted, low-posture policy can remain inside the interval and collect healthy reward | Retain as a simulator termination rule; do not call it comprehensive robot health |
| Primary cause | PPO reference configuration may be unstable across seeds | The proxy has no direct orientation or lateral term, and distinct seeds learn different high-reward gait strategies | Supported mechanism is proxy/health omission plus stochastic optimisation; the contribution of normalisation remains unresolved |
| Next gate | Run a one-factor reference-configuration pilot | The outcome construct itself is not yet adequate | Freeze human-intent and baseline-suitability definitions before any normalisation pilot |

V8 did not retrain a policy. It replayed the five saved final models on the
original 100 endpoint episodes, recorded every step, reproduced the original
summaries exactly, and rendered five matched full trajectories. The 90-degree
inversion and height-below-0.3 quantities are transparent post-hoc diagnostics,
not formal thresholds. Candidate-weight, formal and shaping runs remain
prohibited.
