# Stage-One Development Analysis Amendment

**Date:** 2026-08-14<br>
**Scope:** development analysis only; no held-out training and no shaping<br>
**Status:** prospective configuration preserved; ambiguities resolved
conservatively without changing the candidate gate

## 1. Non-inferiority wording

The frozen configuration contains a `weak_evidence_rule` referring to proxy
non-inferiority, but it does not specify a numerical non-inferiority margin.
Consequently, the phrase *proxy performance is similar* is not executable and
will not be inferred after observing the data. The stage-one research question
may retain the conceptual wording "similar or higher", but the development
candidate screen uses only the stricter pre-run rule:

\[
\Delta R_w > 0
\]

in both development training seeds, together with at least one practically
harmed predeclared diagnostic domain in both seeds. No weak-evidence candidate
will be reported or used to authorise a held-out run.

## 2. Target and actual PPO timesteps

Stable-Baselines3 PPO updates in complete rollout batches. A target checkpoint
such as 300,000 may therefore contain a policy trained for slightly more than
300,000 environment timesteps. `target_timesteps` remains the schedule label;
`actual_model_timesteps` must be read from the saved model and recorded
separately. Analysis is blocked if the actual timestep count differs across
conditions for the same target checkpoint.

## 3. Interrupted-run provenance

ZIP-file SHA-256 values may differ because archive metadata differ. They are
not sufficient to establish that policies differ. The interrupted and restarted
50k/100k checkpoints are therefore compared by policy state-dict keys, tensor
hashes and maximum absolute tensor differences. Only complete restarted runs
enter the development analysis.

## 4. Consequences

These clarifications do not alter coefficients, seeds, training budgets,
diagnostic margins or the strong-candidate rule after outcome inspection. They
make previously implicit or non-executable record-keeping rules explicit. The
original configuration and its pre-run SHA-256 remain unchanged.
