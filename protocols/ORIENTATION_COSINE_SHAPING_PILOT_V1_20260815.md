# Orientation Cosine Shaping Pilot V1

**Status:** frozen development pilot authorised on 15 August 2026.

## 1. Decision addressed

The pilot asks whether adding one bounded torso-orientation penalty to the
otherwise unchanged Ant-v5 reward reduces the posture failure observed in the
development baseline without materially damaging forward competence or creating
new lateral, action-quality or control-effort problems.

It does not constitute a held-out confirmation of mitigation, establish an
optimal reward, or test real-robot safety.

## 2. Intervention

The default Ant-v5 coefficients remain unchanged:

\[
w_{\mathrm{forward}}=1,\quad
w_{\mathrm{healthy}}=1,\quad
w_{\mathrm{control}}=0.5,\quad
w_{\mathrm{contact}}=5\times10^{-4}.
\]

The only intervention is

\[
r_t^{\mathrm{pilot}}=r_t^{\mathrm{Ant}}
-\lambda_\theta\phi_{\cos}(\theta_t),
\qquad
\phi_{\cos}(\theta)=\frac{1-\cos\theta}{2}.
\]

The function is zero when upright, 0.5 at 90 degrees and one when inverted. It
is bounded and does not add another positive survival-like reward floor.

## 3. Development matrix

- Orientation weights: 0.10, 0.25 and 0.50.
- Training seeds: 41201, 41204 and 41205.
- Nominal training budget: 1,000,000 timesteps per policy.
- Checkpoints: 250k, 500k, 750k and 1M.
- Evaluation: 20 deterministic episodes per checkpoint using seeds 51201-51220.
- Primary unit for cross-training claims: independently trained policy.
- Device: CPU only; no observation or reward normalisation.

The seeds are intentionally development-selected to cover three known baseline
behaviour modes: comparatively upright locomotion, high-proxy inverted
locomotion, and fast locomotion with frequent unhealthy termination. This makes
the pilot diagnostic rather than representative. Formal seeds 42001-42008
remain untouched.

## 4. Offline scale check

Before training, the normalised cosine penalty is recomputed from all available
final-checkpoint baseline traces. For each candidate weight, the cumulative
penalty and its ratio to the absolute base return are recorded. Training may
start only if all records are finite, all expected traces are present and the
candidate ordering is monotonic.

## 5. Pilot adjudication

At the 1M endpoint, each shaped policy is paired with the existing baseline
policy having the same training seed and is evaluated on the same evaluation
seeds.

A weight is labelled `promising_development_candidate` only when all of the
following hold:

1. inverted-step fraction decreases by at least 0.05 in at least two of the
   three training-seed pairs;
2. the median fixed-horizon forward-velocity retention ratio is at least 0.90;
3. unhealthy termination does not increase by more than 0.10 in more than one
   training-seed pair;
4. no two guardrail domains show directionally worse seed-level medians greater
   than 20 percent relative to baseline.

The guardrail domains are lateral drift, normalised action roughness, action
saturation and control effort per unit forward distance. The gate is an
operational development rule, not a statistical significance test. All raw
seed-level values and paired episode differences must be reported.

## 6. Video evidence

Videos are complete deterministic episodes encoded as H.264 MP4 at 20 fps,
which matches the Ant-v5 0.05-second environment step. A non-terminated episode
therefore lasts approximately 50 seconds. Early termination is retained rather
than padded or hidden.

The fixed matched panel uses evaluation seed 51216. It includes every checkpoint
for training seed 41204 under the baseline and all three shaping weights, plus
the final checkpoint for training seeds 41201 and 41205. Numerical metrics, not
video appearance, determine the pilot adjudication. Behaviour labels in the
video index use operational terms such as sustained inversion, high-z
termination and full-horizon upright locomotion; `normal` is not assigned as an
unmeasured subjective label.

## 7. Stopping boundary

This execution stops after implementation validation, offline scale checking,
the three-by-three development matrix, complete-video generation and the
step-five adjudication. It must not start lateral shaping, alter PPO, choose a
formal reward, or use held-out formal seeds.
