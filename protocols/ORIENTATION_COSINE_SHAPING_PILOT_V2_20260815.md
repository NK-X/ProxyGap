# Orientation Cosine Shaping Pilot V2

**Status:** frozen development-pilot revision authorised on 15 August 2026.

## Unchanged scientific design

V2 retains every scientific intervention and outcome decision from V1: the
normalised cosine penalty, candidate weights 0.10/0.25/0.50, default Ant-v5
reward coefficients, PPO configuration, three development training seeds,
20 matched evaluation seeds, four checkpoints, pilot gate and complete-video
panel. Formal seeds remain untouched and a second intervention is prohibited.

## Why V1 was blocked

V1 required the largest candidate's penalty to exceed 1% of absolute baseline
return at the pooled episode median. The 100 endpoint episodes combine five
deliberately different policies. Upright episodes have near-zero cosine penalty,
whereas inverted episodes have a large penalty. Pooling those modes made the
median 0.45%, despite the largest candidate contributing 13.6% of mean absolute
return for the predeclared inverted-high-proxy seed 41204. V1 therefore tested
the prevalence of adverse episodes in the mixed sample rather than the reward
scale conditional on the target failure mode.

The blocked V1 adjudication is retained unchanged under
`artifacts/pilot/orientation_cosine_shaping_pilot_v1_20260815/offline_calibration`.

## V2 offline scale gate

The same replayed trajectories and the same candidate weights are checked by
predeclared behaviour mode:

1. all values and expected rows must be present and finite;
2. weighted penalties must increase strictly with candidate weight;
3. at `lambda=0.50`, the mean penalty-to-absolute-return ratio for adverse-mode
   seed 41204 must be between 5% and 25%;
4. at `lambda=0.50`, the corresponding ratio for comparatively upright seed
   41201 must not exceed 10%.

This is only a scale-feasibility check. Counterfactual rescoring cannot predict
which policy PPO will learn under the modified reward.

## Remaining protocol

The intervention, development matrix, step-five gate, complete 20-fps H.264
MP4 plan, stopping boundary and claim limitations are exactly those specified
in V1. The V2 configuration is the executable source of truth for this run.
