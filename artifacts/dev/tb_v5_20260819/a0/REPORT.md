# PAIR0 turn-balance final result

This is the final locomotion-optimisation round. Both branches were trained from the same frozen source and only final checkpoints were evaluated.

| Branch | Turn gate | Standard-slope gate | Combined |
|---|---|---|---|
| C0_STRAIGHT_CONTINUE | FAIL | PASS | FAIL |
| C1_BALANCED_TURN | FAIL | PASS | FAIL |

Decision: `both_fail_turning_HOLD_retain_source_PAIR0`.

Energy remained measurement-only and did not enter reward, checkpoint selection or gates. No fixed-map evaluation, video rendering or promotion occurred in this run.

Optimisation is now hard-stopped regardless of PASS, FAIL or non-evaluable completion. A separate read-only video archive is frozen to seed 96131 and the left/right 0.20 per metre conditions; it cannot be reselected after seeing results.
