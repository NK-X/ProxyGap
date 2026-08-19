# Final PAIR0 standard-slope capability boundary

This is a read-only, deterministic evaluation of one frozen trained policy. It does not estimate a physical maximum slope or random-map generalisation.

| Scene | Mean best progress (m) | Zero-foot fraction | Force-qualified denominator | Corrected sustained slip substeps | Slip events | Falls | Torso | Sustained non-foot | Decision | Failed checks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| flat | 8.387811 | 0.024333 | 12206 | 0 | 0 | 0 | 0 | 0 | PASS | none |
| uphill_4deg | 10.050550 | 0.038000 | 11522 | 0 | 0 | 0 | 0 | 0 | PASS | none |
| uphill_8deg | 8.097968 | 0.028667 | 11325 | 0 | 0 | 0 | 0 | 0 | PASS | none |
| uphill_12deg | 7.334866 | 0.048667 | 10698 | 0 | 0 | 0 | 0 | 0 | PASS | none |
| uphill_16deg | 5.531944 | 0.049000 | 10435 | 0 | 0 | 0 | 0 | 0 | FAIL | effective_progress |
| uphill_20deg | 2.238192 | 0.068333 | 9742 | 0 | 0 | 0 | 0 | 0 | FAIL | zero_foot_within_gate, effective_progress |
| downhill_4deg | 8.770105 | 0.021000 | 12178 | 0 | 0 | 0 | 0 | 0 | FAIL | effective_progress |
| downhill_8deg | 8.944102 | 0.022667 | 11989 | 0 | 0 | 0 | 0 | 0 | PASS | none |
| downhill_12deg | 8.230011 | 0.022000 | 12068 | 0 | 0 | 0 | 0 | 0 | FAIL | effective_progress |
| downhill_16deg | 9.025416 | 0.027333 | 11482 | 0 | 0 | 0 | 0 | 0 | PASS | none |
| downhill_20deg | 8.013466 | 0.018667 | 11615 | 0 | 0 | 0 | 0 | 0 | FAIL | effective_progress |

## Tested brackets

- uphill: conservative tested lower bound = 12; first failing tested angle = 16; raw passing angles = [4, 8, 12].
- downhill: conservative tested lower bound = None; first failing tested angle = 4; raw passing angles = [8, 16].

All five predeclared held-out seeds are retained per scene. Flat has no added progress threshold and is a safety reference only. Energy quantities are measurements only and do not enter any gate.
