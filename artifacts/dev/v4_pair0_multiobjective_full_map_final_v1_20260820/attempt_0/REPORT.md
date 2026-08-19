# V4 + PAIR0 multi-objective full-map final evaluation

All objectives passed: **True**. Two unique route contracts were evaluated over three fresh seeds (six episodes).

| Objective | Weights (time, energy) | Contract | Success | Mean time (s) | Mean positive work (J, proxy) | Slip events | Falls |
|---|---|---|---:|---:|---:|---:|---:|
| time_priority | [0.8, 0.2] | time_and_balanced | 3/3 | 264.550 | 55651.431 | 0 | 0 |
| balanced | [0.5, 0.5] | time_and_balanced | 3/3 | 264.550 | 55651.431 | 0 | 0 |
| energy_priority | [0.2, 0.8] | energy_priority | 3/3 | 259.367 | 55134.172 | 0 | 0 |

Energy is a mechanical-work proxy, not electrical battery energy. Results concern one frozen, previously inspected map.
