# Final PAIR0 flat forward-and-turn diagnostic

This is a read-only evaluation of one frozen checkpoint. Safety has a predeclared PASS/FAIL gate. Turn effectiveness is descriptive only because no directly transferable constant-yaw-rate gate existed before seeing these results.

The two low-speed conditions command 0.10 m/s and ±0.10 rad/s. They are positive-speed, out-of-training-envelope yaw-rate probes, not in-place rotation.

| Condition | Target yaw change (rad) | Actual yaw change (rad) | Ratio | Same-sign episodes | Yaw-rate RMSE | Path curvature | Final reference error (m) | Zero-foot fraction | Safety | Failed safety checks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| straight_055 | 0.000000 | -0.510788 | n/a | n/a | 0.256228 | -0.097737 | 10.908901 | 0.017333 | PASS | none |
| curve_left_010 | 1.650000 | -0.334884 | -0.202960 | 0/5 | 0.217879 | -0.075786 | 11.900035 | 0.013667 | PASS | none |
| curve_right_010 | -1.650000 | -1.847628 | 1.119775 | 5/5 | 0.403231 | -0.130093 | 4.954912 | 0.038667 | PASS | none |
| curve_left_020 | 3.300000 | 0.772112 | 0.233973 | 4/5 | 0.349919 | 0.102081 | 9.918362 | 0.025333 | PASS | none |
| curve_right_020 | -3.300000 | -2.884828 | 0.874190 | 5/5 | 0.455469 | -0.209387 | 7.149118 | 0.047000 | PASS | none |
| curve_left_035 | 5.775000 | -0.089704 | -0.015533 | 3/5 | 0.399934 | -0.008727 | 4.595166 | 0.023000 | PASS | none |
| curve_right_035 | -5.775000 | -1.900307 | 0.329057 | 4/5 | 0.435865 | -0.171585 | 5.553178 | 0.034000 | PASS | none |
| low_speed_yaw_left | 3.000000 | 0.927997 | 0.309332 | 5/5 | 0.238277 | 0.269569 | 1.853482 | 0.011333 | PASS | none |
| low_speed_yaw_right | -3.000000 | -2.124706 | 0.708235 | 5/5 | 0.514634 | -0.201855 | 6.578068 | 0.024667 | PASS | none |

No row contains a turn-effectiveness PASS/FAIL decision. Energy is measurement-only and does not enter safety or tracking decisions. These observations alone cannot certify fixed-map readiness or random-map generalisation.
