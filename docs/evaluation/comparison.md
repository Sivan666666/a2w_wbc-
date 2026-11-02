# WheelV2 50K paired-domain evaluation

3 seeds × 64 env × 200 seconds per method/domain. All frames; no warmup removal.
PLAY observation noise off; reset randomization, actuator delays and native terminations retained.
Position columns show mean ± sample SD of the three seed means (cm).

| Domain | Method | Native EE (cm) | Visual target world EE (cm) | Orientation (deg) | XY velocity error (m/s) | Yaw-rate error (rad/s) | Non-timeout resets |
|---|---|---:|---:|---:|---:|---:|---:|
| original | baseline | 4.39 ± 0.01 | 6.52 ± 0.16 | 1.76 | 0.0630 | 0.0562 | 3 |
| original | precision | 2.81 ± 0.02 | 5.58 ± 0.10 | 3.63 | 0.1119 | 0.0866 | 5 |
| expanded | baseline | 8.27 ± 0.11 | 13.33 ± 0.21 | 2.61 | 0.0771 | 0.0652 | 63 |
| expanded | precision | 2.58 ± 0.03 | 10.32 ± 0.43 | 3.78 | 0.1186 | 0.0889 | 7 |

Native EE = norm(link0-local x/y errors, world z error).
Visual world EE = distance to upstream visualization target; these are different metrics.
The expanded range is out-of-training-distribution for baseline. Range and reward both changed in precision.
This is random-target tracking, not a settled-point reach test or continuous handwriting benchmark.
