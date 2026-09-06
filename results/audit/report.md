# Physics audit

## Findings

### Gravity reconciliation
The code defines `f_body = accel - accel_bias`, rotates it with `R`, and adds `g_world = [0, 0, -9.80665]`. This matches `a_world = R(q) f_body + g_world`; it does not subtract a constant body-frame gravity vector. If a constant body vector `g_body` were subtracted instead, a tilted mount would leave the residual `R(q)g_body - g_world`, which rotates with mount orientation and integrates into quadratic position drift. No ROOT-CAUSE CANDIDATE #1 is raised by the formula itself.

### Timestamp correction: session offset -7.100 s
Cross-correlation changed from -7.100 s to 0.000 s.

### Static test: drift = 47.573 m
Vector ENU drift: `[-41.45130118 -19.24770571  13.21089787]`. Effective constant acceleration: `[-0.20725651 -0.09623853  0.06605449]` m/s²; this is not approximately a g-sized vector.

### Acceptance status
Before/after timestamp correction: static drift and raw IMU drift-scaling values are unchanged because the correction aligns vehicle reference time to phone time; it does not alter the phone IMU samples.
Stationary <2 m: NOT MET in this audit segment. Drift exponent ≈1: NOT MET (measured 2.85). The requested driving <100 m gate was not established by this diagnostic-only audit.

### Round 2 diagnosis
S2 (ZUPT detector coverage, diagnostic segment S-S1): CONFIRMED below the 90% line; fire rate = 87.04% on stationary timesteps.
S3/S4 (tilt/attitude residual trigger): ELIMINATED by the specified 0.4 m/s² horizontal criterion; ENU residual magnitude = 0.343895 m/s².
S1: ELIMINATED; the test path exercises `ErrorStateEKF` correctly.

### Drift scaling: exponent = 2.85
Fit preference: `t²`; durations (s) = [ 5. 10. 20. 30.]; drifts (m) = [  0.69360807   9.19063922  47.57323163 122.93423011].

### Unit audit
Gyro min/max: (-1.1627, 1.3556); accel min/max: (-8.3689, 16.1533).
Gyro deg/s candidate heuristic: True; accel SI plausible: True.

### Cross-correlation alignment
Lag after correction: 0 samples (0.000 s).

## Ranked root-cause hypothesis
1. Cross-stream timing/alignment, because the measured lag exceeds 0.2 s.
2. Current gravity/tilt residual or phone orientation error, because static drift scales superlinearly, but not at a g-sized acceleration.
3. Sensor-unit or axis-convention error if unit checks fail.

## Round 3 — ZUPT detector calibration

The detector now computes variance across each 100-sample time window
(`axis=2`) and sums the three channel variances. Samples 0–98 are warm-up
and cannot produce ZUPT decisions. Calibrated thresholds are
accelerometer score `< 1.45 (m/s²)^2` and gyroscope score
`< 0.006 (rad/s)^2`.

Held-out stationary segments `S-Vw15`, `S-Vta8`, `S-Vta2`, `S-Vw14c`,
and `S-Vw1` fired at `95.08%`, `100.00%`, `97.48%`, `95.88%`, and
`99.53%`. Held-out driving segments `S-Vw16b`, `S-Vw6`, `S-Vw10`,
`S-Vtb12`, and `S-Vta11` fired at `0.00%` each.

Across labeled replay samples, score distributions were:

| Label | Accel min | Accel median | Accel max | Gyro min | Gyro median | Gyro max |
|---|---:|---:|---:|---:|---:|---:|
| Stationary | 0.01165 | 2.57183 | 36.69249 | 0.000022 | 0.030055 | 0.600832 |
| Driving | 0.00775 | 6.35434 | 62.32490 | 0.000016 | 0.089057 | 0.960367 |

Calibration evidence was regenerated in
`results/audit/stationary_zupt_fire.png`. The selected categorized sessions
are primarily Driver E; the available dataset does not provide five distinct
driver identities with qualifying stationary segments.

The full suite after this detector fix was `77 passed, 5 failed`. The five
failures are three pre-existing synthetic evaluation expectations that assume
the broken detector does not fire, plus the two cold-start inertial acceptance
tests (195.163 m stationary and 549.431 m driving). The detector-specific
tests passed: `5 passed, 0 failed`.
