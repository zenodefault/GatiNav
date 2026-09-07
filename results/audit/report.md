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

## Round 4 — AI speed-head gate (negative result)

NoiseNet gained a fourth output (per-axis ZUPT variances + a speed head
regressing `log(1 + v)` against IO-VNBD wheel-encoder speed, composite loss
with the log-variance ZUPT objective). Training data was doubled by fixing
session discovery: case-insensitive vehicle pairing (`V-vta*.csv`) and a
shallower driver-label path recovered 32 → 72 synchronised sessions
(Driver A: 6, B: 1, D: 1, E: 64), with Driver B previously mislabelled
"unknown". The Uncategorised tree is excluded (duplicate stems → leakage).

Held-out (leave-one-driver-out) fractional speed RMSE — the <10%-of-distance
benchmark gate:

| Fold (test driver) | Fractional RMSE (all) | Fractional RMSE (driving) | Windows |
|---|---:|---:|---:|
| Driver A | 85.8% | 72.6% | 5,268 |
| Driver B | 94.5% | 87.3% | 1,235 |
| Driver D (deployed) | **76.9%** | **68.4%** | 1,338 |
| Driver E | 127.7% | 109.3% | 11,730 |

On the deployed model over the full 19,571-window cache:
`corr(pred, GT) = 0.054`; stopped windows predict ≈8.5 m/s (GT 0.11),
highway ≈12.8 m/s (GT 19.4) — the head regresses onto the marginal mean.

**Verdict: the gate failed, 7–9× off target.** The failure is identifiability,
not training: per-session accelerometer RMS is flat across all speed bands in
several sessions (stopped-but-vibrating idle, potholes, phone-mount variance),
so 1 s IMU windows do not determine speed (log-corr ≤ 0.33; the noise head
trains cleanly on the same windows, val log-variance 5.6). The speed head is
kept in the architecture (forward-compatible checkpoint) but is not wired
into the EKF. Candidate follow-up: within-session GNSS-supervised calibration
(learn the vibration→speed map during the first minute of GNSS-available
driving, then extrapolate into the outage) instead of cross-session
regression.

## Round 5 — GNSS-supervised calibration engine (negative result)

Follow-up prototype (`python/ml/calibrate_speed_proto.py`): per-session ridge
regression of log(1+v) on log-vibration window features, supervised by the
vehicle's own speed, evaluated time-held-out. Three configurations, all on
clean labels (the ceiling — GNSS-differentiated phone velocity is noisier):

| Configuration | Driving-only fractional RMSE |
|---|---|
| Session-start calibration (first 60–90 s), full session test | median 71.5%, session-weighted 100.1% |
| Rolling calibration (GNSS-available minutes immediately before a candidate 60 s outage), outage-slice test | mean 76.3%, median 62.3%, p25 47.5% |
| Cross-session CNN (Round 4, for comparison) | deployed 68.4% |

**Verdict: the within-session GNSS-calibration idea fails too.** Even
calibrating on the same road segment moments before the outage, 1 s-window
vibration energy does not determine speed: the scatter floor (acceleration
transients, gear changes, surface texture, idle) sits at ~50–80% fractional
error across all three designs — 5–8× above the <10%-of-distance benchmark
gate. Only 1% of 3,635 rolling outage samples landed under 25%.

Conclusion across Rounds 4–5: **broadband vibration energy does not encode
vehicle speed at useful SNR on IO-VNBD**, within or across sessions. The
published DRNet-class results on this dataset exploit a different mechanism
(per-window attitude estimation + forward-axis acceleration integration with
the network learning de-noising), not vibration amplitude — architecturally
more than the current trunk can express and not approximated by the energy
features. The speed head remains in the architecture, unused by the EKF.
The remaining drift levers are therefore: map matching + NHC engagement
(lateral/heading), ZUPT (stops, already ~4–6 m/30 s), and pre-outage bias
convergence quality.
