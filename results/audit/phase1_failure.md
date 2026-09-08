# Phase 1 calibration and alignment gate evidence

Date: 2026-09-08

## Measured gates

| Gate | Required | Measured | Status |
|---|---:|---:|---|
| Synthetic phone-to-vehicle rotation recovery | <=2 deg / <=0.1 m/s² | 7 known-answer tests pass | PASS |
| Stationary DR | <2 m over 30 s | `test_stationary_30_second_drift_is_below_two_metres` passes | PASS |
| Driving DR | <100 m over 30 s | 3942.178 m | FAIL |

The failed driving measurement is from
`test_driving_30_second_drift_is_below_one_hundred_metres`, which selects
`Categorised IOVNB Dataset/Vta (Driver E)/Vta26/S-Vta26.csv` after earlier
sessions lack a 15 s IMU-static calibration interval. The selected 30 s
window starts at t=0.0 s. The calibration diagnostic reported a zero GNSS
seed speed and zero mounting yaw, so it had no usable pre-outage GNSS course.

## Top three diagnoses

1. **No usable pre-outage anchor in the tested window.** The first moving
   sample is at t=0.0, therefore a GNSS-seeded dead-reckoning state cannot be
   formed from pre-outage data. A Phase 2 mode/anchoring solution is out of
   scope and must not be added here.
2. **Yaw estimator has no qualifying straight GNSS segment.** It retained
   `yaw_mount = 0.0 deg` for the selected session. This is insufficient
   straight GNSS-valid driving, not evidence that a yaw estimate was made.
3. **Static calibration availability is sparse.** Many candidate sessions
   rejected calibration with `ValueError: no IMU-static segment of 15 s`.
   Calibration coverage needs a defined operational fallback before a broader
   benchmark can be considered representative.

No gate was weakened or bypassed. Phase 2 was not started.
