# Stationary deep dive

## Exact failing-test path
`test_stationary_30_second_drift_is_below_two_metres` and `test_driving_30_second_drift_is_below_one_hundred_metres` call `_sessions()` -> `load_pair()` -> `_integrate()` -> `ErrorStateEKF()` -> `predict()` for each sample.
The loop is raw integration, not a separate numerical integrator. ZUPT is disabled; no GNSS update occurs; there is no GNSS-aided period before the measured window. The EKF cold-starts with identity attitude, zero velocity, zero position, zero gyro/accel biases, identity P0, and zero error state.

## Stationary segment instrumentation
Segment start: 5.999 s; duration: 30.001 s.
Existing detector fire percentage: 87.04% (thresholds: 1 s window, temporal accel variance sum < 1.45, temporal gyro variance sum < 0.006; first 99 samples warm up).
🚨 SUSPECT S2 CONFIRMED: detector fire percentage is below 90%.

Mean residual specific force, body frame: `[ 0.05754884 -0.04562492  9.84005681]` m/s²; magnitude 9.840331 m/s².
Mean residual specific force, ENU: `[-0.33577207  0.04515138  0.05901081]` m/s²; magnitude 0.343895 m/s².
The ENU residual is not approximately 0.4 m/s² horizontal; therefore the specified S3/S4 trigger is not met.

Drift scaling exponent: 2.854; approximately t² acceleration-error.
Attitude at segment start: roll/pitch/yaw = [0. 0. 0.] deg.
Accel-derived rest expectation: roll/pitch/yaw = [1.26325615 0.52467646 0.        ] deg; mean body accel = [ 0.090424 -0.217698  9.871816] m/s².

Plots: `stationary_deepdive_drift.png` and `stationary_zupt_fire.png`.
