# Evaluation Specification

This document defines the evaluation outputs for the dead-reckoning system.
Equations and KITTI-convention details below are provisional until checked
against the actual KITTI odometry dev-kit source.

## 1. Evaluation conventions

- Use the ENU frame and SI units.
- Store timestamps as strictly monotonic float64 seconds.
- Compare estimated and reference trajectories only after synchronising their
  timestamps at the IO boundary.
- Ground truth must not be supplied to the filter during replay or evaluation.
- Report results separately for 30 s, 60 s, and 90 s GNSS outages.
- Lower error is better. Ties are not wins.

The project currently reports horizontal position metrics without trajectory
alignment. Whether KITTI-style evaluation requires an alignment transform is
⚠️ VERIFY against the dev-kit source before finalizing this specification.

## 2. Absolute Trajectory Error (ATE)

Let the estimated position at evaluation sample `i` be `p_est_i` and the
reference position be `p_ref_i`. The project horizontal error uses the ENU
horizontal components:

```text
e_i = (p_est_i - p_ref_i)[E,N]
```

The project ATE placeholder is:

```text
ATE = sqrt((1 / N) * sum_i ||e_i||^2)
```

⚠️ VERIFY: Confirm the exact ATE definition, trajectory pairing, alignment
policy, and aggregation convention against the KITTI odometry dev-kit source.

⚠️ VERIFY: Confirm whether the KITTI dev-kit ATE equivalent is reported as
translational RMSE over all poses, and whether this project should retain the
project-specific horizontal-only, no-alignment definition.

Report ATE in metres for every outage duration and configuration.

## 3. Relative Pose Error (RPE)

For a relative interval from sample `i` to sample `j`, define the estimated and
reference relative motions as:

```text
T_est_i,j = inv(T_est_i) * T_est_j
T_ref_i,j = inv(T_ref_i) * T_ref_j
E_i,j = inv(T_ref_i,j) * T_est_i,j
```

The provisional translational RPE is:

```text
RPE_trans = sqrt((1 / M) * sum_k ||trans(E_k)||^2)
```

⚠️ VERIFY: Confirm the exact KITTI relative-pose interval, pose composition
order, error transform, and translational aggregation formula against the
actual dev-kit source.

⚠️ VERIFY: Confirm whether rotational RPE is required in addition to
translational RPE. If reported, define its units and aggregation from the
dev-kit source.

The existing project acceptance metric is a horizontal one-step displacement
RPE. Until the KITTI convention is confirmed, retain that metric as:

```text
RPE_project = sqrt((1 / (N - 1)) * sum_i
                   ||(p_est_i - p_est_{i-1})
                    - (p_ref_i - p_ref_{i-1})||_[E,N]^2)
```

⚠️ VERIFY: Confirm whether this project-specific one-step horizontal RPE may
be reported as the project result alongside the KITTI-convention RPE.

## 4. Required metric reporting

For each outage duration and configuration, report:

- ATE in metres.
- RPE in metres, with the interval definition stated.
- Final horizontal drift in metres.
- Distance travelled in metres.
- Drift divided by distance, as a percentage.
- Number of evaluated outage windows.
- Number and percentage of windows where the CNN configuration beats the
  fixed-noise baseline.

The comparison table must identify the configuration, outage duration, and
whether trajectory alignment was applied.

## 5. `results/final_metrics.md` format

Use the following summary-table structure:

```markdown
## ATE / RPE / drift by outage duration

| Duration s | Config | Alignment | ATE m | RPE m | RPE interval | Drift m | Distance m | Drift/distance % | Windows | Wins | Win rate % |
|---:|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 30 | TBD | none | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
```

Use one row per configuration and outage duration. Use `N/A` when a metric
cannot be computed; do not use `nan%`.

Use the following per-window structure when window-level evidence is needed:

```markdown
## Per-window comparison

| Window | Segment | Outage s | Config | ATE m | RPE m | Drift m | Distance m | Drift/distance % | Win vs baseline |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| TBD | TBD | 30 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
```

⚠️ VERIFY: Confirm the final KITTI-compatible column names, alignment label,
RPE interval label, and required precision before finalizing the results file.

## 6. Demo-video checklist

- Show the native Android application and sensor-capture screen.
- Show timestamp or elapsed-time evidence before and during the outage.
- Clearly mark GNSS outage start and end.
- Show the estimated trajectory during the outage.
- Show the reference trajectory or evaluation overlay where permitted.
- Display the outage duration and active configuration.
- Show the 30 s, 60 s, and 90 s result summaries when those runs exist.
- Show ATE, RPE, drift, drift/distance, and window win-rate values.
- Keep map orientation and frame labels consistent with ENU conventions.
- Do not display ground-truth measurements as filter inputs.
- Include enough run metadata to identify the dataset segment and software
  configuration.
- Ensure every claim shown in the video is supported by
  `results/final_metrics.md`.

⚠️ VERIFY: Confirm the final video duration, resolution, required UI screens,
and whether live reference-track display is allowed in the submission.
