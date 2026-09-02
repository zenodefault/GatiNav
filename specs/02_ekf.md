# Error-State EKF Specification Skeleton

This document defines interfaces, conventions, and acceptance tests for the
error-state EKF. Exact mathematical definitions must be copied from the
project reference paper or reference implementation; they are intentionally
not reproduced here.

## 1. State vector

The filter error state has 15 elements:

| Component | Symbol | Meaning | Units | Initial value |
|---|---|---|---|---|
| Orientation error | `δθ` | Small orientation error | rad | ⚠️ VERIFY |
| Velocity error | `δv` | Navigation-frame velocity error | m/s | ⚠️ VERIFY |
| Position error | `δp` | Navigation-frame position error | m | ⚠️ VERIFY |
| Gyroscope bias error | `δb_g` | Gyroscope bias error | rad/s | ⚠️ VERIFY |
| Accelerometer bias error | `δb_a` | Accelerometer bias error | m/s² | ⚠️ VERIFY |

The implementation shall use a documented, stable ordering matching the table:
orientation error, velocity error, position error, gyro bias error, and
accelerometer bias error.

TODO: Load the exact state ordering and initialization policy from
`reference/brossard_ai_imu_dr.pdf` and `reference/ai-imu-dr/src/`, then encode
known-answer tests for each block and the complete 15-element vector.

## 2. Coordinate frames

- **Body frame:** Sensor/device frame attached to the moving platform.
- **Navigation frame:** Local ENU frame: East, North, Up.
- **Rotation convention:** Hamilton quaternions in `(w, x, y, z)` order.
- **Rotation direction:** Body-to-world rotation uses `R(q)`, consistent with
  `.github/copilot-instructions.md`.
- **Gravity convention:** `+9.80665 m/s²` points UP in ENU; gravity is removed
  from acceleration before integration, consistent with the project rules.

The exact perturbation side, error-angle convention, quaternion composition
order, and reset operation are ⚠️ VERIFY.

TODO: Extract the frame and perturbation conventions from the paper and
reference source, then test one body-to-ENU rotation with a known-answer
orientation.

## 3. Predict step

### 3.1 Nominal-state propagation

The nominal state shall propagate from gyroscope and accelerometer samples:
gyro measurements drive orientation, and gravity-compensated accelerations
drive velocity followed by position.

### [EQUATION BLOCK 1 — HUMAN: paste exact equations from
reference/brossard_ai_imu_dr.pdf section X / reference/ai-imu-dr/src/]
<!-- HUMAN fills this -->

Required fill scope: quaternion/orientation propagation, acceleration
transformation into ENU, gravity handling, and velocity/position propagation.
The integration scheme and timestamp convention are ⚠️ VERIFY.

### 3.2 Error-state transition matrix

The transition matrix `Φ` shall propagate the 15-element error state over one
sample interval. Its block layout, signs, discretization, and dependence on
the nominal state must come from the references.

### [EQUATION BLOCK 2 — HUMAN: paste exact equations from
reference/brossard_ai_imu_dr.pdf section X / reference/ai-imu-dr/src/]
<!-- HUMAN fills this -->

TODO: Copy the exact continuous/discrete transition definition and document
the matrix block ordering against Section 1.

### 3.3 Process noise structure

Process noise `Q` shall identify which blocks are nonzero and what drives
each block, including measurement noise and bias evolution terms where the
reference specifies them. Noise parameter names, correlations, and random
walk assumptions are ⚠️ VERIFY.

### [EQUATION BLOCK 3 — HUMAN: paste exact equations from
reference/brossard_ai_imu_dr.pdf section X / reference/ai-imu-dr/src/]
<!-- HUMAN fills this -->

TODO: Extract the exact `Q` construction, noise-density units, and parameter
defaults; verify them against one deterministic propagation test.

## 4. Update step

### 4.1 Generic measurement update

The filter shall expose a generic measurement update accepting a measurement,
predicted measurement, measurement Jacobian, and measurement covariance. The
innovation, gain, covariance update, and error-state injection/reset must be
filled from the reference rather than inferred.

### [EQUATION BLOCK 4 — HUMAN: paste exact equations from
reference/brossard_ai_imu_dr.pdf section X / reference/ai-imu-dr/src/]
<!-- HUMAN fills this -->

TODO: Copy the exact generic update and reset equations, including the
covariance form required by the reference implementation.

### 4.2 GNSS position measurement

GNSS position updates shall be applied only when GNSS is available. The model
shall define the position residual, the structure of `H`, and measurement
covariance `R`; coordinate conversion into the local ENU frame occurs at the
IO boundary.

### [EQUATION BLOCK 5 — HUMAN: paste exact equations from
reference/brossard_ai_imu_dr.pdf section X / reference/ai-imu-dr/src/]
<!-- HUMAN fills this -->

The GNSS accuracy field and its mapping into `R` are ⚠️ VERIFY.

TODO: Load one synchronised GNSS record, verify its position/accuracy fields,
and copy the exact GNSS Jacobian and covariance policy from the references.

## 5. ZUPT

### 5.1 Stationarity detector

A ZUPT candidate is detected using accelerometer-variance and
gyroscope-variance thresholds over a configurable window.

- Accelerometer variance threshold: ⚠️ VERIFY; tune on IO-VNBD.
- Gyroscope variance threshold: ⚠️ VERIFY; tune on IO-VNBD.
- Detector window duration: ⚠️ VERIFY.
- Minimum/maximum stationary duration: ⚠️ VERIFY.

TODO: Load synchronised smartphone segments, label stationary intervals using
the available vehicle/wheel-speed or ground-truth signal, measure both
variance distributions, and tune thresholds on a training split only.

### 5.2 Zero-velocity pseudo-measurement

When stationarity is accepted, apply a zero-velocity pseudo-measurement. Its
residual, Jacobian, covariance, and interaction with the generic update are
reference-controlled placeholders.

### [EQUATION BLOCK 6 — HUMAN: paste exact equations from
reference/brossard_ai_imu_dr.pdf section X / reference/ai-imu-dr/src/]
<!-- HUMAN fills this -->

TODO: Extract the ZUPT measurement model and covariance from the references,
then add a known-answer stationary update test.

## 6. Bias observability

Gyroscope and accelerometer bias states are corrected indirectly through
measurement updates. ZUPT provides repeated information that velocity should
be zero during detected stationary intervals, constraining bias-driven motion.
GNSS position updates constrain accumulated position error and, over repeated
motion, provide additional information for correcting both bias states.

The rate of convergence, required motion diversity, and observability
limitations are ⚠️ VERIFY from the reference and IO-VNBD experiments.

TODO: Run separate replay experiments with ZUPT only, GNSS only, and both;
measure gyro-bias and accel-bias convergence against the held-out reference
stream.

## 7. Known-answer tests

The implementation MUST pass all three tests below. Test inputs shall be
synthetic, deterministic, and generated without external data files.

1. **Zero-noise synthetic circle:** final position error is `< 0.5 m` after
   60 s.
2. **Injected gyro bias:** inject a `0.01 rad/s` gyro bias; estimated bias
   converges within `5%` of the injected value.
3. **Stationary with ZUPT:** run for 60 s with ZUPT enabled; position drift is
   `< 1 m`.

The exact circle radius, speed, orientation, sample rate, initial covariance,
noise covariance, bias injection axis, convergence interval, and error metric
are ⚠️ VERIFY.

TODO: Define these test fixtures from the human-filled equations, implement
them under `python/tests/`, and record the tolerances as executable assertions.

## 8. Implementation notes

- Filter core uses pure NumPy; do not import or use Torch in the filter core.
- Execution is deterministic; every stochastic test or generated fixture uses
  an explicitly seeded RNG.
- Every matrix operation asserts expected shapes before multiplication,
  addition, inversion, or update.
- The filter has no global mutable state.
- The filter is a class exposing `.predict()` and `.update()`.
- Configuration, state, and covariance ownership must be explicit and
  instance-local.

The class name, constructor signature, covariance storage convention, and
public state-output format are ⚠️ VERIFY.

TODO: Reconcile the implementation interface with the human-filled reference
equations, then write interface and shape tests before implementing the core.
