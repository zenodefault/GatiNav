# ML Noise Adaptation Specification Skeleton

This specification defines the role, data contract, training protocol, and
deployment boundaries for the learned IMU-noise adaptation module. The
authoritative CNN architecture and trajectory-error loss must be transcribed
from the Brossard paper and the reference source; they are not inferred here.

## 1. Network role

The network is a small 1D-CNN consuming raw smartphone gyroscope and
accelerometer windows. It outputs per-timestep measurement-noise scale factors
for the ZUPT pseudo-measurement. In project terms, **the network learns when to
trust the IMU**.

The baseline/reference head outputs the three positive diagonal variances
directly. Its output is passed as `R_zupt = diag(output)` at each accepted
ZUPT opportunity.

### OUR EXTENSION: vehicle-speed head

This project adds a second head beyond the reference paper: estimate vehicle
speed from IMU alone, with no OBD-II input. The head is supervised using the
vehicle/wheel-encoder ground truth assigned in `specs/01_data_iovnbd.md`.

The speed target field, output units, loss weighting, head sharing, and
inference-time use of the speed estimate are ⚠️ VERIFY.

TODO: Load one paired synchronised smartphone and vehicle recording, identify
the wheel-encoder speed field, and define the speed target and alignment
procedure from measured columns rather than assumptions.

## 2. Authoritative architecture

### [ARCHITECTURE BLOCK — HUMAN: transcribe layer types, kernel sizes,
channels, activations, input window length, output dimensions from
reference/ai-imu-dr/src model files + paper architecture table]
NoiseNet input is `(batch, 6, 100)` with channels
`[acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]`. It uses Conv1d `6->64`
and Conv1d `64->64`, both with kernel 5, stride 2, padding 2 and ReLU;
flatten to 1600; Linear `1600->64` with ReLU; and Linear `64->3` with
Softplus. The output is positive `[sigma_x^2, sigma_y^2, sigma_z^2]` and
the exact parameter count is 125187. The speed head is deferred.

TODO: Read the exact model files under `reference/ai-imu-dr/src/` and the
architecture table in `reference/brossard_ai_imu_dr.pdf`; transcribe every
layer, kernel size, channel count, activation, input window length, and output
dimension here before implementing or exporting the network.

No architecture choice, layer count, tensor shape, activation, or output
dimension may be added from general CNN knowledge.

## 3. Input pipeline

- Input channels are raw smartphone accelerometer followed by gyroscope.
- Windows are 1.0 s at 100 Hz: 100 samples per window. This overrides the
  generic 5–10 s placeholder for NoiseNet reactivity.
- Window stride is 0.5 s for the baseline.
- Per-channel standardization uses mean and standard deviation computed on the
  training split **only**.
- Validation and test windows use the frozen training-split statistics; they
  must never contribute to standardization statistics.
- Window metadata retains source segment, driver, start timestamp, and end
  timestamp.

Missing windows are rejected, no padding is used, and zero standard deviations
are replaced by 1.0. Frozen training statistics are used for validation/test.

TODO: Load one synchronised smartphone file, enumerate the measured gyro and
accelerometer columns, then implement a train-only statistics artifact and
verify that validation/test rows cannot alter it.

## 4. Training protocol

### Driver-level cross-validation

Use leave-one-driver-out cross-validation over the usable driver labels in the
selected synchronised tree. The complete `data/` listing exposes five distinct
labels—Driver A, Driver B, Driver C, Driver D, and Driver E—but the
synchronised categorised tree visibly contains Driver A, Driver B, Driver D,
and Driver E; Driver C is visible under the unsynchronised tree.
⚠️ VERIFY the final synchronised driver count and whether Driver C has a
usable synchronised counterpart before fixing the fold count. The README does
not specify whether every label has the same number of usable paired
recordings.

Each fold holds out one driver for testing, keeps driver identity disjoint
between training and validation/test, and computes normalization statistics
from training data only.

TODO: Count and map all driver-labelled folders and paired recordings in
`data/IO-VNBD/Synchronised V abd S datasets/`, then verify the usable driver
count and whether every fold has synchronised smartphone/vehicle data.

### Loss

The baseline trains only the NoiseNet head. For positive target variances `y`
and prediction `ŷ`, use `L = mean((log(ŷ) - log(y))^2)`. Targets are supplied
by the offline reference/filter procedure. No speed target or speed loss is
used because the speed head is deferred.

### [EQUATION BLOCK 1 — HUMAN: paste exact loss from
reference/brossard_ai_imu_dr.pdf section X / reference/ai-imu-dr/src/]
<!-- HUMAN fills this -->

The exact trajectory-error loss, any regularization, multi-head weighting, and
speed-loss definition are ⚠️ VERIFY.

TODO: Transcribe the paper/source loss exactly, identify its required target
trajectory fields, then define and test the additional speed-head loss without
changing the reference-head objective.

### Optimizer and stopping

- Optimizer: Adam.
- Early stopping monitors a validation drift metric.
- Validation drift is the mean log-variance squared error on validation data;
  lower is better. Defaults are learning rate `1e-3`, batch size `32`,
  maximum epochs `100`, patience `10`, and minimum improvement `1e-6`.
- The checkpoint with the lowest validation drift is selected.
- The validation drift metric definition, patience, minimum improvement,
  learning rate, batch size, epoch limit, and checkpoint-selection rule are
  ⚠️ VERIFY.
- Training runs must be deterministic with an explicitly recorded seed and
  deterministic data split.

TODO: Establish the drift metric and training hyperparameters from the
reference where available, then measure their stability across each verified
leave-one-driver-out fold.

## 5. Export contract

The export uses a static `(1, 6, 100)` input and ONNX opset 17. The required
mobile-compatible operators are Conv, Relu, Flatten, Gemm, and Softplus.
TFLite conversion is deferred until the Android build environment is present;
the export script prints this compatibility status.

The deployment export path is:

```text
PyTorch → TorchScript → ONNX → TFLite
```

The exported model must preserve the reference noise-adaptation outputs and
the OUR EXTENSION speed output, with output names, shapes, dtypes, and scaling
recorded as part of the model artifact.

### Operator compatibility gate

Every operator in the finalized architecture must be confirmed as supported
by the selected TFLite conversion/runtime path **before finalizing the
architecture**. This export test is scheduled for **week 3, not week 6**.

| Operator / operation | Source occurrence | TorchScript | ONNX | TFLite | Status |
|---|---|---|---|---|---|
| All architecture operators | ⚠️ VERIFY | ⚠️ VERIFY | ⚠️ VERIFY | ⚠️ VERIFY | ⚠️ VERIFY |
| Noise-scale output transform | ⚠️ VERIFY | ⚠️ VERIFY | ⚠️ VERIFY | ⚠️ VERIFY | ⚠️ VERIFY |
| OUR EXTENSION speed head | ⚠️ VERIFY | ⚠️ VERIFY | ⚠️ VERIFY | ⚠️ VERIFY | ⚠️ VERIFY |

TODO: After transcribing the architecture, enumerate every graph operator,
export a representative checkpoint through all four formats in week 3, run
the TFLite model on representative windows, and compare outputs against
PyTorch within a documented tolerance.

The ONNX opset, dynamic/static input shape policy, quantization policy,
delegate/runtime, and numerical equivalence tolerance are ⚠️ VERIFY.

## 6. Acceptance criteria

The CNN-adapted EKF must beat the fixed-noise EKF on drift for at least 70% of
the held-out test outage windows.

Report, separately for 30 s, 60 s, and 90 s outages:

- Absolute Trajectory Error (ATE).
- Relative Pose Error (RPE).
- Percentage of test outage windows where CNN-adapted drift is lower than
  fixed-noise drift.
- Error relative to distance travelled, with a target below 1–2%.

For this baseline, drift is final horizontal position error, ATE is the root
mean squared horizontal position error over each outage, and RPE is the root
mean squared one-step horizontal displacement error. No alignment is applied;
lower is better; ties are not wins. Report wins divided by evaluated windows
separately for each outage duration.

TODO: Load held-out synchronised segments, carve the 30/60/90 s outages using
`specs/01_data_iovnbd.md`, run both fixed-noise and CNN-adapted EKFs, and
compute the acceptance metrics per outage length without using ground truth
as a filter input.
