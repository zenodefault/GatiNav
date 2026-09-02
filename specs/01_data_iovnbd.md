# IO-VNBD Data Contract

This contract is limited to facts supported by `reference/iovnb_readme.md` and
the current `data/` file listing. Unknown file schemas, units, epochs, and
rates must be measured before implementation relies on them.

## 1. Directory layout

The data listing contains two top-level dataset trees:

```text
data/IO-VNBD/
├── Synchronised V abd S datasets/
└── Unsynchronised V and S Dataset/
```

The spelling `abd` in the synchronised directory name is preserved from the
checked-in files.

In the README, the vehicle stream is described as data from a research vehicle
and includes a GPS receiver, inertial navigation sensors, and wheel-speed
sensors, among other car sensors. The smartphone stream is described as
inertial navigation sensors and a GPS receiver in an Android smartphone,
sampling at 10 Hz.

- `V` = vehicle stream: vehicle-mounted GPS, inertial-navigation, wheel-speed,
  and other vehicle sensors described by the README.
- `S` = smartphone stream: Android-phone inertial-navigation sensors and GPS
  receiver described by the README, with the README-stated 10 Hz sampling.

The synchronised tree contains categorised and uncategorised collections,
including CSV recordings such as `V-*.csv` and `S-*.csv`. The unsynchronised
tree contains categorised vehicle data and an uncategorised combined
vehicle/smartphone collection.

## 2. Per-stream channel contract

Only the smartphone sampling rate and the broad sensor families below are
supported by the README. A blank-looking value is intentionally not inferred.

| Stream | Channel | Unit | Sampling rate | Timestamp format/epoch |
|---|---|---|---|---|
| V (vehicle) | Accelerometer / inertial-navigation sensor | ⚠️ VERIFY | ⚠️ VERIFY | ⚠️ VERIFY |
| V (vehicle) | Gyroscope / inertial-navigation sensor | ⚠️ VERIFY | ⚠️ VERIFY | ⚠️ VERIFY |
| V (vehicle) | Magnetometer | ⚠️ VERIFY | ⚠️ VERIFY | ⚠️ VERIFY |
| V (vehicle) | Wheel encoder / wheel-speed sensor | ⚠️ VERIFY | ⚠️ VERIFY | ⚠️ VERIFY |
| V (vehicle) | Force sensors / other car sensors | ⚠️ VERIFY | ⚠️ VERIFY | ⚠️ VERIFY |
| V (vehicle) | GNSS/GPS receiver | ⚠️ VERIFY | ⚠️ VERIFY | ⚠️ VERIFY |
| V (vehicle) | Ground truth | ⚠️ VERIFY | ⚠️ VERIFY | ⚠️ VERIFY |
| S (smartphone) | Accelerometer / inertial-navigation sensor | ⚠️ VERIFY | 10 Hz | ⚠️ VERIFY |
| S (smartphone) | Gyroscope / inertial-navigation sensor | ⚠️ VERIFY | 10 Hz | ⚠️ VERIFY |
| S (smartphone) | Magnetometer | ⚠️ VERIFY | 10 Hz | ⚠️ VERIFY |
| S (smartphone) | Wheel encoder | ⚠️ VERIFY | ⚠️ VERIFY | ⚠️ VERIFY |
| S (smartphone) | Force sensors | ⚠️ VERIFY | ⚠️ VERIFY | ⚠️ VERIFY |
| S (smartphone) | GNSS/GPS receiver | ⚠️ VERIFY | 10 Hz | ⚠️ VERIFY |
| S (smartphone) | Ground truth | ⚠️ VERIFY | ⚠️ VERIFY | ⚠️ VERIFY |

TODO: Load one representative synchronised `V-*.csv` and one representative
`S-*.csv`; inspect headers and typed values to measure every `⚠️ VERIFY` unit,
channel presence, sampling interval, and timestamp representation/epoch.

## 3. Synchronised versus unsynchronised

The file listing distinguishes a `Synchronised V abd S datasets` tree from an
`Unsynchronised V and S Dataset` tree. The README describes vehicle and
smartphone recordings but does not define the synchronization procedure,
alignment tolerance, or timestamp convention.

Project decision: use **only** the synchronised tree for experiments. In
`python/io/`, resample accepted streams onto a common 100 Hz clock. The
resampling policy, interpolation behavior outside outages, and alignment
tolerance are ⚠️ VERIFY.

TODO: Load one synchronised vehicle CSV and its paired smartphone CSV, compare
their timestamp columns and row times, and measure the observed offset,
jitter, overlap, and whether either stream already has a 100 Hz cadence.

## 4. Role assignment

This is the project data-use contract:

- Smartphone stream: primary filter input.
- Vehicle/wheel-encoder stream: ground truth for training and evaluation.
- GNSS: fused measurement and outage target for evaluation windows.

These roles are experiment decisions, not claims that the README supplies a
particular ground-truth column or fusion-ready schema.

TODO: Load one paired synchronised recording and verify that smartphone,
vehicle/wheel-speed, and GNSS columns can be mapped to these roles without
using any unsynchronised files.

## 5. Loading contract

All loaders in `python/io/` must return the following logical dataclass. The
field shapes and canonical units are the interface; source-column names and
source units remain ⚠️ VERIFY until a CSV is loaded.

```python
from dataclasses import dataclass

import numpy as np


@dataclass
class IOVNBDSample:
    t: np.ndarray          # shape (N,), float64 seconds
    gyro: np.ndarray      # shape (N, 3), rad/s
    accel: np.ndarray     # shape (N, 3), m/s²
    mag: np.ndarray        # shape (N, 3), normalized
    gnss: np.ndarray       # shape (M,), lat/lon/alt + accuracy + t
    gt_pose: np.ndarray    # shape (K,), position + velocity + orientation in ENU
```

The exact structured dtype or nested representation for `gnss` and `gt_pose`
is ⚠️ VERIFY; the semantic fields above are mandatory. All unit conversions
happen **only** in `python/io/`.

TODO: Load one synchronised CSV, enumerate its columns and dtypes, and create a
field mapping that proves the returned arrays have the stated shapes, dtypes,
units, and GNSS/ground-truth members.

## 6. Outage segmentation

For every continuous synchronised segment selected for evaluation:

1. Construct candidate artificial GNSS outage windows of exactly 30 s, 60 s,
   and 90 s where the segment contains sufficient data before, during, and
   after the window.
2. Remove GNSS updates only inside the selected outage interval.
3. Do not interpolate GNSS inside an outage.
4. Keep ground truth separate from filter inputs and use it only for scoring.
5. Choose outage start and end boundaries outside turns; a window that would
   begin or end mid-turn is rejected and another candidate is selected.
6. Record segment identifier, outage duration, start timestamp, end timestamp,
   and the boundary-rejection reason.

The definition of “turn,” the minimum turn-free margin, and the candidate
selection policy are ⚠️ VERIFY.

TODO: Load one synchronised file, derive the available heading/orientation or
vehicle-motion signal, calculate a turn indicator around every candidate
boundary, and measure the turn-free margin and rejection rate before fixing
thresholds.

## 7. Preprocessing pipeline specification

### Gravity removal

Use a low-pass Butterworth gravity estimate in `python/io/`: make the cutoff a
configuration parameter named `gravity_cutoff_hz`, estimate the slowly varying
gravity component from the body-frame accelerometer, and subtract that estimate
before integration. This option is selected because it keeps preprocessing
explicit, testable, and independent of the EKF implementation.

The filter order, boundary handling, and default cutoff are ⚠️ VERIFY.

TODO: Load one synchronised smartphone file, measure its accelerometer spectrum
and stationary intervals, then select and document a cutoff and filter order
that preserve vehicle motion while estimating the gravity component.

### DNN windows

Create 5–10 s windows from the preprocessed smartphone stream with a
configuration parameter for window length in that range and a separate
configuration parameter for stride. Windows must retain their source start and
end timestamps.

The training label schema and the stride value are ⚠️ VERIFY.

TODO: Load one synchronised smartphone/vehicle pair, measure usable continuous
duration and label coverage, then choose a stride that yields complete windows
with valid training labels and record the resulting window count.

## 8. Post-load sanity checklist

Each loaded synchronised stream must pass these checks before entering the
pipeline:

1. Stationary gyro magnitude is approximately Earth’s rotation rate at India’s
   latitude, on the `~10⁻⁵ rad/s` scale. ⚠️ VERIFY expected value.
2. Mean stationary accelerometer magnitude is approximately local gravity
   `g`; the acceptance tolerance is ⚠️ VERIFY.
3. GNSS sample spacing is consistent with the applicable stated rate; the
   vehicle rate is ⚠️ VERIFY, while the README states 10 Hz for the smartphone
   stream.
4. Timestamps are strictly increasing within each loaded channel and contain
   no duplicate rows.
5. No required field contains NaN or infinite values; the missing-value
   policy for optional channels is ⚠️ VERIFY.

TODO: Load one representative synchronised CSV per stream and compute the five
checks, reporting gyro and accelerometer stationary statistics, GNSS spacing,
timestamp monotonicity/duplicates, and invalid-value counts.
