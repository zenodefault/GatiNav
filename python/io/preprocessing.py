"""Preprocessing pipeline (specs/01 section 7).

gravity_remove(accel, fs, gravity_cutoff_hz=1.0) estimates the slowly varying
gravity component in the body frame with an order-2 Butterworth low-pass
(scipy filtfilt, edge-padded; order/boundary/cutoff VERIFY per §7) and
subtracts it from the body-Z axis only, leaving horizontal channels untouched
(project decision: valid for ENU-aligned bodies; VERIFY for arbitrary phone
mounts).

window_segmentation(stream, window_s, stride_s) slices a stream into complete
windows of window_s seconds with stride stride_s seconds (specs/01 §7: window
length in [5, 10], separate stride parameter, windows retain their source
start/end timestamps through stream.t).
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt

from python.io.iovnb_loader import IOVNBDSample

_FILTER_ORDER = 2  # VERIFY per specs/01 §7


def gravity_remove(accel, fs, gravity_cutoff_hz=1.0):
    """Return body-frame linear acceleration with the slowly varying gravity
    component removed (order-2 Butterworth LP, filtfilt edge-padded)."""
    accel = np.asarray(accel, dtype=np.float64)
    if accel.ndim != 2 or accel.shape[1] != 3:
        raise ValueError(f"accel: expected (N, 3), got {accel.shape}")
    fs = float(fs)
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")
    if not 0.0 < gravity_cutoff_hz < fs / 2.0:
        raise ValueError(f"gravity_cutoff_hz must be in (0, fs/2), got "
                         f"{gravity_cutoff_hz}")
    if accel.shape[0] <= 10:
        raise ValueError("accel: too short for filtfilt (need > 10 samples)")
    sos = butter(_FILTER_ORDER, gravity_cutoff_hz, fs=fs, btype="low",
                 output="sos")
    gravity = np.column_stack(
        [sosfiltfilt(sos, accel[:, j]) for j in range(3)]
    )
    linear = accel.copy()
    linear[:, 2] -= gravity[:, 2]  # subtract body-Z gravity component only
    return linear


def window_segmentation(stream, window_s, stride_s):
    """Slice stream into complete windows; each window is an IOVNBDSample
    retaining its source start/end timestamps."""
    window_s = float(window_s)
    if not 5.0 <= window_s <= 10.0:
        raise ValueError(f"window_s must be in [5, 10] per specs/01 §7, got "
                         f"{window_s}")
    stride_s = float(stride_s)
    if stride_s <= 0:
        raise ValueError(f"stride_s must be positive, got {stride_s}")
    dt = float(np.median(np.diff(stream.t)))
    n_win = int(round(window_s / dt))
    n_stride = max(1, int(round(stride_s / dt)))
    n = stream.t.size
    count = 0 if n < n_win else 1 + (n - n_win) // n_stride
    windows = []
    for k in range(count):
        lo = k * n_stride
        hi = lo + n_win
        windows.append(IOVNBDSample(
            t=stream.t[lo:hi],
            gyro=stream.gyro[lo:hi],
            accel=stream.accel[lo:hi],
            mag=stream.mag[lo:hi],
            gnss=stream.gnss[lo:hi],
            gt_pose=stream.gt_pose[lo:hi],
        ))
    return windows


def is_stationary(accel, gyro, accel_variance_threshold=0.01,
                  gyro_variance_threshold=1e-5):
    """Return a per-sample stationarity decision from signal variances."""
    accel = np.asarray(accel, dtype=np.float64)
    gyro = np.asarray(gyro, dtype=np.float64)
    if accel.shape != gyro.shape or accel.ndim != 2 or accel.shape[1] != 3:
        raise ValueError("accel and gyro must both have shape (N, 3)")
    if accel_variance_threshold <= 0 or gyro_variance_threshold <= 0:
        raise ValueError("variance thresholds must be positive")
    a_var = np.var(accel, axis=1)
    g_var = np.var(gyro, axis=1)
    return (a_var <= accel_variance_threshold) & (g_var <= gyro_variance_threshold)