"""Residual-learning dataset (Blueprint Phase 1.2).

Builds training windows for the forward-acceleration residual net:

Input  (N, W): attitude-compensated forward-axis acceleration windows.
Target (N,):   residual = a_measured_forward - a_true_forward at the window
               END sample (causal: only past samples are used, so the model
               is deployment-shaped for a sliding 0.5 s stride feed).

a_true_forward is the wheel-encoder forward acceleration: the time
derivative of the vehicle speed (gt_pose.vx/vy magnitude on the common
clock). The plan's literal formula ``(Input + GT) - Input`` is
self-canceling (= GT); its stated intent — "the difference between the
measured (but compensated) acceleration and the true acceleration of the
vehicle" — is implemented instead, so ``a_denoised = a_measured -
residual`` recovers the true signal.

Cache: results/audit/denoise_windows.npz (version-tagged); sessions are
keyed so driver-held-out folds slice without rebuilding. RAM-safe: one
session at a time via iter_sessions.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from python.calibration.ahrs import Ahrs
from python.ml.dataset import iter_sessions

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "results/audit/denoise_windows.npz"
CACHE_VERSION = 1

WINDOW_S = 5.0   # plan: "sliding window of recent accelerometer data points"
STRIDE_S = 0.5   # inference stride (deployment-shaped)
FS = 100.0


@dataclass
class DenoiseWindows:
    x: np.ndarray       # (N, W) forward accel windows
    target: np.ndarray  # (N,) residual at window end
    driver: np.ndarray  # (N,)
    segment: np.ndarray  # (N,)
    t_end: np.ndarray   # (N,) window-end timestamp (s)

    def __len__(self):
        return self.x.shape[0]


def _wheel_true_forward(sample):
    """Forward acceleration from wheel speed on the common clock (m/s^2).

    The loader already resamples gt_pose onto the smartphone clock with
    km/h -> m/s conversion, so a simple central difference of the speed
    magnitude is the plan's "high-precision wheel encoder data"
    differentiation.
    """
    speed = np.hypot(sample.gt_pose["vx"], sample.gt_pose["vy"])
    dt = 1.0 / FS
    a = np.gradient(speed, dt)
    # ignore spikes from clock/speed glitches: clamp to vehicle-plausible
    # forward acceleration magnitudes
    return np.clip(a, -15.0, 15.0)


def _forward_accel(session, static_s=5.0):
    """Attitude-compensated forward-axis accel for a whole session.

    AHRS is initialised on the first ``static_s`` seconds (plan: "collect
    the first 5 seconds of data"; gravity -> pitch/roll, magnetometer ->
    yaw) and then propagates over the session with accel+mag correction.
    """
    n = session.sample.t.size
    init_n = max(int(round(static_s * FS)), min(n, 5))
    ahrs = Ahrs()
    ahrs.init_static(session.sample.accel[:init_n],
                     session.sample.mag[:init_n],
                     gyro=session.sample.gyro[:init_n])
    return ahrs.run(session.sample.gyro, session.sample.accel,
                    mag=session.sample.mag, dt=1.0 / FS)


def build_windows(session, window_s=WINDOW_S, stride_s=STRIDE_S, fs=FS):
    """One session -> DenoiseWindows (possibly empty if too short)."""
    w = int(round(window_s * fs))
    stride = int(round(stride_s * fs))
    n = session.sample.t.size
    if n < w + 1:
        return DenoiseWindows(np.empty((0, w)), np.empty(0), np.array([]),
                              np.array([]), np.empty(0))
    forward = _forward_accel(session)
    a_true = _wheel_true_forward(session.sample)
    residual = forward - a_true
    xs, targets, drivers, segs, t_ends = [], [], [], [], []
    for lo in range(0, n - w, stride):
        hi = lo + w
        r = residual[hi - 1]  # causal: residual at window end
        if not np.isfinite(r):
            continue
        xs.append(forward[lo:hi].astype(np.float32))
        targets.append(r)
        drivers.append(session.driver)
        segs.append(session.segment)
        t_ends.append(float(session.sample.t[hi - 1]))
    return DenoiseWindows(
        np.asarray(xs, dtype=np.float32), np.asarray(targets),
        np.asarray(drivers), np.asarray(segs), np.asarray(t_ends))


def build_all(data_root, cache_path=CACHE, window_s=WINDOW_S,
              stride_s=STRIDE_S):
    """All sessions -> DenoiseWindows, cached to npz (version-tagged)."""
    cache_path = Path(cache_path)
    if cache_path.is_file():
        data = np.load(cache_path, allow_pickle=True)
        if int(data["version"]) == CACHE_VERSION:
            return DenoiseWindows(data["x"], data["target"],
                                  data["driver"], data["segment"],
                                  data["t_end"])
    sessions = iter_sessions(data_root)
    parts = []
    for i, s in enumerate(sessions, 1):
        part = build_windows(s, window_s, stride_s)
        if len(part):
            parts.append(part)
        del s, part  # release before next load (RAM-safe)
        if i % 10 == 0:
            print(f"  cached {i} sessions, "
                  f"{sum(len(p) for p in parts)} windows", flush=True)
    empty = np.empty(0)
    out = DenoiseWindows(
        (np.concatenate([p.x for p in parts]) if parts
         else np.empty((0, int(window_s * FS)))),
        np.concatenate([p.target for p in parts]) if parts else empty,
        (np.concatenate([p.driver for p in parts]) if parts
         else np.array([], dtype=object)),
        (np.concatenate([p.segment for p in parts]) if parts
         else np.array([], dtype=object)),
        np.concatenate([p.t_end for p in parts]) if parts else empty,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, version=CACHE_VERSION, x=out.x,
                        target=out.target, driver=out.driver,
                        segment=out.segment, t_end=out.t_end)
    return out


def driver_folds(windows):
    """Leave-one-driver-out folds: {test_driver: idx arrays}."""
    drivers = sorted({str(d) for d in windows.driver})
    if len(drivers) < 3:
        raise ValueError("at least three drivers are required")
    folds = {}
    for i, test_driver in enumerate(drivers):
        val_driver = drivers[(i + 1) % len(drivers)]
        d = np.asarray([str(x) for x in windows.driver])
        folds[test_driver] = {
            "train_idx": np.flatnonzero((d != test_driver) & (d != val_driver)),
            "val_idx": np.flatnonzero(d == val_driver),
            "test_idx": np.flatnonzero(d == test_driver),
            "validation_driver": val_driver,
        }
    return folds


def normalize_stats(x, idx):
    """Mean/std of the forward-accel channel over training indices."""
    sub = x[idx].astype(np.float64)
    return float(sub.mean()), float(sub.std(ddof=0) if sub.size > 1 else 1.0)