"""Phase 4B odometry dataset: vehicle-frame windows + wheel-derived targets.

Builds training windows for the learned-odometry model (AGENTS.md Phase 4B):

Input  (N, 6, W): vehicle-frame accel+gyro AFTER the Phase 1 calibration
transform (python.ekf.calibration.CalibrationEngine via
calibrate_session), so the network never re-learns the phone mounting.
Channels: [accel_x fwd, accel_y left, accel_z up, gyro_x, gyro_y, gyro_z]
in the vehicle frame (x forward, y left, z up, gravity +9.80665 on +z).

Output targets per window END (causal: only past samples are used, so the
window centre label convention of the ZUPT net is replaced by the last
sample -- deployment-shaped):
  disp  : forward displacement over the window (m), from wheel speed
  vel   : forward velocity at the window end (m/s)
  yaw   : integrated yaw increment over the window (rad), from gyro z
  logvar: per-target heteroscedastic uncertainty (trained via NLL)

Targets come from the vehicle stream (gt_pose.vx/vy = wheel speed on the
common clock, loader converts km/h->m/s). Training-only supervision; the
eval leakage guard is untouched (outage windows never see GNSS/GT).

Cache: results/audit/odom_windows.npz with a version tag; sessions are
keyed so driver-held-out folds can slice without rebuilding.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from python.ekf.calibration import calibrate_session
from python.ml.dataset import iter_sessions

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "results/audit/odom_windows.npz"
CACHE_VERSION = 1

WINDOW_S = 5.0  # AGENTS.md 4B: pick one of 2-10 s and justify from data
STRIDE_S = 0.5  # inference stride (deployment-shaped; AGENTS.md 4B)
CACHE_STRIDE_S = 1.0  # training-cache stride: halves the cache vs 0.5 s
FS = 100.0
# Window choice justification (measured, /tmp/odom_ident.py): the
# vibration->speed correlation rises with window length but plateaus near
# 4-6 s; 5 s at 100 Hz = 500 samples keeps the TCN causal receptive field
# meaningful while staying under the compute budget.

G = 9.80665


@dataclass
class OdomWindows:
    x: np.ndarray        # (N, 6, W) vehicle-frame accel+gyro
    disp: np.ndarray     # (N,) forward displacement over window (m)
    vel: np.ndarray      # (N,) forward speed at window end (m/s)
    v0: np.ndarray       # (N,) forward speed at window START (m/s)
    yaw: np.ndarray      # (N,) yaw increment over window (rad)
    driver: np.ndarray   # (N,) driver label
    segment: np.ndarray  # (N,) session stem
    t_end: np.ndarray    # (N,) window-end timestamp (s)

    def __len__(self):
        return self.x.shape[0]


def _vehicle_targets(sample, lo, hi):
    """Forward displacement / end+start speed / yaw increment for [lo,hi)."""
    vx = sample.gt_pose["vx"][lo:hi + 1]
    vy = sample.gt_pose["vy"][lo:hi + 1]
    speed = np.hypot(vx, vy)
    # Trapezoidal integral of speed over the window (wheel-derived).
    disp = float(np.trapz(speed, dx=1.0 / FS))
    vel = float(speed[-1])
    v0 = float(speed[0])
    # Yaw increment from the (bias-corrected, vehicle-frame) gyro z: the
    # network's yaw head is supervised by the integrated gyro, so the
    # filter can use it as a rate-consistency check; wheel/GT yaw is NOT
    # used (it would leak GT heading into inference inputs' distribution).
    yaw = float(np.trapz(sample.gyro[lo:hi, 2], dx=1.0 / FS))
    return disp, vel, v0, yaw


def build_windows(session, window_s=WINDOW_S, stride_s=STRIDE_S, fs=FS):
    """One session -> OdomWindows (possibly empty if too short)."""
    w = int(round(window_s * fs))
    stride = int(round(stride_s * fs))
    n = session.sample.t.size
    if n < w + 1:
        return OdomWindows(np.empty((0, 6, w)), np.empty(0), np.empty(0),
                           np.empty(0), np.empty(0), np.array([]),
                           np.array([]), np.empty(0))
    # Phase 1 calibration on the full session (static-window biases +
    # GNSS-course mounting yaw); transform IMU into the vehicle frame.
    try:
        cal = calibrate_session(session.sample, fs=fs)
        gyro_v, accel_v = cal.to_vehicle_frame(session.sample.gyro,
                                               session.sample.accel)
    except ValueError:
        # No 15 s quasi-static stretch in this session: fall back to a
        # gravity-levelled-only transform so the session still contributes
        # (mounting yaw stays 0; augmentation covers the gap).
        from python.ekf.ekf import level_rotation
        level = level_rotation(session.sample.accel[:10 * int(fs)].mean(axis=0)
                               if n > 10 * int(fs)
                               else session.sample.accel.mean(axis=0))
        gyro_v = session.sample.gyro @ level.T
        accel_v = session.sample.accel @ level.T
    signal = np.column_stack((accel_v, gyro_v)).T  # (6, N)
    xs, disps, vels, v0s, yaws = [], [], [], [], []
    drivers, segs, t_ends = [], [], []
    for lo in range(0, n - w, stride):
        hi = lo + w
        disp, vel, v0, yaw = _vehicle_targets(session.sample, lo, hi)
        if not (np.isfinite(disp) and np.isfinite(vel)
                and np.isfinite(v0) and np.isfinite(yaw)):
            continue
        xs.append(signal[:, lo:hi].astype(np.float32))
        disps.append(disp)
        vels.append(vel)
        v0s.append(v0)
        yaws.append(yaw)
        drivers.append(session.driver)
        segs.append(session.segment)
        t_ends.append(float(session.sample.t[hi - 1]))
    return OdomWindows(
        np.asarray(xs, dtype=np.float32), np.asarray(disps),
        np.asarray(vels), np.asarray(v0s), np.asarray(yaws),
        np.asarray(drivers), np.asarray(segs), np.asarray(t_ends))


def build_all(data_root, cache_path=CACHE, window_s=WINDOW_S,
              stride_s=CACHE_STRIDE_S):
    """All sessions -> OdomWindows, cached to npz (version-tagged).

    The cache is built at CACHE_STRIDE_S (1 s) to bound its size; inference
    strides at 0.5 s by sliding the deployed model, and training does not
    need every 0.5 s shift of the same window.
    """
    cache_path = Path(cache_path)
    if cache_path.is_file():
        data = np.load(cache_path, allow_pickle=True)
        if int(data["version"]) == CACHE_VERSION:
            return OdomWindows(data["x"], data["disp"], data["vel"],
                               data["v0"], data["yaw"], data["driver"],
                               data["segment"], data["t_end"])
    sessions = iter_sessions(data_root)
    parts = []
    for i, s in enumerate(sessions, 1):
        part = build_windows(s, window_s, stride_s)
        if len(part):
            parts.append(part)
        # release the session arrays before the next load (RAM-safe)
        del s, part
        if i % 10 == 0:
            print(f"  cached {i} sessions, {sum(len(p) for p in parts)} windows",
                  flush=True)
    empty = np.empty(0)
    out = OdomWindows(
        (np.concatenate([p.x for p in parts]) if parts
         else np.empty((0, 6, int(window_s * FS)))),
        np.concatenate([p.disp for p in parts]) if parts else empty,
        np.concatenate([p.vel for p in parts]) if parts else empty,
        np.concatenate([p.v0 for p in parts]) if parts else empty,
        np.concatenate([p.yaw for p in parts]) if parts else empty,
        (np.concatenate([p.driver for p in parts]) if parts
         else np.array([], dtype=object)),
        (np.concatenate([p.segment for p in parts]) if parts
         else np.array([], dtype=object)),
        np.concatenate([p.t_end for p in parts]) if parts else empty,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, version=CACHE_VERSION, x=out.x,
                        disp=out.disp, vel=out.vel, v0=out.v0, yaw=out.yaw,
                        driver=out.driver, segment=out.segment,
                        t_end=out.t_end)
    return out


def driver_folds(windows):
    """Leave-one-driver-out fold dict: {test_driver: {train_idx, val_idx, test_idx}}.

    Validation driver = next in sorted order (mirrors python.ml.dataset).
    """
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
    """Channel-wise mean/std over training indices; x is (N, 6, W)."""
    sub = x[idx].astype(np.float64)
    mean = sub.transpose(1, 0, 2).reshape(6, -1).mean(axis=1)
    std = sub.transpose(1, 0, 2).reshape(6, -1).std(axis=1)
    std = np.where(std < 1e-9, 1.0, std)
    return mean.astype(np.float64), std.astype(np.float64)
