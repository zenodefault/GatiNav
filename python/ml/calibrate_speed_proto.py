"""Prototype: within-session, GNSS-supervised speed calibration.

Concept (deployment): during GNSS-available driving the phone learns the
*current session's* vibration -> speed map from IMU windows paired with
GNSS-derived speed, then extrapolates that map into the outage. This attacks
the cross-session confound (phone mount, vehicle, idle vibration) that made
the cross-session CNN speed head fail (fractional RMSE ~77%, corr 0.05).

Stage A (this script): feasibility with CLEAN speed labels (gt_pose, the
vehicle's own speed). Calibrate a per-session ridge regression on log-log
vibration features using only the first ~60-90 s of the session, then
evaluate time-held-out on the remainder. Reports the per-session fractional
speed RMSE distribution vs the 76.9% cross-session baseline.

Stage B (next): repeat with GNSS-differentiated velocity labels (noisier,
but the only signal a phone has) to measure calibration survival under label
noise. Engine wiring (update_speed) comes only after a positive Stage A/B.
"""

import re
import sys
from pathlib import Path

import numpy as np

from python.io.iovnb_loader import load_pair, resample_to_common_clock
from python.ml.dataset import _vehicle_pair

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data/IO-VNBD/Synchronised V abd S datasets"
CAL_MIN_S = 60.0      # minimal calibration horizon
CAL_TARGET_S = 90.0   # extend until both stop+drive anchors are present
CAL_MAX_S = 300.0     # hard cap on calibration horizon
V_STOP = 0.5          # m/s anchor: below = stopped
V_DRIVE = 5.0         # m/s anchor: above = driving
WINDOW_S = 1.0
STRIDE_S = 0.5
FOLD = 1e-3           # ridge regularisation on standardised log-features


def window_features(signal):
    """Log-vibration features per 1 s window of (N,6) Acc-Gyro signal.

    Returns (n_windows, n_features). Each accel axis is detrended (DC removal
    = removing gravity, which is constant within a window's mean anyway), then
    log(1 + var) per axis plus accel-magnitude variance.
    """
    n = int(round(WINDOW_S * 100.0))
    stride = int(round(STRIDE_S * 100.0))
    accel = signal[:, :3]
    gyro = signal[:, 3:]
    mag = np.linalg.norm(accel - accel.mean(axis=0), axis=1)
    rows = []
    for lo in range(0, signal.shape[0] - n + 1, stride):
        a = accel[lo:lo + n] - accel[lo:lo + n].mean(axis=0)
        g = gyro[lo:lo + n]
        m = mag[lo:lo + n]
        feats = np.concatenate((
            np.log1p(a.var(axis=0)), np.log1p(g.var(axis=0)),
            [np.log1p(m.var())]))
        rows.append(feats)
    return np.asarray(rows, dtype=np.float64)


def session_speed_profile(sample):
    """GT speed (m/s) at each window centre + window feature rows."""
    t = sample.t
    t_gt = sample.gt_pose["t"]
    v = np.hypot(sample.gt_pose["vx"], sample.gt_pose["vy"])
    n = int(round(WINDOW_S * 100.0))
    stride = int(round(STRIDE_S * 100.0))
    centres = t[n // 2::stride][: (len(t) - n) // stride + 1]
    vs = np.interp(centres, t_gt, v)
    return centres, vs


def ridge_fit(Xc, yc, Xt):
    """Ridge regression y ~ X on standardised features, predict on Xt."""
    mean = Xc.mean(axis=0)
    std = Xc.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    Xcs = (Xc - mean) / std
    Xts = (Xt - mean) / std
    A = Xcs.T @ Xcs + FOLD * np.eye(Xcs.shape[1])
    beta = np.linalg.solve(A, Xcs.T @ yc)
    b0 = yc.mean() - beta @ Xcs.mean(axis=0)
    pred = Xts @ beta + b0
    # clip to plausible log(1+v) before exponentiation (v in m/s <= ~60)
    return np.clip(pred, 0.0, np.log1p(60.0))


def frac_rmse(pred, v):
    if len(v) == 0 or np.mean(v) < 1e-6:
        return None
    rmse = float(np.sqrt(np.mean((pred - v) ** 2)))
    return rmse / float(np.mean(v))


def rolling_outage_eval(outage_s=60.0, cal_s=120.0, gap_s=30.0):
    """Deployment-shaped test: calibrate on the GNSS-available minutes
    immediately before a candidate outage, test on the outage slice itself.

    This mirrors the seamless-deficit use case: the outage is short and
    recent, so the road/surface character of the calibration window matches
    the outage road far better than session-start calibration does.
    """
    n_win = int(round(WINDOW_S * 100.0))
    stride = int(round(STRIDE_S * 100.0))
    rows = []
    n_samples = n_skip = 0
    for s_path in sorted(DATA_ROOT.rglob("S-*.csv")):
        v_path = _vehicle_pair(s_path)
        if v_path is None or "Uncategorised" in str(s_path):
            continue
        m = re.search(r"\((Driver [^)]+)\)", str(s_path))
        driver = m.group(1) if m else "unknown"
        try:
            sample = resample_to_common_clock(load_pair(v_path, s_path))
        except Exception:
            continue
        signal = np.column_stack((sample.accel, sample.gyro))
        feats = window_features(signal)
        t = sample.t
        t_gt = sample.gt_pose["t"]
        v = np.hypot(sample.gt_pose["vx"], sample.gt_pose["vy"])
        k = min(len(feats), (len(t) - n_win) // stride + 1)
        centres = t[n_win // 2::stride][:k]
        vs = np.interp(centres, t_gt, v)[:k]
        cal_n = int(cal_s / STRIDE_S)
        out_n = int(outage_s / STRIDE_S)
        gap_n = int(gap_s / STRIDE_S)
        lo = cal_n + gap_n
        for start in range(lo, k - out_n, int(10.0 / STRIDE_S)):
            cb = start - cal_n - gap_n
            ce = start - gap_n
            cal = vs[cb:ce]
            if not ((cal < V_STOP).any() and (cal > V_DRIVE).any()):
                continue
            yc = np.log1p(cal)
            pred = np.expm1(ridge_fit(feats[cb:ce], yc, feats[start:start + out_n]))
            vout = vs[start:start + out_n]
            drive = vout > 1.0
            if not drive.any():
                continue
            n_samples += 1
            fd = frac_rmse(pred[drive], vout[drive])
            if fd is None:
                n_skip += 1
                continue
            rows.append((driver, s_path.name, fd))
            if not np.isfinite(fd) or fd > 5.0:
                n_skip += 1
    if not rows:
        print("no rolling outage samples")
        return
    fds = np.array([r[2] for r in rows])
    print(f"rolling-outage samples: {len(rows)} (skipped {n_skip})")
    print(f"frac RMSE (driving-only): mean = {fds.mean() * 100:.1f}%  "
          f"median = {np.median(fds) * 100:.1f}%  "
          f"p25 = {np.percentile(fds, 25) * 100:.1f}%  "
          f"p75 = {np.percentile(fds, 75) * 100:.1f}%")
    good = fds < 0.5
    print(f"samples under 50%: {good.sum()}/{len(fds)}"
          f" ({good.mean() * 100:.0f}%)")
    print(f"samples under 25%: {(fds < 0.25).sum()}/{len(fds)}"
          f" ({(fds < 0.25).mean() * 100:.0f}%)")
    return fds


def main(limit=None):
    rows_all, skipped = [], 0
    for s_path in sorted(DATA_ROOT.rglob("S-*.csv")):
        v_path = _vehicle_pair(s_path)
        if v_path is None:
            continue
        if "Uncategorised" in str(s_path):
            continue  # duplicate stems without driver labels
        m = re.search(r"\((Driver [^)]+)\)", str(s_path))
        driver = m.group(1) if m else "unknown"
        try:
            sample = resample_to_common_clock(load_pair(v_path, s_path))
        except Exception as exc:  # corrupted session
            print(f"  skip {s_path.name}: {exc}")
            skipped += 1
            continue
        signal = np.column_stack((sample.accel, sample.gyro))
        feats = window_features(signal)
        centres, v = session_speed_profile(sample)
        k = min(len(feats), len(v))
        feats, v = feats[:k], v[:k]
        # calibration horizon: from session start until both anchors present
        cal_n = int(CAL_MIN_S / STRIDE_S)
        stop_n = min(int(CAL_MAX_S / STRIDE_S), len(v))
        cal_end = None
        for end in range(cal_n, stop_n + 1):
            seg = v[:end]
            if (seg < V_STOP).any() and (seg > V_DRIVE).any():
                cal_end = end
                break
        if cal_end is None or cal_end >= len(v) - 10:
            print(f"  skip {s_path.name} ({driver}): no stop+drive anchors "
                  f"in first {CAL_MAX_S:.0f}s or session too short")
            skipped += 1
            continue
        # time-held-out test: everything AFTER calibration
        yc = np.log1p(v[:cal_end])
        pred_log = ridge_fit(feats[:cal_end], yc, feats[cal_end:])
        pred = np.expm1(pred_log)
        vtest = v[cal_end:]
        fa = frac_rmse(pred, vtest)
        drive = vtest > 1.0
        fd = frac_rmse(pred[drive], vtest[drive]) if drive.any() else None
        rows_all.append((driver, s_path.name, cal_end * STRIDE_S, len(v),
                         fa, fd))
        if limit is not None and len(rows_all) >= limit:
            break
    if not rows_all:
        print("no sessions calibrated")
        return
    print(f"\ncalibrated {len(rows_all)} sessions (skipped {skipped})")
    print(f"{'session':<14} {'driver':<10} {'cal_s':>6} {'test_s':>7} "
          f"{'frac_all':>9} {'frac_drive':>11}")
    for driver, name, cal_s, n, fa, fd in rows_all:
        print(f"{name:<14} {driver:<10} {cal_s:>6.0f} {n * STRIDE_S:>7.0f} "
              f"{fa * 100:>8.1f}% "
              f"{(fd * 100 if fd is not None else float('nan')):>10.1f}%")
    fa_all = np.array([r[4] for r in rows_all])
    fd_all = np.array([r[5] for r in rows_all if r[5] is not None])
    w = np.array([r[3] for r in rows_all], dtype=np.float64)
    print(f"\noverall frac RMSE (session-weighted): all = "
          f"{np.average(fa_all, weights=w) * 100:.1f}%  driving-only = "
          f"{np.average(fd_all, weights=w) * 100:.1f}%")
    print(f"per-session median (all) = {np.median(fa_all) * 100:.1f}%  "
          f"(driving) = {np.median(fd_all) * 100:.1f}%")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "rolling":
        rolling_outage_eval()
    else:
        limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
        main(limit=limit)