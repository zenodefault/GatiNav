"""Private simulation core for python/eval/run_evaluation.py.

Runs the four cumulative EKF configurations (raw-INS, +ZUPT, +CNN,
+NHC+matching) over GNSS outage windows and computes the specs/03 section 6
metrics (ATE, RPE, drift; no alignment). GNSS updates only outside the
outage; ground truth never enters the filter. Public entry points live in
run_evaluation.py.
"""

from pathlib import Path

import numpy as np

from python.ekf.ekf import ErrorStateEKF
from python.ekf.ekf_cnn import CNNEKF
from python.io.iovnb_loader import load_pair, resample_to_common_clock
from python.io.outages import carve_outages

FS = 100.0
OUTAGE_DURATIONS_S = (30.0, 60.0, 90.0)
WINDOWS_PER_DURATION = 3
MIN_EXTRA_S = 20.0
ACCEL_VAR_MAX = 0.1
GYRO_VAR_MAX = 1e-4
HEADING_RESIDUAL_DEG = 10.0
_A = 6378137.0
CONFIG_KEYS = ("raw", "zupt", "cnn", "full")


def session_pairs(data_root):
    root = Path(data_root)
    s_files = {p.name[2:].lower(): p for p in root.rglob("S-*.csv")}
    v_files = {p.name[2:].lower(): p for p in root.rglob("V-*.csv")}
    pairs = [(s_files[k], v_files[k]) for k in s_files.keys() & v_files.keys()]
    return sorted(pairs, key=lambda p: p[0].stat().st_size + p[1].stat().st_size)


def find_windows(data_root):
    """First accepted outage window per session, capped per duration."""
    counts = {d: 0 for d in OUTAGE_DURATIONS_S}
    found = []
    for s_path, v_path in session_pairs(data_root):
        if all(counts[d] >= WINDOWS_PER_DURATION for d in OUTAGE_DURATIONS_S):
            break
        try:
            sess = resample_to_common_clock(load_pair(v_path, s_path), fs=FS)
        except Exception:
            continue  # skip pairs the loader cannot align (row-count quirks)
        length = sess.t[-1] - sess.t[0]
        for d in OUTAGE_DURATIONS_S:
            if counts[d] >= WINDOWS_PER_DURATION or length < d + MIN_EXTRA_S:
                continue
            carved = carve_outages(sess, d)
            if carved:
                found.append((s_path.stem, carved[0]))
                counts[d] += 1
    return found


def _gnss_enu(masked):
    """Smartphone GNSS as ENU metres; origin = first valid fix."""
    lat, lon = np.deg2rad(masked.gnss["lat_deg"]), np.deg2rad(masked.gnss["lon_deg"])
    valid = np.isfinite(lat) & np.isfinite(lon) \
        & np.isfinite(masked.gnss["accuracy_m"])
    px = np.full(lat.size, np.nan)
    py, pz, acc = px.copy(), px.copy(), px.copy()
    if valid.any():
        i0 = int(np.argmax(valid))
        px[valid] = (lon[valid] - lon[i0]) * np.cos(lat[i0]) * _A
        py[valid] = (lat[valid] - lat[i0]) * _A
        pz[valid] = masked.gnss["alt_m"][valid] - masked.gnss["alt_m"][i0]
        acc[valid] = masked.gnss["accuracy_m"][valid]
    return px, py, pz, acc, valid


def _seed(masked, px, py, valid, start_t):
    """Position at the first pre-outage fix; velocity from the last 2 s of
    pre-outage fixes (fallback: the last two fixes)."""
    t = masked.gnss["t"]
    before = np.nonzero(valid & (t < start_t))[0]
    if before.size >= 2:
        recent = before[t[before] >= start_t - 2.0]
        if recent.size < 2:
            recent = before[-2:]
        i0, i1 = recent[0], recent[-1]
        dt = t[i1] - t[i0]
        if dt > 0:
            vel = np.array([(px[i1] - px[i0]) / dt, (py[i1] - py[i0]) / dt, 0.0])
            first = before[0]
            return np.array([px[first], py[first], 0.0]), vel
    return np.zeros(3), np.zeros(3)


def _stationary_mask(accel, gyro):
    n = accel.shape[0]
    w = int(round(1.0 * FS))
    mask = np.zeros(n, dtype=bool)
    if n >= w:
        from numpy.lib.stride_tricks import sliding_window_view
        av = sliding_window_view(accel, w, axis=0).var(axis=1).sum(axis=1)
        gv = sliding_window_view(gyro, w, axis=0).var(axis=1).sum(axis=1)
        mask[w - 1:] = (av < ACCEL_VAR_MAX) & (gv < GYRO_VAR_MAX)
    return mask


def _yaw(ekf):
    r = getattr(ekf, "rotation")
    return np.arctan2(r[1, 0], r[0, 0])


def _pass(masked, start_t, end_t, mode, pos0, vel0, model, stationary, gps,
          headings=None):
    """Estimate over the outage interval for one configuration."""
    n = masked.t.size
    dt = float(masked.t[1] - masked.t[0])
    kw = dict(position=pos0, velocity=vel0)
    ekf = CNNEKF(model=model, **kw) if mode in ("cnn", "full") \
        else ErrorStateEKF(**kw)
    px, py, pz, acc, valid = gps
    est, nhc_applied, k = [], 0, 0
    for i in range(n):
        if i > 0:  # state is seeded at t[0]; advance before later samples
            ekf.predict(masked.gyro[i], masked.accel[i], dt)
        t = masked.t[i]
        if valid[i] and (t < start_t or t >= end_t):
            ekf.update_gnss(np.array([px[i], py[i], pz[i]]), max(acc[i], 1e-3))
        in_window = start_t <= t < end_t
        if in_window and stationary[i]:
            if mode == "zupt":
                ekf.update_zupt()
            elif mode in ("cnn", "full"):
                window = np.concatenate((masked.accel[i - 99:i + 1],
                                         masked.gyro[i - 99:i + 1]), axis=1)
                ekf.update_zupt(window)
        if mode == "full" and in_window and headings is not None:
            h = headings[k]
            if np.isfinite(h) and abs(np.angle(np.exp(1j * (_yaw(ekf) - h)))) \
                    <= np.deg2rad(HEADING_RESIDUAL_DEG):
                ekf.update_nhc(h)
                nhc_applied += 1
        if in_window:
            est.append(ekf.position.copy())
            k += 1
    return np.asarray(est), nhc_applied


def _match_headings(matcher, est_xy):
    headings = np.full(est_xy.shape[0], np.nan)
    if matcher is None or est_xy.shape[0] == 0:
        return headings
    try:
        matches = matcher.match(est_xy)
    except Exception:
        return headings
    if matches:
        idx = np.linspace(0, est_xy.shape[0] - 1, len(matches)).astype(int)
        for j, m in enumerate(matches):
            headings[idx[j]] = m.heading
    return headings


def compute_metrics(est, gt):
    """specs/03 section 6: ate/rpe/drift on horizontal positions, no alignment."""
    est = np.asarray(est, dtype=float)[:, :2]
    gt = np.asarray(gt, dtype=float)[:, :2]
    err = np.sqrt(((est - gt) ** 2).sum(axis=1))
    step = np.sqrt(((np.diff(est, axis=0) - np.diff(gt, axis=0)) ** 2).sum(axis=1))
    dist = np.sqrt((np.diff(gt, axis=0) ** 2).sum(axis=1)).sum()
    ate = float(np.sqrt(np.mean(err ** 2)))
    rpe = float(np.sqrt(np.mean(step ** 2))) if step.size else 0.0
    drift = float(err[-1]) if err.size else float("nan")
    rel = 100.0 * drift / dist if dist > 0 else float("nan")
    return {"ate": ate, "rpe": rpe, "drift": drift, "rel": rel}
