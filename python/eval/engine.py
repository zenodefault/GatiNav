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
from python.io.iovnb_loader import load_pair, resample_to_common_clock
from python.io.outages import OutageWindow, _mask_gnss,\
    find_outage_candidates

FS = 100.0
OUTAGE_DURATIONS_S = (30.0, 60.0, 90.0)
WINDOWS_PER_DURATION = 3
MIN_EXTRA_S = 20.0
ACCEL_VAR_MAX = 1.45
GYRO_VAR_MAX = 0.006
HEADING_RESIDUAL_DEG = 10.0
_A = 6378137.0
CONFIG_KEYS = ("raw", "zupt", "cnn", "full")
# The outage is entered mid-journey: require this much pre-outage session
# time with valid GNSS so biases can converge before the outage (specs/01
# §6.1 is a 5 s data margin; the benchmark scenario needs real warm-up).
MIN_PREOUTAGE_S = 60.0
# The filter is initialised this long before the outage (leveling + GNSS
# fusion), bounding evaluation cost on long sessions.
PRE_RUN_S = 60.0
LEVEL_WINDOW_S = 2.0
GNSS_VEL_WINDOW_S = 2.0  # central-difference window for GNSS velocity
GNSS_VEL_ACCURACY_MPS = 4.0  # 1-sigma per-axis velocity noise (VERIFY)
ZUPT_STRIDE_S = 0.5  # ZUPT/CNN pseudo-measurement stride (specs/03 §3)


def session_pairs(data_root):
    root = Path(data_root)
    s_files = {p.name[2:].lower(): p for p in root.rglob("S-*.csv")}
    v_files = {p.name[2:].lower(): p for p in root.rglob("V-*.csv")}
    pairs = [(s_files[k], v_files[k]) for k in s_files.keys() & v_files.keys()]
    return sorted(pairs, key=lambda p: p[0].stat().st_size + p[1].stat().st_size)


def find_windows(data_root, min_preoutage_s=MIN_PREOUTAGE_S):
    """First qualifying outage window per session, capped per duration.

    A window qualifies when at least ``min_preoutage_s`` of session time
    precedes it and the pre-outage interval carries valid GNSS on at least
    half its samples, so the filter can converge biases before the outage.
    """
    counts = {d: 0 for d in OUTAGE_DURATIONS_S}
    found = []
    for s_path, v_path in session_pairs(data_root):
        if all(counts[d] >= WINDOWS_PER_DURATION for d in OUTAGE_DURATIONS_S):
            break
        try:
            sess = resample_to_common_clock(load_pair(v_path, s_path), fs=FS)
        except Exception:
            continue  # skip pairs the loader cannot align
        length = sess.t[-1] - sess.t[0]
        for d in OUTAGE_DURATIONS_S:
            if counts[d] >= WINDOWS_PER_DURATION or length < d + MIN_EXTRA_S:
                continue
            # Copy-free candidate scan; mask only the chosen window (masking
            # every candidate of a 3.5 h session would copy gigabytes).
            pending_rejected = []
            for start_t, end_t, reasons in find_outage_candidates(sess, d):
                if reasons:
                    pending_rejected.append((start_t, "; ".join(reasons)))
                    continue
                pre = (sess.t < start_t)
                if sess.t[pre].size < 1:
                    continue
                if start_t - sess.t[0] < min_preoutage_s:
                    continue
                gnss_ok = np.isfinite(sess.gnss["lat_deg"][pre])
                if gnss_ok.sum() < 0.5 * pre.sum():
                    pending_rejected.append(
                        (start_t, "insufficient pre-outage GNSS"))
                    continue
                i0 = int(np.searchsorted(sess.t, start_t))
                i1 = int(np.searchsorted(sess.t, end_t))
                window = OutageWindow(
                    start_t=start_t, end_t=end_t, duration_s=d,
                    masked=_mask_gnss(sess, i0, i1),
                    rejected=list(pending_rejected))
                found.append((s_path.stem, window))
                counts[d] += 1
                break
    return found


def _gnss_enu(masked):
    """Smartphone GNSS as ENU metres; origin = first valid fix."""
    lat, lon = np.deg2rad(masked.gnss["lat_deg"]), np.deg2rad(masked.gnss["lon_deg"])
    valid = (np.isfinite(lat) & np.isfinite(lon)
             & np.isfinite(masked.gnss["alt_m"])
             & np.isfinite(masked.gnss["accuracy_m"]))
    px = np.full(lat.size, np.nan)
    py, pz, acc = px.copy(), px.copy(), px.copy()
    if valid.any():
        i0 = int(np.argmax(valid))
        px[valid] = (lon[valid] - lon[i0]) * np.cos(lat[i0]) * _A
        py[valid] = (lat[valid] - lat[i0]) * _A
        pz[valid] = masked.gnss["alt_m"][valid] - masked.gnss["alt_m"][i0]
        acc[valid] = masked.gnss["accuracy_m"][valid]
    return px, py, pz, acc, valid


def _init_state(masked, px, py, pz, valid, run_start_t):
    """(position, velocity, rotation) at the run start (ENU, m, m/s).

    Position comes from the first valid GNSS fix at/after run_start_t;
    velocity from two fixes a couple of seconds later; rotation from
    accelerometer gravity leveling over the first LEVEL_WINDOW_S seconds.
    """
    t = masked.t
    start_idx = int(np.searchsorted(t, run_start_t))
    fixes = np.nonzero(valid)[0]
    fixes = fixes[fixes >= start_idx]
    if fixes.size == 0:
        return np.zeros(3), np.zeros(3), np.eye(3)
    i0 = fixes[0]
    position = np.array([px[i0], py[i0], pz[i0]])
    vel = np.zeros(3)
    later = fixes[(t[fixes] >= t[i0] + 1.0) & (t[fixes] <= t[i0] + 5.0)]
    if later.size >= 1:
        i1 = later[0]
        dt = t[i1] - t[i0]
        if dt > 0:
            vel = np.array([(px[i1] - px[i0]) / dt, (py[i1] - py[i0]) / dt,
                            (pz[i1] - pz[i0]) / dt])
    accel = masked.accel
    end_idx = int(np.searchsorted(t, run_start_t + LEVEL_WINDOW_S))
    rotation = np.eye(3)
    if end_idx > start_idx:
        from python.ekf.ekf import level_rotation
        rotation = level_rotation(accel[start_idx:end_idx].mean(axis=0))
    return position, vel, rotation


def _stationarity_scores(accel, gyro):
    """Return temporal variance scores; the first 99 samples are warm-up."""
    n = accel.shape[0]
    w = int(round(1.0 * FS))
    av = np.zeros(n)
    gv = np.zeros(n)
    if n < w:
        return av, gv
    from numpy.lib.stride_tricks import sliding_window_view
    av[w - 1:] = sliding_window_view(
        accel, w, axis=0).var(axis=2).sum(axis=1)
    gv[w - 1:] = sliding_window_view(
        gyro, w, axis=0).var(axis=2).sum(axis=1)
    return av, gv


LOW_SPEED_MPS = 0.5  # true stops vs slow rolling (Blueprint Phase 2.1)


def _stationary_mask(accel, gyro, speed=None, low_speed_mps=LOW_SPEED_MPS):
    """Variance-based stationarity, optionally gated by a low-speed bound.

    ``speed`` (m/s, per sample) is the Phase 2.1 strengthening: a stop is
    only trusted when the signal variance is low AND the derived speed is
    below ``low_speed_mps``, rejecting slow rolling movement that the
    variance test alone would label stationary. When ``speed`` is None
    the detector is the plain variance mask (backward compatible).
    """
    av, gv = _stationarity_scores(accel, gyro)
    mask = np.zeros(accel.shape[0], dtype=bool)
    if accel.shape[0] >= int(round(FS)):
        mask[99:] = (av[99:] < ACCEL_VAR_MAX) & (gv[99:] < GYRO_VAR_MAX)
    if speed is not None:
        speed = np.asarray(speed, dtype=np.float64)
        if speed.shape != (accel.shape[0],):
            raise ValueError("speed must match accel length")
        mask &= (speed < low_speed_mps)
    return mask


def _yaw(ekf):
    r = getattr(ekf, "rotation")
    return np.arctan2(r[1, 0], r[0, 0])


def _gnss_velocity(px, py, pz, valid, i, win):
    """Central-difference ENU velocity at sample i over +/- win samples."""
    if i - win < 0 or i + win >= px.size:
        return None
    if not (valid[i - win] and valid[i + win]):
        return None
    dt = 2.0 * win
    return np.array([(px[i + win] - px[i - win]) / dt,
                     (py[i + win] - py[i - win]) / dt,
                     (pz[i + win] - pz[i - win]) / dt])


def _pass(masked, start_t, end_t, mode, pos0, vel0, model, stationary, gps,
          headings=None, rotation=None, run_start_t=None, scenario=None,
          nhc_lateral_sigma=5.0):
    """Estimate over [run_start_t, end_t) for one configuration.

    The filter is initialised at run_start_t (default start_t - PRE_RUN_S)
    with attitude leveling and fused with GNSS position+velocity and ZUPT
    while GNSS is available; the outage interval itself is inertial-only
    (plus ZUPT/NHC per mode). Ground truth never enters the filter.

    ``scenario`` (optional) is the per-sample structured-scenario mask
    from python.matching.scenario (Blueprint plan 3.3): in tunnel/highway
    stretches a trusted match engages the position-domain NHC road update
    with ``nhc_lateral_sigma``; elsewhere the plain velocity NHC is used.
    """
    t = masked.t
    dt = float(t[1] - t[0])
    if run_start_t is None:
        run_start_t = max(t[0], start_t - PRE_RUN_S)
    i_start = int(np.searchsorted(t, run_start_t))
    i_end = int(np.searchsorted(t, end_t))
    kw = dict(position=pos0, velocity=vel0, rotation=rotation)
    if mode in ("cnn", "full"):
        from python.ekf.ekf_cnn import CNNEKF
        ekf = CNNEKF(model=model, **kw)
    else:
        ekf = ErrorStateEKF(**kw)
    px, py, pz, acc, valid = gps
    vel_win = int(round(GNSS_VEL_WINDOW_S / dt))
    zupt_stride = max(1, int(round(ZUPT_STRIDE_S / dt)))
    est, nhc_applied, k, last_zupt_i = [], 0, 0, -10**9
    for i in range(i_start, i_end):
        if i > i_start:  # state is seeded at t[i_start]
            ekf.predict(masked.gyro[i], masked.accel[i], dt)
        tt = t[i]
        pre_outage = tt < start_t
        if valid[i] and pre_outage:
            ekf.update_gnss(np.array([px[i], py[i], pz[i]]), max(acc[i], 1e-3))
            if i % vel_win == 0:
                vel = _gnss_velocity(px, py, pz, valid, i, vel_win)
                if vel is not None:
                    ekf.update_gnss_velocity(vel, GNSS_VEL_ACCURACY_MPS)
        # ZUPT fires on stationary samples at the specs/03 stride (0.5 s),
        # not at 100 Hz: the CNN forward pass at every sample costs seconds
        # per window and the zero-velocity pseudo-measurement is redundant
        # at IMU rate. First stationary sample of a stop always fires.
        if stationary[i] and i - last_zupt_i >= zupt_stride:
            if mode == "zupt":
                ekf.update_zupt()
            elif mode in ("cnn", "full"):
                window = np.concatenate((masked.accel[i - 99:i + 1],
                                         masked.gyro[i - 99:i + 1]), axis=1)
                ekf.update_zupt(window)
            last_zupt_i = i
        in_window = not pre_outage
        if mode == "full" and in_window and headings is not None:
            h = headings[k]
            if np.isfinite(h) and abs(np.angle(np.exp(1j * (_yaw(ekf) - h)))) \
                    <= np.deg2rad(HEADING_RESIDUAL_DEG):
                structured = (scenario is not None and scenario[k])
                if structured:
                    # plan 3.3: aggressive NHC in tunnels/highways --
                    # position-domain road constraint, larger gain
                    ekf.update_nhc_road(h, lateral_sigma=nhc_lateral_sigma,
                                        confidence=0.9)
                else:
                    ekf.update_nhc(h)
                nhc_applied += 1
        if in_window:
            est.append(ekf.position.copy())
            k += 1
    return np.asarray(est), nhc_applied


def _match_headings(matcher, est_xy, gate_residual_m=None):
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
            if gate_residual_m is not None and \
                    getattr(m, "residual", float("inf")) > gate_residual_m:
                continue  # estimate is off the matched road: distrust it
            headings[idx[j]] = m.heading
        valid = np.isfinite(headings)
        # only interpolate across short internal gaps of a trusted match;
        # a gated (off-road) stretch must NOT be filled with road headings
        if gate_residual_m is None and valid.sum() > 1 and not np.all(valid):
            samples = np.flatnonzero(valid)
            unwrapped = np.unwrap(headings[valid])
            headings[~valid] = np.interp(
                np.flatnonzero(~valid), samples, unwrapped
            )
            headings = np.angle(np.exp(1j * headings))
    return headings


def compute_metrics(est, gt):
    """specs/03 section 6: ate/rpe/drift on horizontal positions, no alignment."""
    est = np.asarray(est, dtype=float)[:, :2]
    gt = np.asarray(gt, dtype=float)[:, :2]
    err = np.sqrt(((est - gt) ** 2).sum(axis=1))
    step = np.sqrt(((np.diff(est, axis=0) - np.diff(gt, axis=0)) ** 2).sum(axis=1))
    dist = float(np.sqrt((np.diff(gt, axis=0) ** 2).sum(axis=1)).sum())
    ate = float(np.sqrt(np.mean(err ** 2)))
    rpe = float(np.sqrt(np.mean(step ** 2))) if step.size else 0.0
    drift = float(err[-1]) if err.size else float("nan")
    finite = np.isfinite(dist) and dist > 0 and np.isfinite(drift)
    rel = 100.0 * drift / dist if finite else float("nan")
    return {"ate": ate, "rpe": rpe, "drift": drift, "rel": rel,
            "dist": dist if np.isfinite(dist) else 0.0}
