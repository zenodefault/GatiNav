"""In-vehicle alignment and IMU calibration (Phase 1).

CalibrationEngine turns raw phone-frame IMU into vehicle-frame IMU:

1. ``calibrate_stationary`` — on 10-20 s of detected stationary time,
   estimates the gyro bias (mean gyro) and the accelerometer bias (mean
   specific force minus gravity in the gravity-leveled frame), and the
   leveling rotation (phone pitch/roll from the gravity direction).
2. ``calibrate_yaw`` — during straight GNSS-valid driving, estimates the
   mounting yaw offset between the leveled phone frame and the vehicle
   course, from GNSS fix-to-fix course vs the bias-corrected integrated
   gyro yaw. No compass, no ground truth.
3. ``movement_flags`` — detects mount movement (sudden gravity-direction
   or heading change) from low-passed IMU; engine-idle vibration is
   high-frequency and must NOT trigger it.
4. ``to_vehicle_frame`` — transforms every IMU sample: subtract biases,
   rotate leveling, rotate mounting yaw. Vehicle frame: x forward, y
   left, z up (gravity +9.80665 on +z at rest).

GNSS course and velocity are derived from consecutive fixes (ENU
displacement), so no external speed feed and no convention assumptions.
Ground truth is never read by this module.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, sosfiltfilt

from python.ekf.ekf import level_rotation

G = 9.80665
_A = 6378137.0
_EARTH_CUTOFF_HZ = 0.5  # low-pass separating gravity direction from vibration


def _rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _circular_mean(angles):
    angles = np.asarray(angles, dtype=np.float64)
    return float(np.arctan2(np.mean(np.sin(angles)),
                            np.mean(np.cos(angles))))


def _wrap(angle):
    return (np.asarray(angle, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


class CalibrationEngine:
    """Estimate and apply the phone-to-vehicle alignment (Phase 1)."""

    def __init__(self):
        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)
        self.rotation = np.eye(3)  # phone -> vehicle, set by calibration
        self.level = np.eye(3)     # phone -> gravity-leveled phone frame
        self.yaw_mount = 0.0       # leveled-phone vs vehicle course (rad)
        self.calibrated = False

    # -- 1. stationary calibration ----------------------------------------
    def calibrate_stationary(self, gyro, accel):
        """Estimate biases + leveling from a stationary IMU stretch.

        ``gyro``/``accel`` are (N, 3) phone-frame samples covering at least
        10-20 s of stationary time. Returns self for chaining.
        """
        gyro = np.asarray(gyro, dtype=np.float64)
        accel = np.asarray(accel, dtype=np.float64)
        if gyro.shape != accel.shape or gyro.ndim != 2 or gyro.shape[1] != 3:
            raise ValueError("gyro and accel must both have shape (N, 3)")
        if gyro.shape[0] < 10:
            raise ValueError("need >= 10 stationary samples to calibrate")
        self.gyro_bias = np.mean(gyro, axis=0)
        f_mean = np.mean(accel, axis=0)
        self.level = level_rotation(f_mean)
        # Vertical accel-bias component is indistinguishable from a gravity
        # magnitude error; the horizontal part is the observable bias.
        self.accel_bias = f_mean - self.level.T @ np.array([0.0, 0.0, G])
        self.yaw_mount = 0.0
        self.rotation = self.level
        self.calibrated = True
        return self

    # -- 2. mounting yaw from GNSS course ---------------------------------
    def calibrate_yaw(self, t, gyro, gnss, speed_min=2.0,
                      course_window_s=1.0, max_course_rate_deg=10.0):
        """Estimate the phone-vs-vehicle yaw offset from GNSS course.

        Uses only GNSS fixes (ENU displacement course) and the bias-
        corrected integrated gyro yaw. Straight driving = course rate
        below ``max_course_rate_deg`` per second.
        """
        if not self.calibrated:
            raise RuntimeError("call calibrate_stationary first")
        omega = (gyro - self.gyro_bias) @ self.level.T
        yaw_phone = np.concatenate((
            [0.0], np.cumsum(0.5 * (omega[1:, 2] + omega[:-1, 2])
                             * np.diff(t))))
        east, north, speed, valid, fix_idx = _gnss_tracks(gnss, t)
        if fix_idx.size < 3:
            return self  # not enough fixes: keep yaw_mount = 0
        t_fix = t[fix_idx]
        dt_fix = np.diff(t_fix)
        dt_fix[dt_fix == 0.0] = np.inf
        step = np.hypot(np.diff(east[fix_idx]), np.diff(north[fix_idx]))
        fix_speed = step / dt_fix
        course = np.arctan2(np.diff(north[fix_idx]), np.diff(east[fix_idx]))
        ok = (fix_speed >= speed_min) & (fix_speed <= 50.0) \
            & np.isfinite(course)
        # straight segments only: course change over course_window_s small
        max_rate = np.deg2rad(max_course_rate_deg)
        ok &= np.abs(_wrap(np.diff(course, prepend=course[0]))) \
            <= max_rate * dt_fix
        if ok.sum() < 2:
            return self  # no usable driving course: keep yaw_mount = 0
        mid_t = 0.5 * (t_fix[1:] + t_fix[:-1])[ok]
        yaw_at = np.interp(mid_t, t, yaw_phone)
        self.yaw_mount = _circular_mean(course[ok] - yaw_at)
        self.rotation = _rot_z(self.yaw_mount) @ self.level
        return self

    # -- 3. mount-movement detection --------------------------------------
    def movement_flags(self, accel, gyro, fs, angle_deg=5.0,
                       gyro_thresh=0.15):
        """Per-sample mount-movement flags from low-passed IMU.

        Idle vibration is high-frequency: the 0.5 Hz low-pass removes it,
        so engine idling does not trigger the detector.
        """
        accel = np.asarray(accel, dtype=np.float64)
        gyro = np.asarray(gyro, dtype=np.float64)
        if accel.shape[0] <= 10:
            raise ValueError("movement detection needs > 10 samples")
        sos = butter(2, _EARTH_CUTOFF_HZ, fs=fs, btype="low", output="sos")
        gravity_dir = np.column_stack(
            [sosfiltfilt(sos, accel[:, j]) for j in range(3)])
        norm = np.linalg.norm(gravity_dir, axis=1, keepdims=True)
        norm[norm < 1e-9] = 1.0
        gravity_dir = gravity_dir / norm
        u_ref = self.rotation.T @ np.array([0.0, 0.0, 1.0])
        cos_dev = np.clip(gravity_dir @ u_ref, -1.0, 1.0)
        tilt = np.rad2deg(np.arccos(cos_dev))
        heading = np.linalg.norm((gyro - self.gyro_bias) @ self.rotation.T,
                                 axis=1)
        return (tilt > angle_deg) | (heading > gyro_thresh)

    # -- 4. frame transform ------------------------------------------------
    def to_vehicle_frame(self, gyro, accel, mag=None):
        """Transform (N, 3) phone-frame samples into the vehicle frame."""
        gyro = np.asarray(gyro, dtype=np.float64)
        accel = np.asarray(accel, dtype=np.float64)
        r_t = self.rotation.T
        gyro_v = (gyro - self.gyro_bias) @ r_t
        accel_v = (accel - self.accel_bias) @ r_t
        if mag is None:
            return gyro_v, accel_v
        return gyro_v, accel_v, np.asarray(mag) @ r_t

    # -- helpers for gate measurement / seeding ----------------------------
    def seed_state(self, gnss, t, start_t, speed_max=0.5, lookahead_s=(1.0, 5.0)):
        """(position_enu, velocity_enu, course_rad) at ``start_t`` from GNSS.

        Velocity comes from the median of fix-pair velocities with a
        1-5 s baseline (mirrors the engine's ``_init_state``): fixes 0.1 s
        apart carry ~zero displacement at low speed and only multipath
        wander at standstill, so a short lookahead underestimates motion.
        """
        east, north, speed, valid, fix_idx = _gnss_tracks(gnss, t)
        if fix_idx.size == 0:
            return np.zeros(3), np.zeros(3), 0.0
        i0 = fix_idx[np.searchsorted(t[fix_idx], start_t, side="right") - 1]
        pos = np.array([east[i0], north[i0], 0.0])
        vel = np.zeros(3)
        course = 0.0
        lo, hi = lookahead_s
        later = fix_idx[(t[fix_idx] >= t[i0] + lo) & (t[fix_idx] <= t[i0] + hi)]
        vels = []
        for j in later:
            dt = t[j] - t[i0]
            if dt <= 0:
                continue
            d = np.array([east[j] - east[i0], north[j] - north[i0]])
            v = d / dt
            # A GNSS reacquisition jump can ramp hundreds of m/s through
            # the interpolated grid; cap to vehicle-plausible speeds.
            if np.linalg.norm(v) > 45.0:
                continue
            if np.linalg.norm(d) / dt > speed_max:
                vels.append(v)
        if vels:
            vels = np.asarray(vels)
            median = np.median(vels, axis=0)
            # Consistency gate: keep only pairs agreeing with the median
            # (within 50% + 2 m/s); a lone huge pair must not dominate.
            agree = np.linalg.norm(vels - median, axis=1) < 0.5 * np.linalg.norm(
                median) + 2.0
            if agree.sum() >= 1:
                vel[:2] = np.mean(vels[agree], axis=0)
                course = float(np.arctan2(vel[1], vel[0]))
        return pos, vel, course


def _gnss_tracks(gnss, t):
    """ENU east/north (m, origin = first valid fix), speed, validity.

    Returns fix_idx: indices of DISTINCT consecutive fixes. The phone CSV
    repeats the last fix across rows (fixes update at ~1 Hz while rows are
    at 10-100 Hz), so differencing consecutive valid rows would see ~zero
    displacement and produce garbage speed/course; the distinct-fix
    compression gives real fix-to-fix displacement.
    """
    lat = np.deg2rad(gnss["lat_deg"])
    lon = np.deg2rad(gnss["lon_deg"])
    valid = (np.isfinite(lat) & np.isfinite(lon)
             & np.isfinite(gnss["accuracy_m"]))
    east = np.full(lat.size, np.nan)
    north = np.full(lat.size, np.nan)
    if valid.any():
        i0 = int(np.argmax(valid))
        east[valid] = ((lon[valid] - lon[i0]) * np.cos(lat[i0]) * _A)
        north[valid] = (lat[valid] - lat[i0]) * _A
    speed = np.full(lat.size, np.nan)
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return east, north, speed, valid, np.array([], dtype=int)
    # Distinct-fix compression with a minimum spacing. Two structures
    # occur in practice: (a) raw rows repeat the last fix (fixes ~10 s
    # apart) and (b) a resampled session has GNSS interpolated onto a
    # 100 Hz grid (every row moves a little). Keep a row when it moved
    # > 1 m from the last kept fix (real displacement; multipath wander
    # is metres, not millimetres) or when >= 1 s has elapsed since it
    # (real fix cadence). This mirrors the engine's 1-5 s fix spacing.
    kept = [idx[0]]
    for i in idx[1:]:
        moved = np.hypot(east[i] - east[kept[-1]], north[i] - north[kept[-1]])
        if moved > 1.0 or (t[i] - t[kept[-1]]) >= 1.0:
            kept.append(i)
    fix_idx = np.asarray(kept, dtype=int)
    if fix_idx.size > 1:
        step = np.hypot(np.diff(east[fix_idx]), np.diff(north[fix_idx]))
        dt = np.diff(t[fix_idx])
        dt[dt == 0.0] = np.inf
        speed[fix_idx[1:]] = step / dt
    return east, north, speed, valid, fix_idx


def _quasi_static_1s(session, fs, accel_var_max, gravity_tol,
                     gyro_var_max=0.02, gyro_mean_max=0.05):
    """Per-1-s-window quasi-static boolean over the whole session.

    Four criteria: accel variance, mean specific-force magnitude near g,
    gyro variance, and gyro MEAN magnitude. The mean criterion is what
    rejects constant-rate turns (e.g. smooth circular test-track driving:
    zero variance, but a persistent ~0.3 rad/s gyro mean; at rest the 1 s
    mean is ~0 because vibration averages out).
    """
    n = int(round(1.0 * fs))
    a_win = sliding_window_view(session.accel, n, axis=0)
    g_win = sliding_window_view(session.gyro, n, axis=0)
    a_var = a_win.var(axis=2).sum(axis=1)
    g_var = g_win.var(axis=2).sum(axis=1)
    a_mean_mag = np.linalg.norm(a_win.mean(axis=2), axis=1)
    g_mean_mag = np.linalg.norm(g_win.mean(axis=2), axis=1)
    return ((a_var < accel_var_max) & (g_var < gyro_var_max)
            & (g_mean_mag < gyro_mean_max)
            & (np.abs(a_mean_mag - G) < gravity_tol))


def quasi_static_mask(session, fs=None, accel_var_max=0.5, gravity_tol=0.5,
                      gyro_var_max=0.004, gyro_mean_max=0.05):
    """Per-SAMPLE quasi-static mask (True inside 1 s quasi-static stretches)."""
    if fs is None:
        fs = 1.0 / float(np.median(np.diff(session.t)))
    ok = _quasi_static_1s(session, fs, accel_var_max, gravity_tol,
                          gyro_var_max, gyro_mean_max)
    mask = np.zeros(session.t.size, dtype=bool)
    mask[:ok.size] = ok  # sample i carries the decision of its 1 s window
    return mask


def find_static_window(session, minimum_s=15.0, fs=None,
                       accel_var_max=0.5, gravity_tol=0.5, max_windows=20,
                       gyro_mean_max=0.05):
    """All quasi-static >= minimum_s windows (IMU-only detection).

    A 1 s stretch is quasi-static when the summed accelerometer variance is
    below ``accel_var_max``, the mean specific-force magnitude is within
    ``gravity_tol`` of g, and the mean gyro magnitude is below
    ``gyro_mean_max`` (rejects constant-rate turns, which have zero
    variance but a persistent rate). Constant straight-line velocity also
    passes, which is acceptable for bias estimation (its gyro mean is ~0).
    Measured true-static accel variance is ~0.05 (max 0.37).
    Returns up to ``max_windows`` non-overlapping (start, end) pairs.
    """
    if fs is None:
        fs = 1.0 / float(np.median(np.diff(session.t)))
    ok = _quasi_static_1s(session, fs, accel_var_max, gravity_tol,
                          gyro_mean_max=gyro_mean_max)
    n_need = int(np.ceil(minimum_s * fs))
    run = np.convolve(ok.astype(int), np.ones(n_need, dtype=int), "valid")
    starts = np.flatnonzero(run == n_need)
    windows = []
    last_end = -1
    for s in starts:
        if s <= last_end:
            continue  # non-overlapping only
        windows.append((int(s), int(s + n_need)))
        last_end = int(s + n_need)
        if len(windows) >= max_windows:
            break
    if not windows:
        raise ValueError("no IMU-static segment of %.0f s" % minimum_s)
    return windows


def calibrate_session(session, fs=None, static_s=15.0, static_windows=None):
    """Full calibration for one session: quasi-static windows + yaw.

    Gyro/accel biases are the MEDIAN of the per-window means across all
    detected quasi-static windows, so one contaminated window (smooth
    driving with real yaw rate / road dynamics) cannot poison the
    calibration. The leveling rotation comes from the least-variant
    window.
    """
    if fs is None:
        fs = 1.0 / float(np.median(np.diff(session.t)))
    if static_windows is None:
        static_windows = find_static_window(session, minimum_s=static_s,
                                            fs=fs)
    gyro_means, accel_means, variances = [], [], []
    for start, end in static_windows:
        gyro_means.append(np.mean(session.gyro[start:end], axis=0))
        accel_means.append(np.mean(session.accel[start:end], axis=0))
        variances.append(float(np.var(session.accel[start:end], axis=0)
                               .sum()))
    engine = CalibrationEngine()
    engine.gyro_bias = np.median(np.asarray(gyro_means), axis=0)
    best = int(np.argmin(variances))
    probe = CalibrationEngine()
    probe.calibrate_stationary(session.gyro[static_windows[best][0]:
                                             static_windows[best][1]],
                               session.accel[static_windows[best][0]:
                                             static_windows[best][1]])
    engine.level = probe.level
    # Accel bias: per-window observable part, median across windows.
    bias = [m - engine.level.T @ np.array([0.0, 0.0, G])
            for m in accel_means]
    engine.accel_bias = np.median(np.asarray(bias), axis=0)
    engine.calibrated = True
    engine.calibrate_yaw(session.t, session.gyro, session.gnss)
    return engine
