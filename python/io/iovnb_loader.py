"""IO-VNBD synchronised pair loader (specs/01 sections 1, 3, 5).

load_pair(v_csv, s_csv) reads a synchronised vehicle (V-*) + smartphone (S-*)
CSV pair into an IOVNBDSample; all unit conversions live here (t ms->s;
gyro/accel already rad/s, m/s^2; mag uT->unit vector; gt_pose km/h->m/s,
deg->ENU yaw quaternion, GPS->ENU m with origin = first fix).

The synchronized recordings require a measured vehicle-clock correction of
-7.1 s relative to the smartphone clock; callers may override it per session.
Timestamp-glitch rows (backward jumps, duplicate timestamps, NaN clocks) are
dropped from both streams before any channel is read, so row counts never
need to match and single corrupted rows cannot reject a session.

resample_to_common_clock(session, fs) puts every stream onto one uniform fs Hz
clock (specs/01 §3) by linear interpolation. Per §6.3 GNSS is never
interpolated across a gap (interval > 3x median inter-fix time; threshold
VERIFY per §3): gap interiors carry NaN in every GNSS field except t, and the
rule is asserted. gt_pose is warped onto the smartphone clock without offset
correction (alignment VERIFY per §3).

Measured quirk (Vw13): vehicle 'Height (km)' is corrupt (51-59 km), so gt_pose
z uses the smartphone GPS altitude paired by row index (both streams 10 Hz).
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_WGS84_A = 6378137.0  # WGS84 semi-major axis, metres
VEHICLE_TIME_OFFSET_S = -7.1

GNSS_DTYPE = np.dtype([("t", "f8"), ("lat_deg", "f8"), ("lon_deg", "f8"), ("alt_m", "f8"), ("accuracy_m", "f8")])
GT_POSE_DTYPE = np.dtype([("t", "f8"), ("x", "f8"), ("y", "f8"), ("z", "f8"), ("vx", "f8"), ("vy", "f8"), ("vz", "f8"), ("qw", "f8"), ("qx", "f8"), ("qy", "f8"), ("qz", "f8")])

@dataclass
class IOVNBDSample:
    t: np.ndarray  # (N,) float64 seconds, smartphone clock
    gyro: np.ndarray  # (N, 3) rad/s
    accel: np.ndarray  # (N, 3) m/s^2
    mag: np.ndarray  # (N, 3) normalised unit vector
    gnss: np.ndarray  # (M,) GNSS_DTYPE
    gt_pose: np.ndarray  # (K,) GT_POSE_DTYPE

def _header_index(headers, *keys):
    """Column index of the first header containing all keys, else ValueError."""
    norm = [h.strip().lower() for h in headers]
    for key in keys:
        k = key.strip().lower()
        try:
            return next(i for i, h in enumerate(norm) if k in h)
        except StopIteration:
            raise ValueError(f"CSV missing column {key!r}; header: {headers}")

def _header_index_any(headers, *alternatives):
    """Return the first matching column from a list of header alternatives."""
    for alternative in alternatives:
        keys = (alternative,) if isinstance(alternative, str) else alternative
        try:
            return _header_index(headers, *keys)
        except ValueError:
            continue
    raise ValueError(f"CSV missing columns matching {alternatives!r}; header: {headers}")

def _read_csv(path):
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows[0], rows[1:]

def _float_column(rows, idx):
    out = []
    for r in rows:
        try:
            out.append(float(r[idx].strip()))
        except (IndexError, ValueError):
            out.append(float("nan"))
    return np.asarray(out, dtype=np.float64)


def _monotonic_filter(values):
    """Boolean mask keeping samples with finite, strictly increasing values.

    Drops single glitch rows (backward jumps, duplicate timestamps, or NaN
    clocks) seen in some IO-VNBD recordings (e.g. S-S2 has one backward row,
    V-vtb2 one duplicate timestamp). The mask is cheap and order-preserving.
    """
    values = np.asarray(values, dtype=np.float64)
    keep = np.zeros(values.size, dtype=bool)
    last = -np.inf
    for i, value in enumerate(values):
        if np.isfinite(value) and value > last:
            keep[i] = True
            last = value
    return keep

def cross_correlation_lag(phone_t, phone_accel, vehicle_t, vehicle_speed):
    """Return correlation lag in phone samples and seconds."""
    accel = np.linalg.norm(phone_accel, axis=1)
    speed = np.interp(phone_t, vehicle_t, vehicle_speed)
    a, b = accel - accel.mean(), speed - speed.mean()
    corr = np.correlate(a, b, mode="full")
    lags = np.arange(-b.size + 1, a.size)
    index = int(np.argmax(corr))
    dt = np.median(np.diff(phone_t))
    return int(lags[index]), float(lags[index] * dt)


def estimate_time_offset(phone_t, phone_accel, vehicle_t, vehicle_speed):
    """Return lag to add to normalized vehicle timestamps.

    A positive lag means vehicle time is shifted later; the correction is
    therefore ``vehicle_t_corrected = vehicle_t_normalized + lag``.
    """
    return cross_correlation_lag(
        phone_t, phone_accel, vehicle_t, vehicle_speed)[1]


def _manifest_path(s_csv):
    return Path(s_csv).with_suffix(".manifest.json")


def _session_offset(s_csv, phone_t, phone_accel, vehicle_t, vehicle_speed):
    path = _manifest_path(s_csv)
    if path.is_file():
        data = json.loads(path.read_text())
        return float(data["vehicle_time_offset_s"])
    offset = estimate_time_offset(phone_t, phone_accel, vehicle_t, vehicle_speed)
    path.write_text(json.dumps({
        "vehicle_time_offset_s": offset,
        "sign_convention": "add to normalized vehicle timestamps",
    }, indent=2) + "\n")
    return offset


def load_pair(v_csv, s_csv, vehicle_time_offset_s=None) -> IOVNBDSample:
    """Load a synchronized V/S pair and apply its cached time correction."""
    v_csv = Path(v_csv)
    s_csv = Path(s_csv)
    if not v_csv.is_file():
        raise FileNotFoundError(f"vehicle CSV not found: {v_csv}")
    if not s_csv.is_file():
        raise FileNotFoundError(f"smartphone CSV not found: {s_csv}")

    v_head, v_rows = _read_csv(v_csv)
    s_head, s_rows = _read_csv(s_csv)
    # smartphone stream (t, gyro, accel, mag, gnss)
    i_t = _header_index(s_head, "time since start")
    i_lat = _header_index(s_head, "gps latitude")
    i_lon = _header_index(s_head, "gps longitude")
    i_alt = _header_index(s_head, "gps altitude")
    i_acc = _header_index(s_head, "gps accuracy")

    def cols(prefix):
        if prefix == "gyroscope":
            # IO-VNBD uses either Cartesian axes or Euler names for gyro data.
            names = (
                ("gyroscope x", "gyroscope roll"),
                ("gyroscope y", "gyroscope pitch"),
                ("gyroscope z", "gyroscope yaw"),
            )
        else:
            names = tuple((f"{prefix} {c}",) for c in "xyz")
        idx = [_header_index_any(s_head, *aliases) for aliases in names]
        return np.column_stack([_float_column(s_rows, i) for i in idx])

    t_raw = _float_column(s_rows, i_t) / 1000.0  # ms -> s
    if t_raw.size == 0 or not np.isfinite(t_raw[0]):
        raise ValueError("smartphone timestamps must start with a finite value")
    # Drop timestamp-glitch rows (backward/duplicate/NaN clocks) before
    # extracting any channel so all arrays stay row-aligned.
    keep_s = _monotonic_filter(t_raw)
    if keep_s.sum() < len(s_rows):
        s_rows = [r for r, keep in zip(s_rows, keep_s) if keep]
        t_raw = _float_column(s_rows, i_t) / 1000.0
    t = t_raw - t_raw[0]
    accel = cols("accelerometer")
    gyro = cols("gyroscope")
    mag_raw = cols("magnetic field")
    mag_norm = np.linalg.norm(mag_raw, axis=1)
    mag = np.zeros_like(mag_raw)
    ok = mag_norm > 0
    mag[ok] = mag_raw[ok] / mag_norm[ok, None]

    n = len(t)
    gnss = np.zeros(n, dtype=GNSS_DTYPE)
    gnss["t"] = t
    for f, i in (("lat_deg", i_lat), ("lon_deg", i_lon),
                 ("alt_m", i_alt), ("accuracy_m", i_acc)):
        gnss[f] = _float_column(s_rows, i)

    # vehicle stream (gt_pose)
    i_vt = _header_index_any(
        v_head,
        ("time since start",),
        ("time since start of day",),
    )
    i_vlat = _header_index(v_head, "latitude")
    i_vlon = _header_index(v_head, "longitude")
    i_vel = _header_index(v_head, "velocity")
    i_hdg = _header_index(v_head, "heading")

    v_t_raw = _float_column(v_rows, i_vt)
    if v_t_raw.size == 0 or not np.isfinite(v_t_raw[0]):
        raise ValueError("vehicle timestamps must start with a finite value")
    # Same glitch filter on the vehicle stream (e.g. V-vtb2 duplicates).
    keep_v = _monotonic_filter(v_t_raw)
    if keep_v.sum() < len(v_rows):
        v_rows = [r for r, keep in zip(v_rows, keep_v) if keep]
        v_t_raw = _float_column(v_rows, i_vt)
    v_t_normalized = v_t_raw - v_t_raw[0]
    v_lat = _float_column(v_rows, i_vlat)
    v_lon = _float_column(v_rows, i_vlon)
    speed = _float_column(v_rows, i_vel) / 3.6  # km/h -> m/s
    heading_deg = _float_column(v_rows, i_hdg)
    if vehicle_time_offset_s is None:
        vehicle_time_offset_s = _session_offset(
            s_csv, t, accel, v_t_normalized, speed)
    if not np.isscalar(vehicle_time_offset_s):
        raise ValueError("vehicle_time_offset_s must be a scalar")
    v_t = v_t_normalized + float(vehicle_time_offset_s)
    k = len(v_rows)
    lat0 = np.deg2rad(v_lat[0]) if k else 0.0
    lon0 = np.deg2rad(v_lon[0]) if k else 0.0
    valid_alt = np.isfinite(gnss["alt_m"])
    if not np.any(valid_alt):
        raise ValueError("smartphone altitude must contain a finite value")
    alt_t = gnss["t"][valid_alt]
    alt_values = gnss["alt_m"][valid_alt]
    alt0 = float(alt_values[0])  # corrupt V height -> S GPS altitude
    gt_pose = np.zeros(k, dtype=GT_POSE_DTYPE)
    if k:
        gt_pose["t"] = v_t
        gt_pose["x"] = (np.deg2rad(v_lon) - lon0) * np.cos(lat0) * _WGS84_A
        gt_pose["y"] = (np.deg2rad(v_lat) - lat0) * _WGS84_A
        gt_pose["z"] = np.interp(v_t, alt_t, alt_values) - alt0
        theta = np.deg2rad(heading_deg)
        psi = np.pi / 2.0 - theta  # ENU yaw from East, CCW
        gt_pose["vx"] = speed * np.sin(theta)
        gt_pose["vy"] = speed * np.cos(theta)
        gt_pose["qw"] = np.cos(psi / 2.0)
        gt_pose["qz"] = np.sin(psi / 2.0)
    sample = IOVNBDSample(t=t, gyro=gyro, accel=accel, mag=mag, gnss=gnss,
                          gt_pose=gt_pose)
    _assert_valid(sample)
    return sample

def resample_to_common_clock(session, fs=100.0) -> IOVNBDSample:
    """Resample every stream of session onto a uniform fs Hz clock (§3)."""
    fs = float(fs)
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")
    t0, t1 = session.t[0], session.t[-1]
    n_out = int(np.ceil((t1 - t0) * fs))
    t_out = t0 + np.arange(n_out) / fs
    gyro = _interp_cols(t_out, session.t, session.gyro)
    accel = _interp_cols(t_out, session.t, session.accel)
    mag = _interp_cols(t_out, session.t, session.mag)
    gnss = _resample_gnss(session.gnss, t_out)
    gt_pose = _resample_gt_pose(session.gt_pose, t_out)
    out = IOVNBDSample(t=t_out, gyro=gyro, accel=accel, mag=mag, gnss=gnss, gt_pose=gt_pose)
    _assert_valid(out, allow_gnss_nan=True)
    return out

def _interp_cols(t_out, t_in, values):
    return np.column_stack([np.interp(t_out, t_in, values[:, j])
                            for j in range(values.shape[1])])

def _resample_gnss(gnss, t_out):
    out = np.zeros(t_out.size, dtype=GNSS_DTYPE)
    out["t"] = t_out
    for f in gnss.dtype.names:
        if f != "t":
            out[f] = np.interp(t_out, gnss["t"], gnss[f])
    dt = np.diff(gnss["t"])
    thresh = 3.0 * np.median(dt) if dt.size else np.inf  # gap rule, VERIFY §3
    for i in np.nonzero(dt > thresh)[0]:
        inside = (t_out > gnss["t"][i]) & (t_out < gnss["t"][i + 1])
        for f in gnss.dtype.names:
            if f != "t":
                out[f][inside] = np.nan
        # specs/01 §6.3: never interpolate GNSS inside an outage.
        assert not np.any(np.isfinite(np.concatenate(
            [out[f][inside] for f in gnss.dtype.names if f != "t"])))
    return out

def _resample_gt_pose(gt_pose, t_out):
    out = np.zeros(t_out.size, dtype=GT_POSE_DTYPE)
    out["t"] = t_out
    for f in gt_pose.dtype.names:
        if f != "t":
            out[f] = np.interp(t_out, gt_pose["t"], gt_pose[f])
    return out

def _assert_valid(sample, allow_gnss_nan=False):
    """Construction asserts: shapes, dtypes, monotonic t, finite values."""
    n = sample.t.shape[0]
    if n == 0:
        raise ValueError("empty smartphone stream")
    for name, arr, shape in (("t", sample.t, (n,)), ("gyro", sample.gyro, (n, 3)),
                             ("accel", sample.accel, (n, 3)), ("mag", sample.mag, (n, 3))):
        if arr.shape != shape or arr.dtype != np.float64:
            raise ValueError(f"{name}: expected {shape} float64, got {arr.shape} {arr.dtype}")
    for name, arr in (("gnss", sample.gnss), ("gt_pose", sample.gt_pose)):
        if arr.dtype.names is None or not all(arr[f].dtype == np.float64 for f in arr.dtype.names):
            raise ValueError(f"{name}: expected all-float64 structured array")
    for name, arr in (("t", sample.t), ("gnss.t", sample.gnss["t"]), ("gt_pose.t", sample.gt_pose["t"])):
        if arr.size and not np.all(np.diff(arr) > 0):
            raise ValueError(f"{name}: timestamps not strictly increasing")
    for name, arr in (("t", sample.t), ("gyro", sample.gyro), ("accel", sample.accel),
                      ("mag", sample.mag), ("gnss", sample.gnss), ("gt_pose", sample.gt_pose)):
        fields = arr.dtype.names or ()
        arrs = [arr] if not fields else [arr[f] for f in fields]
        if not all(np.all(np.isfinite(v)) for v in arrs):
            if allow_gnss_nan and name == "gnss":
                continue
            raise ValueError(f"{name}: contains NaN or infinite values")