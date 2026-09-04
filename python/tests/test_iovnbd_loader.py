"""IO-VNBD loader test (specs/01 sections 5 and 8).

Loads the shortest available synchronised V/S session from data/ (measured:
session Vw13) and asserts shapes/dtypes, strictly monotonic timestamps, no
NaN/inf, and gyro/accel/mag/GNSS plausibility per the sanity checklist.
"""

import re
from pathlib import Path

import numpy as np
import pytest

from python.io.iovnb_loader import (GNSS_DTYPE, GT_POSE_DTYPE, IOVNBDSample,
                                    load_pair, resample_to_common_clock)

DATA_ROOT = (
    Path(__file__).resolve().parents[2]
    / "data/IO-VNBD-master/Synchronised V abd S datasets"
    / "Uncategorised IOVNB Dataset"
)
S_DIR = DATA_ROOT / "S-Dataset"
V_DIR = DATA_ROOT / "V-Dataset"

GRAVITY = 9.80665


def _pointer_size(path):
    """Real byte size from an LFS pointer line (data may still be pointers)."""
    head = path.read_text(errors="replace")[:200]
    m = re.search(r"size (\d+)", head)
    return int(m.group(1)) if m else path.stat().st_size


def shortest_synchronised_pair():
    """Smallest S/V session pair by total file size in the sync tree."""
    if not S_DIR.is_dir() or not V_DIR.is_dir():
        pytest.fail(
            f"dataset tree missing at {DATA_ROOT} — restore data/IO-VNBD-master "
            "before running (see data-health report)"
        )
    s_files = {p.name: p for p in S_DIR.glob("S-*.csv")}
    v_files = {p.name: p for p in V_DIR.glob("V-*.csv")}
    pairs = []
    for name, s_path in s_files.items():
        session = name[len("S-"):]
        v_path = v_files.get("V-" + session)
        if v_path is not None:
            pairs.append((_pointer_size(s_path) + _pointer_size(v_path),
                          s_path, v_path))
    if not pairs:
        pytest.fail(f"no synchronised S/V session pair found under {DATA_ROOT}")
    _, s_path, v_path = min(pairs, key=lambda p: p[0])
    return s_path, v_path


@pytest.fixture(scope="module")
def sample():
    s_path, v_path = shortest_synchronised_pair()
    return load_pair(v_path, s_path)


def test_shapes_and_dtypes(sample):
    n = sample.t.shape[0]
    assert n >= 100
    for arr in (sample.t, sample.gyro, sample.accel, sample.mag):
        assert arr.dtype == np.float64
    assert sample.gyro.shape == (n, 3)
    assert sample.accel.shape == (n, 3)
    assert sample.mag.shape == (n, 3)
    assert sample.gnss.dtype == GNSS_DTYPE
    assert sample.gnss.shape[0] >= 1
    assert sample.gt_pose.dtype == GT_POSE_DTYPE
    assert sample.gt_pose.shape[0] >= 1


def test_timestamps_strictly_monotonic(sample):
    assert np.all(np.diff(sample.t) > 0)
    assert np.all(np.diff(sample.gnss["t"]) > 0)
    assert np.all(np.diff(sample.gt_pose["t"]) > 0)


def test_no_nan_or_infinite(sample):
    for arr in (sample.t, sample.gyro, sample.accel, sample.mag, sample.gnss,
                sample.gt_pose):
        fields = arr.dtype.names or ()
        values = [arr] if not fields else [arr[f] for f in fields]
        assert all(np.all(np.isfinite(v)) for v in values)


def test_gyro_accel_magnitudes_plausible(sample):
    # Sanity checklist items 2/5: mean accel magnitude ~ g while driving
    # (measured 9.8 m/s^2 on Vw13, motion adds ~+/-0.5); gyro bound is a
    # driving-plausibility ceiling far above the measured ~1.3 rad/s.
    accel_norm = np.linalg.norm(sample.accel, axis=1)
    assert 8.5 <= accel_norm.mean() <= 11.0
    gyro_norm = np.linalg.norm(sample.gyro, axis=1)
    assert np.max(gyro_norm) <= 5.0
    # mag is a unit vector by contract
    assert np.allclose(np.linalg.norm(sample.mag, axis=1), 1.0, atol=1e-6)


def test_gnss_plausible(sample):
    # Smartphone GNSS is 10 Hz per the README; measured mean dt ~0.1 s.
    dt = np.diff(sample.gnss["t"])
    assert 0.05 <= dt.mean() <= 0.2
    # Vw13 is a UK recording near (52.27 N, -2.12 E); lat/lon sanity only.
    assert np.all(sample.gnss["lat_deg"] > 0.0)
    assert np.all(np.abs(sample.gnss["lon_deg"] - (-2.12)) < 0.5)
    assert np.all(sample.gnss["accuracy_m"] >= 0.0)


def test_gt_pose_plausible(sample):
    # Vehicle speed measured 94-113 km/h -> 26-31 m/s on Vw13.
    speed = np.hypot(sample.gt_pose["vx"], sample.gt_pose["vy"])
    assert 10.0 <= speed.mean() <= 40.0
    # Hamilton quaternion is a unit quaternion.
    q = np.column_stack((sample.gt_pose["qw"], sample.gt_pose["qx"],
                         sample.gt_pose["qy"], sample.gt_pose["qz"]))
    assert np.allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-9)
    # ENU z is the smartphone-altitude delta; small for this flat UK session.
    assert np.all(np.abs(sample.gt_pose["z"]) < 25.0)
    assert sample.gt_pose["t"].shape == sample.gt_pose["vx"].shape


def _sine_session(duration=100.0, freq=0.001, fs_in=10.0):
    """Synthetic 10 Hz session whose channels are a very slow sine, so linear
    interpolation to 100 Hz has error far below 1e-6 (known-answer test)."""
    t = np.arange(0.0, duration, 1.0 / fs_in)
    n = t.size
    wave = np.sin(2.0 * np.pi * freq * t)
    gyro = np.column_stack([wave] * 3)
    gnss = np.zeros(n, dtype=GNSS_DTYPE)
    gnss["t"] = t
    gnss["lat_deg"] = wave
    gnss["lon_deg"] = wave
    gnss["alt_m"] = wave
    gnss["accuracy_m"] = wave
    gt_pose = np.zeros(n, dtype=GT_POSE_DTYPE)
    gt_pose["t"] = t
    gt_pose["x"] = wave
    return IOVNBDSample(t=t, gyro=gyro, accel=gyro.copy(), mag=gyro.copy(),
                        gnss=gnss, gt_pose=gt_pose)


def test_resampled_length(sample):
    out = resample_to_common_clock(sample, fs=100.0)
    duration = sample.t[-1] - sample.t[0]
    assert abs(out.t.size - duration * 100.0) <= 1.0
    assert out.gyro.shape[0] == out.accel.shape[0] == out.mag.shape[0]
    assert out.gnss.shape[0] == out.gt_pose.shape[0] == out.t.size


def test_resample_introduces_no_nan(sample):
    # Vw13 has no GNSS gaps, so resampling must keep every value finite.
    out = resample_to_common_clock(sample, fs=100.0)
    for arr in (out.t, out.gyro, out.accel, out.mag, out.gnss, out.gt_pose):
        fields = arr.dtype.names or ()
        arrs = [arr] if not fields else [arr[f] for f in fields]
        assert all(np.all(np.isfinite(v)) for v in arrs)


def test_resampled_gnss_timestamps_strictly_increasing(sample):
    out = resample_to_common_clock(sample, fs=100.0)
    assert np.all(np.diff(out.gnss["t"]) > 0)
    assert np.all(np.diff(out.gt_pose["t"]) > 0)


def test_resample_sine_interpolation_error():
    freq = 0.001  # max |x''| = (2*pi*f)^2 ~ 4e-5 -> linear error < 1e-6
    sess = _sine_session(freq=freq)
    out = resample_to_common_clock(sess, fs=100.0)
    true = np.sin(2.0 * np.pi * freq * out.t)
    assert np.max(np.abs(out.gyro[:, 0] - true)) < 1e-6
    assert np.max(np.abs(out.gnss["lat_deg"] - true)) < 1e-6
    assert np.max(np.abs(out.gt_pose["x"] - true)) < 1e-6


def test_gnss_gap_never_interpolated():
    # specs/01 §6.3: a GNSS outage must not be interpolated across.
    sess = _sine_session(duration=10.0)
    keep = ~((sess.gnss["t"] > 4.0) & (sess.gnss["t"] < 6.0))
    gnss = sess.gnss[keep]  # gap 4.0 s -> 6.0 s, dt = 2.0 s > 3x median
    sess = IOVNBDSample(t=sess.t, gyro=sess.gyro, accel=sess.accel,
                        mag=sess.mag, gnss=gnss, gt_pose=sess.gt_pose)
    out = resample_to_common_clock(sess, fs=100.0)
    inside = (out.t > 4.0 + 1e-9) & (out.t < 6.0 - 1e-9)
    assert inside.any()
    for f in ("lat_deg", "lon_deg", "alt_m", "accuracy_m"):
        assert np.all(np.isnan(out.gnss[f][inside]))
    assert np.all(np.isfinite(out.gnss["t"][inside]))
    # outside the gap interpolation is normal and finite
    outside = ~inside
    for f in ("lat_deg", "lon_deg", "alt_m", "accuracy_m"):
        assert np.all(np.isfinite(out.gnss[f][outside]))