"""Tests for specs/01 section 6 outage segmentation."""

import numpy as np
import pytest

from python.io.iovnb_loader import GNSS_DTYPE, GT_POSE_DTYPE, IOVNBDSample, load_pair
from python.io.outages import TURN_FREE_MARGIN_S, carve_outages
from test_iovnbd_loader import shortest_synchronised_pair
GRAVITY = 9.80665


def _straight_turn_session(duration=120.0, turn_start=2.0, turn_end=12.0,
                           turn_total_deg=90.0, fs=100.0):
    """Straight driving with one constant-rate turn; ENU yaw from heading."""
    t = np.arange(0.0, duration, 1.0 / fs)
    n = t.size
    heading = np.zeros(n)
    mid = (t >= turn_start) & (t < turn_end)
    heading[mid] = (t[mid] - turn_start) * turn_total_deg / (turn_end - turn_start)
    heading[t >= turn_end] = turn_total_deg
    yaw = np.deg2rad(90.0 - heading)  # loader convention: psi = pi/2 - heading
    gnss = np.zeros(n, dtype=GNSS_DTYPE)
    gnss["t"] = t
    gnss["lat_deg"] = 52.27
    gnss["lon_deg"] = -2.12
    gnss["alt_m"] = 107.0
    gnss["accuracy_m"] = 4.0
    gt = np.zeros(n, dtype=GT_POSE_DTYPE)
    gt["t"] = t
    gt["vx"] = 10.0
    gt["qw"] = np.cos(yaw / 2.0)
    gt["qz"] = np.sin(yaw / 2.0)
    return IOVNBDSample(t=t, gyro=np.zeros((n, 3)),
                        accel=np.tile([0.0, 0.0, GRAVITY], (n, 1)),
                        mag=np.tile([1.0, 0.0, 0.0], (n, 1)),
                        gnss=gnss, gt_pose=gt)


def _gnss_nan_region(masked, start_t, end_t):
    inside = (masked.t >= start_t - 1e-12) & (masked.t < end_t - 1e-12)
    outside = ~inside
    return inside, outside


def test_outage_duration_exact():
    sess = _straight_turn_session()
    windows = carve_outages(sess, 30.0)
    assert windows
    for w in windows:
        assert w.duration_s == 30.0
        assert w.end_t - w.start_t == pytest.approx(30.0)
        inside, _ = _gnss_nan_region(w.masked, w.start_t, w.end_t)
        assert inside.sum() == 3000  # 30 s at 100 Hz


def test_gnss_masked_inside_valid_outside():
    sess = _straight_turn_session()
    gt_before = sess.gt_pose.copy()
    for w in carve_outages(sess, 60.0):
        inside, outside = _gnss_nan_region(w.masked, w.start_t, w.end_t)
        assert inside.any() and outside.any()
        for f in ("lat_deg", "lon_deg", "alt_m", "accuracy_m"):
            assert np.all(np.isnan(w.masked.gnss[f][inside]))
            assert np.all(np.isfinite(w.masked.gnss[f][outside]))
        assert np.all(np.isfinite(w.masked.gnss["t"]))
        # ground truth is set aside intact for scoring (§6.4)
        for f in w.masked.gt_pose.dtype.names:
            assert np.array_equal(w.masked.gt_pose[f], gt_before[f])


def test_boundaries_near_straight():
    # First (and last) sample of every outage must be turn-free per §6.5:
    # |heading change| over +/- 3 s <= 5 deg.
    sess = _straight_turn_session()
    yaw = 2.0 * np.arctan2(sess.gt_pose["qz"], sess.gt_pose["qw"])
    dt = 0.01
    w = int(round(3.0 / dt))
    for out in carve_outages(sess, 30.0):
        for boundary in (out.start_t, out.end_t):
            i = int(round((boundary - sess.t[0]) / dt))
            lo, hi = max(0, i - w), min(sess.t.size - 1, i + w)
            d = abs((yaw[hi] - yaw[lo] + np.pi) % (2 * np.pi) - np.pi)
            assert np.rad2deg(d) <= 5.0 + 1e-9


def test_mid_turn_candidates_rejected():
    # Turn occupies 2-12 s. The turn metric (|heading change| > 5 deg over
    # +/- 3 s) stays active until t=15 s, and the 3 s turn-free margin stacks
    # on top, so candidates 5..17 s are rejected and the first accepted
    # window starts at t=18 s.
    sess = _straight_turn_session(turn_start=2.0, turn_end=12.0)
    windows = carve_outages(sess, 30.0)
    assert windows
    assert windows[0].start_t == pytest.approx(18.0)
    assert len(windows[0].rejected) == 13
    assert all("mid-turn" in r for _, r in windows[0].rejected)


def test_short_session_returns_empty():
    s_path, v_path = shortest_synchronised_pair()
    sample = load_pair(v_path, s_path)
    assert carve_outages(sample, 30.0) == []


def test_outage_s_validation():
    sess = _straight_turn_session()
    with pytest.raises(ValueError):
        carve_outages(sess, 45.0)