"""Tests for python/eval/run_evaluation.py (specs/04 section 6 + specs/03
section 6 metrics). Synthetic, deterministic, offline: no OSM download and
no real CNN weights are touched."""

import json
from pathlib import Path

import numpy as np
import pytest

from python.eval.engine import _gnss_enu, _match_headings
from python.eval.run_evaluation import (compute_metrics, evaluate_all,
                                        evaluate_window, export_session_trajectories,
                                        write_metrics, write_plot)
from python.io.iovnb_loader import GNSS_DTYPE, GT_POSE_DTYPE, IOVNBDSample
from python.io.outages import carve_outages

GRAVITY = 9.80665
_A = 6378137.0
LAT0, LON0 = np.deg2rad(52.0), np.deg2rad(0.0)


def _to_geo(x, y):
    """Inverse of the loader ENU mapping (x East, y North, z 0)."""
    lat = np.rad2deg(LAT0 + y / _A)
    lon = np.rad2deg(LON0 + x / (_A * np.cos(LAT0)))
    return lat, lon


def _straight_session(duration=120.0, fs=100.0, speed=10.0):
    """Constant-speed straight-East session, 10 Hz GNSS, at fs Hz."""
    n = int(round(duration * fs))
    t = np.arange(n, dtype=np.float64) / fs
    x = speed * t
    gyro = np.zeros((n, 3))
    accel = np.tile([0.0, 0.0, GRAVITY], (n, 1))
    mag = np.tile([1.0, 0.0, 0.0], (n, 1))
    gnss = np.zeros(n, dtype=GNSS_DTYPE)
    gnss["t"] = t
    for f in gnss.dtype.names:
        if f != "t":
            gnss[f] = np.nan  # non-fix rows stay NaN until assigned
    gnss["accuracy_m"] = 5.0
    step = int(round(fs / 10.0))
    fix = np.zeros(n, dtype=bool)
    fix[::step] = True
    lat, lon = _to_geo(x, np.zeros(n))
    gnss["lat_deg"][fix] = lat[fix]
    gnss["lon_deg"][fix] = lon[fix]
    gnss["alt_m"][fix] = 100.0
    gt = np.zeros(n, dtype=GT_POSE_DTYPE)
    gt["t"] = t
    gt["x"] = x
    gt["vx"] = speed
    gt["qw"] = 1.0
    return IOVNBDSample(t=t, gyro=gyro, accel=accel, mag=mag, gnss=gnss,
                        gt_pose=gt)


def _first_window(session, duration=30.0):
    windows = carve_outages(session, duration)
    assert windows, "synthetic session should yield an outage window"
    return windows[0]


class _StubMatcher:
    """Returns one match per point on a fixed road heading."""

    def __init__(self, heading=0.0):
        self.heading = heading

    def match(self, trajectory_xy):
        return [type("M", (), {"heading": self.heading})()
                for _ in range(len(trajectory_xy))]


def test_compute_metrics_known_answer():
    n = 100
    t = np.arange(n, dtype=float)
    gt = np.column_stack((0.1 * t, np.zeros(n)))
    # constant +3 m East offset: ate == drift == 3, rpe == 0, rel exact.
    est = gt + np.array([3.0, 0.0])
    m = compute_metrics(est, gt)
    assert m["ate"] == pytest.approx(3.0, abs=1e-9)
    assert m["drift"] == pytest.approx(3.0, abs=1e-9)
    assert m["rpe"] == pytest.approx(0.0, abs=1e-9)
    assert m["rel"] == pytest.approx(300.0 / 9.9, rel=1e-6)
    # doubling the per-step displacement: rpe == the 0.1 m step.
    est2 = np.column_stack((gt[:, 0] * 2.0, np.zeros(n)))
    assert compute_metrics(est2, gt)["rpe"] == pytest.approx(0.1, rel=1e-9)


def test_gnss_enu_rejects_invalid_altitude():
    sess = _straight_session(duration=2.0)
    sess.gnss["alt_m"][0] = np.nan
    _, _, _, _, valid = _gnss_enu(sess)
    assert not valid[0]
    expected = np.isfinite(sess.gnss["lat_deg"])
    expected &= np.isfinite(sess.gnss["lon_deg"])
    expected &= np.isfinite(sess.gnss["alt_m"])
    expected &= np.isfinite(sess.gnss["accuracy_m"])
    assert np.array_equal(valid, expected)


def test_sparse_match_headings_are_interpolated_circularly():
    class SparseMatcher:
        def match(self, trajectory_xy):
            return [type("M", (), {"heading": 3.0})(),
                    type("M", (), {"heading": -3.0})()]

    headings = _match_headings(SparseMatcher(), np.zeros((5, 2)))
    assert np.all(np.isfinite(headings))
    assert abs(abs(headings[2]) - np.pi) < 0.2


def test_straight_session_metrics_small_and_four_configs():
    sess = _straight_session()
    w = _first_window(sess, 30.0)
    matcher = _StubMatcher(heading=0.0)
    r = evaluate_window(w.masked, w.start_t, w.end_t, "W01-30s", "V-test",
                        matcher=matcher, model=None)
    n = int(30.0 * 100.0)
    for key in ("raw", "zupt", "cnn", "full"):
        assert r.traj[key].shape == (n, 2)
    assert r.nhc_applied > 0
    assert r.fallback == ""
    assert r.duration_s == 30.0
    # Constant straight East with clean IMU and matched speed seed: small.
    for key in ("raw", "zupt", "cnn", "full"):
        m = compute_metrics(r.traj[key], r.gt)
        assert m["ate"] < 5.0 and m["drift"] < 5.0 and m["rpe"] < 5.0
    # GNSS-frozen is the last pre-outage fix, constant over the outage.
    assert np.allclose(r.frozen[0], r.frozen[-1])


def test_no_matcher_falls_back_without_nhc():
    sess = _straight_session()
    w = _first_window(sess, 30.0)
    r = evaluate_window(w.masked, w.start_t, w.end_t, "W01-30s", "V-test",
                        matcher=None, model=None)
    assert r.fallback == "match unavailable"
    assert r.nhc_applied == 0
    assert np.allclose(r.traj["full"], r.traj["cnn"], atol=1e-9)


def test_empty_matches_falls_back():
    sess = _straight_session()
    w = _first_window(sess, 30.0)

    class _NoMatch:
        def match(self, trajectory_xy):
            return []

    r = evaluate_window(w.masked, w.start_t, w.end_t, "W01-30s", "V-test",
                        matcher=_NoMatch(), model=None)
    assert r.fallback == "no road match accepted"
    assert r.nhc_applied == 0
    assert np.allclose(r.traj["full"], r.traj["cnn"], atol=1e-9)


def test_export_session_trajectories_json(tmp_path):
    out = tmp_path / "sessions" / "session_01" / "trajectories.json"
    payload = {
        "raw": [(1.0, 2.0), (3.0, 4.0)],
        "inertial": [(5.0, 6.0)],
        "fused": [(7.0, 8.0)],
    }
    export_session_trajectories("session_01", payload, out)
    data = json.loads(out.read_text())
    assert data["session_id"] == "session_01"
    assert data["raw"] == [[1.0, 2.0], [3.0, 4.0]]
    assert data["inertial"] == [[5.0, 6.0]]
    assert data["fused"] == [[7.0, 8.0]]


def test_evaluate_all_and_file_outputs(tmp_path):
    sess = _straight_session(duration=120.0)
    w30 = _first_window(sess, 30.0)
    results = evaluate_all([("V-test30", w30)], matcher=_StubMatcher(0.0))
    assert len(results) == 1
    assert results[0].window_id == "W01-30s"
    md = tmp_path / "final_metrics.md"
    png = tmp_path / "final_comparison.png"
    write_metrics(results, md)
    write_plot(results, png)
    text = md.read_text()
    assert "Per-window comparison (specs/04 section 6)" in text
    assert "ATE / RPE / drift by outage duration" in text
    assert "raw-INS" in text and "+NHC+matching" in text
    assert "W01-30s" in text and "V-test30" in text
    assert png.exists() and png.stat().st_size > 1000
