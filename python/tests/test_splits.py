"""Phase 0 artifact tests: frozen split + exact outage-window records."""

import numpy as np
import pytest

from python.eval.splits import (build_split, load_window_records,
                                load_split, rebuild_window, save_split,
                                save_windows, session_key, window_records)
from python.io.iovnb_loader import GNSS_DTYPE, IOVNBDSample
from python.io.outages import OutageWindow


def _make_root(tmp_path):
    """Tiny fake dataset tree spanning four drivers and both layouts."""
    root = tmp_path / "data"
    for rel in ("Categorised IOVNB Dataset/S (Driver A)/Sa01/S-Sa01.csv",
                "Categorised IOVNB Dataset/M (Driver B)/M01/S-M01.csv",
                "Categorised IOVNB Dataset/Y (Driver D)/Y01/S-Y01.csv",
                "Categorised IOVNB Dataset/Vw (Driver E)/Vw04/S-Vw04.csv",
                "Uncategorised IOVNB Dataset/S-Dataset/S-Vw04.csv"):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
        (p.parent / ("V-" + p.name[2:])).touch()
    return root


def test_session_key_categorised_tree():
    key = session_key("d/Categorised IOVNB Dataset/Vw (Driver E)/Vw04/"
                      "S-Vw4.csv")
    assert key["driver"] == "Driver E"
    assert key["route"] == "Vw"
    assert key["session"] == "Vw4"
    assert key["device"] == "unknown"


def test_session_key_uncategorised_tree():
    key = session_key("d/Uncategorised IOVNB Dataset/S-Dataset/S-Vw9.csv")
    assert key["driver"] == "unknown"
    assert key["session"] == "Vw9"


def test_split_is_deterministic_and_driver_level(tmp_path):
    root = _make_root(tmp_path)
    first, second = build_split(root), build_split(root)
    assert first == second  # frozen: no randomness
    sessions = first["sessions"]
    assert sessions["Vw04"]["split"] == "test"    # last sorted driver -> E
    assert sessions["Y01"]["split"] == "val"      # second-to-last -> D
    assert sessions["Sa01"]["split"] == "train"   # A, B -> train
    assert sessions["M01"]["split"] == "train"
    # every split carries driver, route and device metadata
    for s in sessions.values():
        assert {"driver", "route", "device", "split"} <= set(s)


def test_split_round_trip(tmp_path):
    root = _make_root(tmp_path)
    path = save_split(build_split(root), tmp_path / "split.json")
    assert path.is_file()
    assert load_split(tmp_path / "split.json") == build_split(root)


def test_window_records_round_trip(tmp_path):
    root = _make_root(tmp_path)
    w = OutageWindow(start_t=10.0, end_t=40.0, duration_s=30.0,
                     masked=None, rejected=["turn too close"])
    records = window_records(root, windows=[("S-Vw04", w)])
    assert records[0]["start_t"] == 10.0
    assert records[0]["rejected"] == ["turn too close"]
    save_windows(records, tmp_path / "windows.json")
    assert load_window_records(tmp_path / "windows.json") == records


def _sample(n=1000, fs=100.0):
    t = np.arange(n, dtype=np.float64) / fs
    gnss = np.zeros(n, dtype=GNSS_DTYPE)
    gnss["t"] = t
    gnss["lat_deg"] = 52.0 + 1e-5 * t
    gnss["lon_deg"] = -0.75
    gnss["accuracy_m"] = 5.0
    return IOVNBDSample(
        t=t, gyro=np.zeros((n, 3)), mag=np.zeros((n, 3)),
        accel=np.tile([0.0, 0.0, 9.80665], (n, 1)),
        gnss=gnss,
        gt_pose={"t": t.copy(), "x": t.copy(), "y": np.zeros(n),
                 "z": np.zeros(n), "qx": np.zeros(n), "qy": np.zeros(n),
                 "qz": np.zeros(n), "qw": np.ones(n), "vx": np.ones(n),
                 "vy": np.zeros(n), "vz": np.zeros(n)})


def test_rebuild_window_masks_gnss_and_keeps_gt():
    """Masked interval: every GNSS field NaN except t; gt_pose intact."""
    sess = _sample()
    record = {"session": "S-Vw04", "start_t": 3.0, "end_t": 6.0,
              "duration_s": 3.0, "rejected": []}
    w = rebuild_window(sess, record)
    inside = (sess.t >= 3.0) & (sess.t < 6.0)
    outside = ~inside
    for f in GNSS_DTYPE.names:
        if f == "t":
            assert np.allclose(w.masked.gnss[f], sess.gnss[f])
        else:
            assert np.all(np.isnan(w.masked.gnss[f][inside]))
            assert np.all(np.isfinite(w.masked.gnss[f][outside]))
    for f in sess.gt_pose:
        assert np.array_equal(w.masked.gt_pose[f], sess.gt_pose[f])
    assert w.duration_s == 3.0 and w.start_t == 3.0 and w.end_t == 6.0
