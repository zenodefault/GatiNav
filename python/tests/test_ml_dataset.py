import numpy as np
import pytest

from python.ekf.ekf import ErrorStateEKF
from python.ekf.rotations import rotation_matrix_to_quaternion
from python.io.iovnb_loader import IOVNBDSample
from python.ml.dataset import (Session, _vehicle_pair, build_leave_one_driver_out,
                               denormalize, discover_sessions)


def _session(driver, segment, n=1000):
    t = np.arange(n, dtype=np.float64) / 100.0
    zeros = np.zeros((n, 3))
    gnss = np.zeros(n, dtype=[("t", "f8")])
    gt = np.zeros(n, dtype=[("t", "f8")])
    sample = IOVNBDSample(t, zeros + 1.0, zeros + 2.0, zeros, gnss, gt)
    return Session(driver, segment, sample)


def test_leave_one_driver_out_counts_and_no_leakage(tmp_path):
    sessions = [_session(f"D{i}", f"S{i}") for i in range(3)]
    folds = build_leave_one_driver_out(sessions, 1.0, 0.5, tmp_path)
    assert set(folds) == {"D0", "D1", "D2"}
    for test_driver, fold in folds.items():
        assert test_driver not in set(fold["train"][0].driver)
        assert fold["validation_driver"] not in set(fold["train"][0].driver)
        assert set(fold["test"][0].driver) == {test_driver}


def test_vehicle_pair_case_insensitive_fallback(tmp_path):
    """Vta/Vtb urban categories name vehicle files V-vta*.csv (lowercase v)."""
    s = tmp_path / "S-Vta2.csv"
    v = tmp_path / "V-vta2.csv"
    s.write_text("")
    v.write_text("")
    assert _vehicle_pair(s) == v
    # strict naming still wins when present
    strict = tmp_path / "V-Vta2.csv"
    strict.write_text("")
    assert _vehicle_pair(s) == strict
    # no pair at all -> None
    (tmp_path / "V-vta99.csv").write_text("")
    orphan = tmp_path / "S-Vta9.csv"
    orphan.write_text("")
    assert _vehicle_pair(orphan) is None


def test_discover_sessions_finds_case_mismatched_pairs(tmp_path, monkeypatch):
    from python.ml import dataset as ds
    root = tmp_path / "Categorised IOVNB Dataset" / "Vta (Driver E)" / "Vta02"
    root.mkdir(parents=True)
    (root / "S-Vta2.csv").write_text("")
    (root / "V-vta2.csv").write_text("")
    calls = []

    def fake_load(v_path, s_path):
        calls.append((v_path.name, s_path.name))
        return object()

    monkeypatch.setattr(ds, "load_pair", fake_load)
    monkeypatch.setattr(ds, "resample_to_common_clock", lambda sample: sample)
    sessions = discover_sessions(tmp_path)
    assert len(sessions) == 1
    assert calls == [("V-vta2.csv", "S-Vta2.csv")]
    assert sessions[0].driver == "Driver E"
    assert sessions[0].segment == "Vta2"


def test_window_shapes_and_normalization_reversibility(tmp_path):
    folds = build_leave_one_driver_out(
        [_session(f"D{i}", f"S{i}") for i in range(3)], 1.0, 0.5, tmp_path
    )
    fold = folds["D0"]
    raw, normalized = fold["train"]
    assert raw.x.shape == (19, 100, 6)
    assert np.array_equal(raw.x[0, 0], [2.0, 2.0, 2.0, 1.0, 1.0, 1.0])
    assert normalized.shape == raw.x.shape
    restored = denormalize(normalized, fold["mean"], fold["std"])
    assert np.allclose(restored, raw.x, atol=1e-12)
    cached = np.load(tmp_path / "D0.npz")
    assert np.allclose(cached["mean"], fold["mean"])
