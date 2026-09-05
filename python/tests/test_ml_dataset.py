import numpy as np

from python.ekf.ekf import ErrorStateEKF
from python.ekf.rotations import rotation_matrix_to_quaternion
from python.io.iovnb_loader import IOVNBDSample
from python.ml.dataset import Session, build_leave_one_driver_out, denormalize


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
