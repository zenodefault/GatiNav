import numpy as np

from python.eval.physics_audit import cross_correlation_alignment, drift_scaling
from python.io.iovnb_loader import GT_POSE_DTYPE, GNSS_DTYPE, IOVNBDSample


def _session():
    t = np.arange(4000, dtype=np.float64) / 100.0
    gt = np.zeros(t.size, dtype=GT_POSE_DTYPE)
    gt["t"] = t
    gnss = np.zeros(t.size, dtype=GNSS_DTYPE)
    gnss["t"] = t
    return IOVNBDSample(
        t=t, gyro=np.zeros((t.size, 3)),
        accel=np.tile([0.0, 0.0, 9.80665], (t.size, 1)),
        mag=np.tile([1.0, 0.0, 0.0], (t.size, 1)),
        gnss=gnss, gt_pose=gt)


def test_drift_scaling_returns_requested_windows():
    result = drift_scaling(_session())
    np.testing.assert_allclose(result["durations_s"], [5, 10, 20, 30])
    assert result["drifts_m"].shape == (4,)


def test_cross_correlation_reports_zero_lag():
    session = _session()
    session.gt_pose["vx"] = np.sin(session.t)
    session.accel[:, 0] = np.sin(session.t)
    result = cross_correlation_alignment(session)
    assert abs(result["lag_seconds"]) <= 0.01
