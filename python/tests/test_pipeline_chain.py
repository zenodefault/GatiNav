"""End-to-end chain test: loader (B1) -> resample (B2) -> preprocessing (B3)
-> outage segmentation (B4), verifying the stages link on real and synthetic
streams."""

import numpy as np

from python.io.iovnb_loader import (GNSS_DTYPE, GT_POSE_DTYPE, IOVNBDSample,
                                    load_pair, resample_to_common_clock)
from python.io.outages import carve_outages
from python.io.preprocessing import gravity_remove, window_segmentation
from test_iovnbd_loader import shortest_synchronised_pair


def test_chain_on_real_pair():
    s_path, v_path = shortest_synchronised_pair()
    raw = load_pair(v_path, s_path)
    rs = resample_to_common_clock(raw, fs=100.0)  # B2
    duration = raw.t[-1] - raw.t[0]
    assert abs(rs.t.size - duration * 100.0) <= 1.0

    linear = gravity_remove(rs.accel, fs=100.0)  # B3
    assert linear.shape == rs.accel.shape
    assert np.all(np.isfinite(linear))
    pre = IOVNBDSample(t=rs.t, gyro=rs.gyro, accel=linear, mag=rs.mag,
                       gnss=rs.gnss, gt_pose=rs.gt_pose)
    wins = window_segmentation(pre, window_s=5.0, stride_s=2.5)
    assert wins
    for w in wins[:3]:
        assert w.t.size == int(round(5.0 * 100.0))
        assert w.gyro.shape == w.accel.shape == (w.t.size, 3)
        assert w.gnss.shape[0] == w.gt_pose.shape[0] == w.t.size

    assert carve_outages(rs, 30.0) == []


def test_outages_after_resample_on_synthetic():
    # Synthetic straight 120 s session on a 100 Hz clock feeds carve_outages.
    t = np.arange(0.0, 120.0, 0.01)
    n = t.size
    gnss = np.zeros(n, dtype=GNSS_DTYPE)
    gnss["t"] = t
    gnss["lat_deg"] = 52.27
    gnss["lon_deg"] = -2.12
    gt = np.zeros(n, dtype=GT_POSE_DTYPE)
    gt["t"] = t
    gt["vx"] = 10.0
    gt["qw"] = 1.0  # yaw = 0 (straight)
    sess = IOVNBDSample(t=t, gyro=np.zeros((n, 3)),
                        accel=np.tile([0.0, 0.0, 9.80665], (n, 1)),
                        mag=np.tile([1.0, 0.0, 0.0], (n, 1)),
                        gnss=gnss, gt_pose=gt)
    rs = resample_to_common_clock(sess, fs=100.0)  # B2 (uniform already)
    windows = carve_outages(rs, 60.0)  # B4
    assert windows
    assert windows[0].duration_s == 60.0
    inside = (rs.t >= windows[0].start_t) & (rs.t < windows[0].end_t)
    assert np.all(np.isnan(windows[0].masked.gnss["lat_deg"][inside]))
    assert np.all(np.isfinite(windows[0].masked.gnss["lat_deg"][~inside]))