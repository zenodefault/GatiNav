from pathlib import Path

import numpy as np
import pytest

from python.eval.engine import _stationary_mask, session_pairs
from python.io.iovnb_loader import load_pair


ROOT = Path(__file__).resolve().parents[2] / \
    "data/IO-VNBD/Synchronised V abd S datasets"
CALIBRATION = ("S-Vw15.csv", "S-Vta8.csv", "S-Vta2.csv",
               "S-Vw14c.csv", "S-Vw1.csv")
DRIVING = ("S-Vw16b.csv", "S-Vw6.csv", "S-Vw10.csv",
           "S-Vtb12.csv", "S-Vta11.csv")


def _named_sessions(names):
    pairs = {s.name: (s, v) for s, v in session_pairs(ROOT)}
    out = []
    for name in names:
        if name not in pairs:
            pytest.fail(f"calibration session missing: {name}")
        s, v = pairs[name]
        out.append(load_pair(v, s))
    return out


def _stationary_fraction(session, expected_stationary):
    speed = np.interp(
        session.t, session.gt_pose["t"],
        np.hypot(session.gt_pose["vx"], session.gt_pose["vy"]))
    labels = speed < 0.5 if expected_stationary else speed > 2.0
    mask = _stationary_mask(session.accel, session.gyro)
    indices = np.flatnonzero(labels & (np.arange(labels.size) >= 99))
    if not indices.size:
        pytest.fail("labeled calibration segment is empty")
    cuts = np.flatnonzero(np.diff(indices) > 1)
    starts = np.r_[0, cuts + 1]
    ends = np.r_[cuts + 1, indices.size]
    runs = [(indices[a], indices[b - 1] + 1)
            for a, b in zip(starts, ends) if b - a >= 100]
    run = max(runs, key=lambda pair: pair[1] - pair[0])
    usable = np.zeros(labels.size, dtype=bool)
    usable[run[0]:run[1]] = True
    return float(np.mean(mask[usable]))


def test_held_out_stationary_segments_fire_at_least_95_percent():
    rates = [_stationary_fraction(s, True) for s in _named_sessions(CALIBRATION)]
    assert min(rates) >= 0.95


def test_held_out_driving_segments_fire_below_5_percent():
    rates = [_stationary_fraction(s, False) for s in _named_sessions(DRIVING)]
    assert max(rates) < 0.05
