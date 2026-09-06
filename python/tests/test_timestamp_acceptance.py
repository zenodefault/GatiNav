from pathlib import Path

import numpy as np
import pytest

from python.eval.physics_audit import _integrate, _static_window
from python.eval.engine import session_pairs
from python.io.iovnb_loader import load_pair


ROOT = Path(__file__).resolve().parents[2] / "data/IO-VNBD/Synchronised V abd S datasets"


def _sessions():
    out = []
    for s_path, v_path in session_pairs(ROOT):
        try:
            out.append(load_pair(v_path, s_path))
        except (OSError, ValueError):
            continue
    if not out:
        pytest.fail("no real synchronized sessions available")
    return out


def test_stationary_30_second_drift_is_below_two_metres():
    for session in _sessions():
        try:
            start, _ = _static_window(session, minimum_s=30.0)
        except ValueError:
            continue
        count = int(round(30.0 / np.median(np.diff(session.t)))) + 1
        drift = np.linalg.norm(_integrate(session, start, count)[-1])
        assert drift < 2.0
        return
    pytest.fail("no real 30-second stationary segment available")


def test_driving_30_second_drift_is_below_one_hundred_metres():
    for session in _sessions():
        speed = np.hypot(session.gt_pose["vx"], session.gt_pose["vy"])
        moving = np.flatnonzero(speed > 2.0)
        if not moving.size:
            continue
        start = int(moving[0])
        count = int(round(30.0 / np.median(np.diff(session.t)))) + 1
        if start + count > len(session.t):
            continue
        drift = np.linalg.norm(_integrate(session, start, count)[-1])
        assert drift < 100.0
        return
    pytest.fail("no real 30-second driving segment available")
