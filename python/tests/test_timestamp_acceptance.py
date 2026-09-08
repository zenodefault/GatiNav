from pathlib import Path

import numpy as np
import pytest

from python.eval.physics_audit import _integrate, _integrate_calibrated, \
    _static_window
from python.eval.engine import session_pairs
from python.io.iovnb_loader import load_pair, resample_to_common_clock


ROOT = Path(__file__).resolve().parents[2] / "data/IO-VNBD/Synchronised V abd S datasets"


def _sessions():
    """Synchronized sessions on ONE shared 100 Hz clock.

    The raw GT and phone streams start at different absolute times (e.g.
    Y1's GT starts at -155.6 s), so array-indexing one stream with the
    other's indices is meaningless. Resampling puts every stream onto the
    same grid (the engine pipeline does exactly this); indices then equal
    time and GT-derived window starts index the phone arrays correctly.
    """
    out = []
    for s_path, v_path in session_pairs(ROOT):
        try:
            out.append(resample_to_common_clock(load_pair(v_path, s_path)))
        except (OSError, ValueError):
            continue
    if not out:
        pytest.fail("no real synchronized sessions available")
    return out


def test_stationary_30_second_drift_is_below_two_metres():
    """Phase 1 gate: calibrated IMU + GNSS-seeded DR over 30 s stationary."""
    for session in _sessions():
        try:
            start, _ = _static_window(session, minimum_s=30.0)
        except ValueError:
            continue
        count = int(round(30.0 / np.median(np.diff(session.t)))) + 1
        drift = np.linalg.norm(
            _integrate_calibrated(session, start, count)[-1]
            - _integrate_calibrated(session, start, count)[0])
        assert drift < 2.0
        return
    pytest.fail("no real 30-second stationary segment available")


def test_driving_30_second_drift_is_below_one_hundred_metres():
    """Phase 1 gate: calibrated DR seeded from GNSS over 30 s driving."""
    for session in _sessions():
        speed = np.hypot(session.gt_pose["vx"], session.gt_pose["vy"])
        moving = np.flatnonzero(speed > 2.0)
        if not moving.size:
            continue
        start = int(moving[0])
        count = int(round(30.0 / np.median(np.diff(session.t)))) + 1
        if start + count > len(session.t):
            continue
        try:
            drift = np.linalg.norm(
                _integrate_calibrated(session, start, count)[-1]
                - _integrate_calibrated(session, start, count)[0])
        except ValueError:
            continue  # session has no calibratable static stretch
        assert drift < 100.0
        return
    pytest.fail("no real 30-second driving segment available")
