"""Outage segmentation (specs/01 section 6).

carve_outages(session, outage_s) builds artificial GNSS outage windows of
exactly 30/60/90 s on a uniform-clock (resampled) session. For every accepted
window it returns an OutageWindow: a copy of the session whose GNSS fields are
masked (NaN) strictly inside the interval - no interpolation inside an outage
(§6.3) - with gt_pose left intact for scoring only (§6.4), the interval
bounds, and the rejected-candidate reasons (§6.6).

Boundary samples must be at least TURN_FREE_MARGIN_S away from any turn,
where a turn is |heading change| > TURN_THRESHOLD_DEG over +/- TURN_WINDOW_S
(§6.5, heading from the gt_pose ENU yaw). Candidates need DATA_MARGIN_S of
data before and after the window (§6.1). All thresholds are VERIFY per §6.
"""

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from python.io.iovnb_loader import IOVNBDSample

TURN_THRESHOLD_DEG = 5.0
TURN_WINDOW_S = 3.0
TURN_FREE_MARGIN_S = 3.0
DATA_MARGIN_S = 5.0
CANDIDATE_STEP_S = 1.0


@dataclass
class OutageWindow:
    start_t: float  # outage start timestamp (s)
    end_t: float  # outage end timestamp (s), start_t + duration
    duration_s: float
    masked: IOVNBDSample  # session copy, gnss NaN inside [start_t, end_t)
    rejected: list  # (start_t, reason) candidates skipped before acceptance


def _yaw_enu(sample):
    qw = sample.gt_pose["qw"]
    qz = sample.gt_pose["qz"]
    return 2.0 * np.arctan2(qz, qw)


def _wrap_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _heading_change(yaw, dt, win_s):
    """|heading change| over +/- win_s around every sample (radians)."""
    w = max(1, int(round(win_s / dt)))
    n = yaw.size
    out = np.zeros(n)
    for i in range(n):
        lo, hi = max(0, i - w), min(n - 1, i + w)
        out[i] = abs(_wrap_pi(yaw[hi] - yaw[lo]))
    return out


def _mask_gnss(session, i0, i1):
    gnss = session.gnss.copy()
    for f in gnss.dtype.names:
        if f != "t":
            gnss[f][i0:i1] = np.nan
    return IOVNBDSample(t=session.t.copy(), gyro=session.gyro.copy(),
                        accel=session.accel.copy(), mag=session.mag.copy(),
                        gnss=gnss, gt_pose=session.gt_pose.copy())


def carve_outages(session, outage_s):
    """Return accepted GNSS outage windows of exactly outage_s seconds."""
    outage_s = float(outage_s)
    if outage_s not in (30.0, 60.0, 90.0):
        raise ValueError(f"outage_s must be in {{30, 60, 90}}, got {outage_s}")
    t = session.t
    n = t.size
    if n < 2:
        return []
    dt = float(np.median(np.diff(t)))
    n_out = int(round(outage_s / dt))
    d_idx = int(round(DATA_MARGIN_S / dt))
    if n < n_out + 2 * d_idx:
        return []

    yaw = _yaw_enu(session)
    change = _heading_change(yaw, dt, TURN_WINDOW_S)
    m_idx = int(round(TURN_FREE_MARGIN_S / dt))
    turn_free = np.ones(n, dtype=bool)
    if 2 * m_idx + 1 <= n:
        turn_free[m_idx:n - m_idx] = (
            sliding_window_view(change, 2 * m_idx + 1).max(axis=1)
            <= np.deg2rad(TURN_THRESHOLD_DEG)
        )

    step = max(1, int(round(CANDIDATE_STEP_S / dt)))
    windows = []
    rejected = []
    start_idx = d_idx
    while start_idx + n_out <= n - d_idx:
        end_idx = start_idx + n_out
        reasons = []
        if not turn_free[start_idx]:
            reasons.append("start mid-turn (|heading change| > 5 deg "
                           "within +/- 3 s)")
        if not turn_free[end_idx - 1]:
            reasons.append("end mid-turn (|heading change| > 5 deg "
                           "within +/- 3 s)")
        if reasons:
            rejected.append((float(t[start_idx]), "; ".join(reasons)))
            start_idx += step
            continue
        windows.append(OutageWindow(
            start_t=float(t[start_idx]),
            end_t=float(t[start_idx]) + outage_s,
            duration_s=outage_s,
            masked=_mask_gnss(session, start_idx, end_idx),
            rejected=list(rejected),
        ))
        rejected = []
        start_idx = end_idx
    return windows