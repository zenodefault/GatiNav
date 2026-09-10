"""Driving-scenario detection (Blueprint plan 3.3).

Detects structured scenarios — tunnels and highways — from GNSS-signal
history and the speed profile, so the engine can engage stronger NHC
pseudo-measurements there (process noise up, NHC measurement noise down).

  tunnel : GNSS is lost while the vehicle keeps moving at speed for a
           sustained stretch (a blackout in an otherwise-moving trip)
  highway: sustained high speed with healthy GNSS
  urban  : everything else (low speed, stop-start, idle)

Pure signal logic; no I/O. The engine consumes ``label``/``is_structured``
to decide how aggressively to constrain the filter.
"""

import numpy as np

TUNNEL_MIN_LOSS_S = 5.0   # GNSS-loss stretch long enough to call a tunnel
TUNNEL_MIN_SPEED = 2.0    # vehicle must still be moving (m/s)
HIGHWAY_SPEED = 18.0      # sustained speed to call a highway (m/s ~ 65 km/h)
HIGHWAY_MIN_S = 30.0      # sustained for this long


def detect_scenario(t, speed, gnss_valid, dt=None,
                    tunnel_min_loss_s=TUNNEL_MIN_LOSS_S,
                    tunnel_min_speed=TUNNEL_MIN_SPEED,
                    highway_speed=HIGHWAY_SPEED,
                    highway_min_s=HIGHWAY_MIN_S):
    """Per-sample scenario labels over a uniform-clock stream.

    ``t`` (N,), ``speed`` (N,) m/s (GNSS-derived or wheel), ``gnss_valid``
    (N,) boolean. Returns (labels, is_structured): labels is an array of
    "tunnel"/"highway"/"urban" strings; is_structured is the boolean mask
    of tunnel/highway samples.
    """
    t = np.asarray(t, dtype=np.float64)
    speed = np.asarray(speed, dtype=np.float64)
    gnss_valid = np.asarray(gnss_valid, dtype=bool)
    if not (t.ndim == 1 and speed.shape == t.shape
            and gnss_valid.shape == t.shape):
        raise ValueError("t, speed, gnss_valid must be equal-length 1-D")
    if t.size == 0:
        return np.array([], dtype=object), np.zeros(0, dtype=bool)
    if dt is None:
        dt = float(np.median(np.diff(t)))
    if dt <= 0:
        raise ValueError("dt must be positive")

    n = t.size
    labels = np.full(n, "urban", dtype=object)

    # --- tunnel: GNSS lost while moving, for >= tunnel_min_loss_s --------
    loss_starts = np.flatnonzero(
        (~gnss_valid) & np.concatenate(([True], gnss_valid[:-1])))
    loss_ends = np.flatnonzero(
        gnss_valid & np.concatenate(([False], ~gnss_valid[:-1])))
    if not gnss_valid[0]:
        loss_starts = np.concatenate(([0], loss_starts))
    if not gnss_valid[-1]:
        loss_ends = np.concatenate((loss_ends, [n]))
    for start, end in zip(loss_starts, loss_ends):
        if end - start < int(round(tunnel_min_loss_s / dt)):
            continue
        moving = np.mean(speed[start:end] > tunnel_min_speed)
        if moving >= 0.5:  # majority of the loss stretch at speed
            labels[start:end] = "tunnel"

    # --- highway: sustained speed with healthy GNSS ----------------------
    moving = speed > highway_speed
    run_n = int(round(highway_min_s / dt))
    if run_n > 0:
        kernel = np.ones(run_n, dtype=int)
        count = np.convolve(moving.astype(int), kernel, "same")
        # interior of sustained-speed stretches with healthy GNSS; tunnels
        # (already labelled) always win, and never get extended across
        highway = ((count >= run_n) & gnss_valid & (labels == "urban"))
        labels[highway] = "highway"
        # extend each qualifying run one run_n either side, but only over
        # samples that are NOT a tunnel blackout
        idx = np.flatnonzero(highway)
        if idx.size:
            lo = max(0, int(idx.min()) - run_n)
            hi = min(n, int(idx.max()) + run_n)
            extend = (labels[lo:hi] == "urban") & (speed[lo:hi] > 0.5)
            labels[lo:hi][extend] = "highway"

    structured = (labels == "tunnel") | (labels == "highway")
    return labels, structured