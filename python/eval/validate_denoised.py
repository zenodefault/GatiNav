"""Isolation validation of the denoised odometry chain (Blueprint Phase 1.3).

Runs the plan's mandated validation step before the denoised odometry is
reintegrated into the full EKF: simulate a GNSS outage and compare the
drift of (1) the new denoised odometry against (2) raw double integration
and (3) the old CNN-based speed estimator.

For each discovered outage window the three chains are propagated open-loop
(same seeding, same AHRS attitude, GNSS never re-enters):

  raw      : double-integrate attitude-compensated forward accel
  denoised : same but a_denoised = a_measured - residual_net(window)
  cnn      : integrate the old NoiseNet speed head (log(1+v) -> expm1)

Output is a per-window drift/distance table plus a gate verdict:
denoised must beat raw on median drift for the chain to be considered
viable for EKF reintegration.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from python.calibration.ahrs import Ahrs
from python.eval.engine import find_windows
from python.ml.denoise.dataset import FS, WINDOW_S
from python.ml.denoise.model import ResidualTCN
from python.ml.train import load_checkpoint

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data/IO-VNBD/Synchronised V abd S datasets"
WEIGHTS = ROOT / "python/ml/weights/denoise.pt"
CNN_WEIGHTS = ROOT / "python/ml/weights/noisenet.pt"
STRIDE_S = 0.5


def _load_denoiser(path=WEIGHTS):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = ResidualTCN()
    model.load_state_dict(state["model"])
    model.eval()
    return model, float(state["mean"]), float(state["std"])


def _residual_at(model, mean, std, window):
    """Residual for the current forward-accel window (causal)."""
    x = torch.from_numpy(((window - mean) / std).astype(np.float32))
    with torch.no_grad():
        return float(model(x[None, None, :])[0, 0])


def _forward_accel_run(session, i_start):
    """AHRS attitude-compensated forward accel from i_start onward."""
    ahrs = Ahrs()
    init_n = min(max(int(round(5.0 * FS)), 5), session.t.size)
    ahrs.init_static(
        session.accel[:init_n],
        session.mag[:init_n],
        gyro=session.gyro[:init_n],
    )
    out = []
    for i in range(i_start, session.t.size):
        ahrs.update(
            session.gyro[i],
            session.accel[i],
            session.mag[i],
            1.0 / FS,
        )
        out.append(ahrs.forward_accel(session.accel[i]))
    return np.asarray(out)


def _chain_forward(
    session,
    i0,
    i1,
    forward,
    residual_fn=None,
    stride=int(round(STRIDE_S * FS)),
    window_n=int(round(WINDOW_S * FS)),
):
    """Integrate forward accel over [i0, i1) -> (displacement, drift).

    residual_fn(window) -> scalar applied as a_denoised = a - residual at
    the window END sample. Velocity starts at 0 (the open-loop drift
    comparison cares about the *acceleration* chain; both chains share the
    same initial velocity).
    """
    seg = forward[i0:i1]
    if residual_fn is not None:
        corrected = seg.copy()
        for k in range(0, seg.size, stride):
            lo = max(0, k - window_n)
            window = seg[lo:k] if k > lo else seg[k : k + 1]
            if window.size < 2:
                continue
            r = residual_fn(window)
            for j in range(k, min(k + stride, seg.size)):
                corrected[j] = seg[j] - r
        seg = corrected
    v = np.cumsum(seg) / FS
    disp = np.cumsum(v) / FS
    return float(disp[-1])


def _cnn_speed_integral(session, i0, i1, model):
    """Integrate the old NoiseNet speed head over the outage interval."""
    from python.ekf.ekf_cnn import CNNEKF

    wrapper = CNNEKF(model=model)
    mean, std = wrapper.mean, wrapper.std
    speed_sum = 0.0
    n = 0
    for i in range(i0, i1):
        if i < 99:
            continue
        window = np.concatenate(
            (session.accel[i - 99 : i + 1], session.gyro[i - 99 : i + 1]),
            axis=1,
        )
        window = (window - mean) / std
        with torch.no_grad():
            out = model(torch.from_numpy(window.T[None]).float()).numpy()[0]
        speed = float(np.expm1(out[3]))  # speed head: log(1 + v)
        speed_sum += speed
        n += 1
    return speed_sum / max(n, 1) / FS  # meters travelled (approx)


def validate(
    data_root=DATA_ROOT, denoise_path=WEIGHTS, cnn_path=CNN_WEIGHTS, max_windows=None
):
    windows = find_windows(data_root)
    if max_windows:
        windows = windows[:max_windows]
    denoiser = None
    if Path(denoise_path).is_file():
        denoiser = _load_denoiser(denoise_path)
    cnn_model = None
    if Path(cnn_path).is_file():
        from python.ml.cnn import NoiseNet

        cnn_model = NoiseNet()
        load_checkpoint(cnn_model, cnn_path)
        cnn_model.eval()
    if denoiser is None and cnn_model is None:
        print("no models found; only raw chain will run")
    rows = []
    for session_name, window in windows:
        sess = window.masked
        i0 = int(np.searchsorted(sess.t, window.start_t))
        i1 = int(np.searchsorted(sess.t, window.end_t))
        forward = _forward_accel_run(sess, i0)
        gt_disp = np.hypot(
            sess.gt_pose["x"][i1 - 1] - sess.gt_pose["x"][i0],
            sess.gt_pose["y"][i1 - 1] - sess.gt_pose["y"][i0],
        )
        raw = _chain_forward(sess, i0, i1, forward)
        denoised = None
        if denoiser is not None:
            model, mean, std = denoiser
            denoised = _chain_forward(
                sess,
                i0,
                i1,
                forward,
                residual_fn=lambda w: _residual_at(model, mean, std, w),
            )
        cnn_disp = None
        if cnn_model is not None:
            cnn_disp = _cnn_speed_integral(sess, i0, i1, cnn_model)
        rows.append((session_name, window.duration_s, gt_disp, raw, denoised, cnn_disp))
    print(
        f"{'window':<10} {'dur':>4} {'GT m':>8} {'raw m':>8} "
        f"{'denoised m':>10} {'cnn m':>8}"
    )
    raw_drifts, den_drifts = [], []
    for name, dur, gt, raw, den, cnn in rows:
        print(
            f"{name:<10} {dur:>4.0f} {gt:>8.1f} {raw:>8.1f} "
            f"{(den if den is not None else float('nan')):>10.1f} "
            f"{(cnn if cnn is not None else float('nan')):>8.1f}"
        )
        raw_drifts.append(abs(raw - gt))
        if den is not None:
            den_drifts.append(abs(den - gt))
    if raw_drifts:
        print(f"\nmedian |drift - GT|: raw {np.median(raw_drifts):.2f} m")
    if den_drifts:
        print(f"median |drift - GT|: denoised {np.median(den_drifts):.2f} m")
        verdict = np.median(den_drifts) < np.median(raw_drifts) if raw_drifts else False
        print(f"GATE (denoised < raw on median drift): {'PASS' if verdict else 'FAIL'}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-windows", type=int, default=None)
    args = parser.parse_args()
    validate(max_windows=args.max_windows)
