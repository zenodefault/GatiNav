"""Train NoiseNet on real IO-VNBD windows (specs/03 section 4).

Targets: per-window ZUPT measurement-variance, conditioned on vehicle speed
(ground truth used ONLY as a training label, never as a filter input):

    sigma(v) = 0.3 + 2.0 * v   (m/s),  target = sigma^2 per axis

At standstill the network must output a tight ZUPT covariance (the zero-
velocity pseudo-measurement is trustworthy); while moving it must output a
large one, effectively disabling ZUPT so the filter is never dragged to zero
velocity on a moving vehicle. Loss is the specs/03 log-variance objective;
folds are leave-one-driver-out with train-only normalization.

Deploys the fold model with the lowest validation drift to
python/ml/weights/noisenet.pt and writes results/audit/cnn_training.md.
"""

from dataclasses import dataclass
from pathlib import Path
import json
import time

import numpy as np
import torch

from python.ml.dataset import discover_sessions
from python.ml.train import log_variance_loss, seed_everything, \
    train_noise_net

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data/IO-VNBD/Synchronised V abd S datasets"
WEIGHTS = ROOT / "python/ml/weights/noisenet.pt"
REPORT = ROOT / "results/audit/cnn_training.md"

WINDOW_S = 1.0
STRIDE_S = 5.0  # training-efficiency stride (specs/03 uses 0.5 s at inference)
SIGMA_0 = 0.3  # m/s ZUPT sigma at standstill
SIGMA_PER_MPS = 2.0  # extra sigma per m/s of vehicle speed
MAX_WINDOWS_PER_SESSION = 3000


def _speed_target(sample, center_t):
    """Vehicle speed at window center via gt_pose interpolation (label only)."""
    t = sample.gt_pose["t"]
    speed = np.hypot(sample.gt_pose["vx"], sample.gt_pose["vy"])
    v = float(np.interp(center_t, t, speed))
    sigma = SIGMA_0 + SIGMA_PER_MPS * v
    return sigma ** 2


def build_training_windows(sessions):
    """(x, y, driver) arrays: 1 s windows, stride STRIDE_S, capped per session."""
    xs, ys, drivers = [], [], []
    for session in sessions:
        sample = session.sample
        n_win = int(round(WINDOW_S * 100.0))
        stride = int(round(STRIDE_S * 100.0))
        signal = np.column_stack((sample.accel, sample.gyro))
        count = 0
        for lo in range(0, signal.shape[0] - n_win + 1, stride):
            if count >= MAX_WINDOWS_PER_SESSION:
                break
            center = sample.t[lo + n_win // 2]
            xs.append(signal[lo:lo + n_win].T)  # (6, 100) channel-first
            ys.append(_speed_target(sample, center))
            drivers.append(session.driver)
            count += 1
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64), \
        np.asarray(drivers)


def run(sessions=None, out_weights=WEIGHTS, out_report=REPORT, seed=0,
        epochs=60):
    t0 = time.time()
    if sessions is None:
        sessions = discover_sessions(DATA_ROOT)
    x, y, drivers = build_training_windows(sessions)
    unique = sorted(set(drivers))
    if len(unique) < 3:
        raise ValueError("at least three drivers are required")
    print(f"sessions={len(sessions)} windows={x.shape[0]} drivers={unique}")

    # leave-one-driver-out: next driver is validation (specs/03 section 4)
    def _stats(sel):
        mean = x[sel].reshape(-1, 6).mean(axis=0)
        std = x[sel].reshape(-1, 6).std(axis=0)
        std = np.where(std == 0.0, 1.0, std)
        return mean.astype(np.float64), std.astype(np.float64)

    fold_results = []
    best = None
    for i, test_driver in enumerate(unique):
        val_driver = unique[(i + 1) % len(unique)]
        train = (drivers != test_driver) & (drivers != val_driver)
        val = drivers == val_driver
        test = drivers == test_driver
        mean, std = _stats(train)
        train_x = (x[train] - mean) / std
        val_x = (x[val] - mean) / std
        test_x = (x[test] - mean) / std
        checkpoint = out_weights.with_name(
            f"noisenet_fold_{test_driver}.pt")
        model, history, path = train_noise_net(
            train_x, y[train], val_x, y[val], checkpoint, seed=seed,
            epochs=epochs, patience=12)
        with torch.no_grad():
            test_drift = float(log_variance_loss(model(test_x), y[test]))
        result = {"test_driver": test_driver, "val_driver": val_driver,
                  "checkpoint": str(path),
                  "validation_drift": float(history["validation"][-1]),
                  "test_drift": test_drift,
                  "windows": {"train": int(train.sum()), "val": int(val.sum()),
                              "test": int(test.sum())},
                  "mean": mean.tolist(), "std": std.tolist()}
        fold_results.append(result)
        print(f"  fold {test_driver}: val={result['validation_drift']:.4f} "
              f"test={test_drift:.4f} (train={int(train.sum())}, "
              f"val={int(val.sum())}, test={int(test.sum())})")
        if best is None or result["validation_drift"] < best[
                "validation_drift"]:
            best = result

    state = torch.load(best["checkpoint"], map_location="cpu",
                       weights_only=True)
    state["mean"] = best["mean"]
    state["std"] = best["std"]
    out_weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, out_weights)

    report = {
        "target": f"sigma = {SIGMA_0} + {SIGMA_PER_MPS} * v m/s, "
                  "target = sigma^2 per axis (GT speed is a training label)",
        "window_s": WINDOW_S, "stride_s": STRIDE_S,
        "sessions": len(sessions), "windows": int(x.shape[0]),
        "deployed": best["checkpoint"],
        "deployed_driver": best["test_driver"],
        "folds": fold_results, "elapsed_s": round(time.time() - t0, 1),
    }
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"deployed {best['checkpoint']} -> {out_weights}")
    print(f"elapsed {report['elapsed_s']}s; report: {out_report}")
    return report


if __name__ == "__main__":
    seed_everything(0)
    run()