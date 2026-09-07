"""Train NoiseNet on real IO-VNBD windows (specs/03 section 4).

Targets: per-window ZUPT measurement-variance. A regression from a 1 s IMU
window to continuous vehicle speed is not identifiable on real data (window
vibration correlates with speed at log-r <= 0.33 and stopped/moving feature
clouds overlap), so the network collapses onto the marginal mean and ZUPT
degenerates. Instead the task is restricted to the feature-consistent
extremes that deployment actually exercises -- the filter only queries the
network on windows the variance detector accepts as stationary -- and the
vehicle-speed ground truth is used ONLY as a training label:

    quiet window + GT stopped  ->  sigma = 0.3  m/s  (tight ZUPT)
    busy  window + GT driving  ->  sigma = 20   m/s  (ZUPT disabled)

Windows in the unidentifiable middle (slow creep, or vibration inconsistent
with the motion label) are dropped from training. Loss is the specs/03
log-variance objective; folds are leave-one-driver-out with train-only
normalization.

Deploys the fold model with the lowest validation drift to
python/ml/weights/noisenet.pt and writes results/audit/cnn_training.md.
"""

from pathlib import Path
import json
import time

import numpy as np
import torch

from python.ml.cnn import NoiseNet
from python.ml.dataset import discover_sessions
from python.ml.train import log_variance_loss, seed_everything, \
    train_noise_net
from python.eval.engine import ACCEL_VAR_MAX, GYRO_VAR_MAX

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data/IO-VNBD/Synchronised V abd S datasets"
WEIGHTS = ROOT / "python/ml/weights/noisenet.pt"
REPORT = ROOT / "results/audit/cnn_training.md"
WINDOWS_CACHE = ROOT / "results/audit/cnn_training_windows.npz"

WINDOW_S = 1.0
STRIDE_S = 5.0  # training-efficiency stride (specs/03 uses 0.5 s at inference)
SIGMA_STOP = 0.3  # m/s tight ZUPT sigma for genuinely stationary windows
SIGMA_DRIVE = 20.0  # m/s loose sigma, effectively disables ZUPT while moving
V_STOP_MAX = 0.5  # m/s: window center below this is a stop label
V_DRIVE_MIN = 4.0  # m/s: window center above this is a driving label
MAX_WINDOWS_PER_SESSION = 3000


def _window_is_quiet(signal, lo, n_win):
    """Window is 'quiet' iff it passes the detector thresholds (engine.py)."""
    win = signal[lo:lo + n_win]
    av = win[:, :3].var(axis=0).sum()
    gv = win[:, 3:].var(axis=0).sum()
    return bool(av < ACCEL_VAR_MAX and gv < GYRO_VAR_MAX)


def build_training_windows(sessions):
    """(x, y, driver) arrays: feature-consistent stopped/driving extremes.

    A window is kept only when its vibration agrees with its vehicle-speed
    label: quiet + stopped gets the tight target, busy + driving the loose
    one. Ambiguous windows (creep speeds, or motion the IMU does not
    register) carry no learnable signal and are dropped.
    """
    xs, ys, drivers, kinds = [], [], [], []
    for session in sessions:
        sample = session.sample
        n_win = int(round(WINDOW_S * 100.0))
        stride = int(round(STRIDE_S * 100.0))
        signal = np.column_stack((sample.accel, sample.gyro))
        t_gt = sample.gt_pose["t"]
        speed = np.hypot(sample.gt_pose["vx"], sample.gt_pose["vy"])
        count = 0
        for lo in range(0, signal.shape[0] - n_win + 1, stride):
            if count >= MAX_WINDOWS_PER_SESSION:
                break
            center = sample.t[lo + n_win // 2]
            v = float(np.interp(center, t_gt, speed))
            quiet = _window_is_quiet(signal, lo, n_win)
            if quiet and v <= V_STOP_MAX:
                sigma = SIGMA_STOP
            elif not quiet and v >= V_DRIVE_MIN:
                sigma = SIGMA_DRIVE
            else:
                continue  # unidentifiable: no learnable signal
            xs.append(signal[lo:lo + n_win].T)  # (6, 100) channel-first
            ys.append(np.full(3, sigma ** 2))
            drivers.append(session.driver)
            kinds.append("stop" if quiet else "drive")
            count += 1
    kinds = np.asarray(kinds)
    print("kept windows by kind:",
          {k: int((kinds == k).sum()) for k in ("stop", "drive")})
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64), \
        np.asarray(drivers)


def run(sessions=None, out_weights=WEIGHTS, out_report=REPORT, seed=0,
        epochs=60, windows_cache=WINDOWS_CACHE):
    t0 = time.time()
    cache = Path(windows_cache) if windows_cache else None
    if cache is not None and cache.is_file():
        data = np.load(cache)
        x, y, drivers = data["x"], data["y"], data["drivers"]
        print(f"loaded {x.shape[0]} windows from cache {cache.name}")
    else:
        if sessions is None:
            sessions = discover_sessions(DATA_ROOT)
        x, y, drivers = build_training_windows(sessions)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, x=x, y=y, drivers=drivers)
            print(f"cached {x.shape[0]} windows to {cache.name}")
    unique = sorted(set(drivers))
    if len(unique) < 3:
        raise ValueError("at least three drivers are required")
    n_sessions = len(sessions) if sessions is not None else "(cached)"
    print(f"sessions={n_sessions} windows={x.shape[0]} drivers={unique}")

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
        mean_r, std_r = mean.reshape(1, 6, 1), std.reshape(1, 6, 1)
        train_x = (x[train] - mean_r) / std_r
        val_x = (x[val] - mean_r) / std_r
        test_x = (x[test] - mean_r) / std_r
        checkpoint = out_weights.with_name(
            f"noisenet_fold_{test_driver}.pt")
        if checkpoint.is_file():
            state = torch.load(checkpoint, map_location="cpu",
                               weights_only=True)
            model = NoiseNet()
            model.load_state_dict(state["model"])
            model.eval()
            history = {"validation": [state["validation_drift"]]}
            path = checkpoint
            print(f"  fold {test_driver}: resumed existing checkpoint "
                  f"(val={state['validation_drift']:.4f})")
        else:
            model, history, path = train_noise_net(
                train_x, y[train], val_x, y[val], checkpoint, seed=seed,
                epochs=epochs, patience=12)
        with torch.no_grad():
            test_x_t = torch.as_tensor(test_x, dtype=torch.float32)
            test_drift = float(log_variance_loss(model(test_x_t), y[test]))
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
        "target": "sigma = 0.3 m/s (quiet+GT-stopped) or 20 m/s "
                  "(busy+GT-driving), target = sigma^2 per axis; ambiguous "
                  "windows dropped; GT speed is a training label only",
        "drop": "windows with GT speed in (V_STOP_MAX, V_DRIVE_MIN) or "
                "vibration inconsistent with the motion label",
        "window_s": WINDOW_S, "stride_s": STRIDE_S,
        "sessions": len(sessions) if sessions is not None else None,
        "windows": int(x.shape[0]),
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