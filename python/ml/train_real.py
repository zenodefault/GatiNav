"""Train NoiseNet on real IO-VNBD windows (specs/03 section 4).

Two heads share one trunk:

* Variance head (columns 0..2): per-window ZUPT measurement variances. A
  regression from a 1 s IMU window to continuous vehicle speed is not
  identifiable for this task on real data (window vibration correlates with
  speed at log-r <= 0.33 and stopped/moving feature clouds overlap), so the
  ZUPT targets are restricted to the feature-consistent extremes deployment
  actually exercises -- the filter only queries the network on windows the
  variance detector accepts as stationary:

      quiet window + GT stopped  ->  sigma = 0.3  m/s  (tight ZUPT)
      busy  window + GT driving  ->  sigma = 20   m/s  (ZUPT disabled)

  Ambiguous windows (slow creep, or vibration inconsistent with the motion
  label) are dropped from the noise loss via a keep-mask.

* Speed head (column 3): regresses log(1 + v) with v the IO-VNBD vehicle
  wheel-encoder speed at the window centre (m/s). All windows contribute to
  this term, so the model learns the full speed profile including creep and
  transients. Targets are log-normalized so the loss directly minimises
  *fractional* speed error -- the quantity the <10%-of-distance benchmark
  depends on.

Loss = noise_loss (log-variance MSE, kept windows only)
     + LAMBDA_SPEED * speed_loss (log-speed MSE, all windows).

Folds are leave-one-driver-out with train-only normalization. Deploys the
fold model with the lowest composite validation loss to
python/ml/weights/noisenet.pt and writes results/audit/cnn_training.md
including the held-out fractional speed RMSE (all windows and driving-only).
"""

from pathlib import Path
import json
import time

import numpy as np
import torch

from python.ml.cnn import NoiseNet
from python.ml.dataset import discover_sessions
from python.ml.train import seed_everything
from python.eval.engine import ACCEL_VAR_MAX, GYRO_VAR_MAX

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data/IO-VNBD/Synchronised V abd S datasets"
WEIGHTS = ROOT / "python/ml/weights/noisenet.pt"
REPORT = ROOT / "results/audit/cnn_training.md"
WINDOWS_CACHE = ROOT / "results/audit/cnn_training_windows.npz"

CACHE_VERSION = 3  # bump when the cached target layout changes
WINDOW_S = 1.0
STRIDE_S = 5.0  # training-efficiency stride (specs/03 uses 0.5 s at inference)
SIGMA_STOP = 0.3  # m/s tight ZUPT sigma for genuinely stationary windows
SIGMA_DRIVE = 20.0  # m/s loose sigma, effectively disables ZUPT while moving
V_STOP_MAX = 0.5  # m/s: window centre below this is a stop label
V_DRIVE_MIN = 4.0  # m/s: window centre above this is a driving label
LAMBDA_SPEED = 1.0  # relative weight of the log-speed term in the loss
MAX_WINDOWS_PER_SESSION = 3000


def _window_is_quiet(signal, lo, n_win):
    """Window is 'quiet' iff it passes the detector thresholds (engine.py)."""
    win = signal[lo:lo + n_win]
    av = win[:, :3].var(axis=0).sum()
    gv = win[:, 3:].var(axis=0).sum()
    return bool(av < ACCEL_VAR_MAX and gv < GYRO_VAR_MAX)


def build_training_windows(sessions):
    """(x, y_sigma, kept, v, drivers) arrays for the two-head task.

    x: (N, 6, 100) channel-first IMU windows. y_sigma: (N, 3) per-axis
    variance targets (sigma^2) meaningful only where kept is True. v: (N,)
    wheel-encoder speed (m/s) at the window centre, GT for the speed head.
    A window is 'kept' for the noise loss only when its vibration agrees
    with its vehicle-speed label: quiet + stopped (tight) or busy + driving
    (loose). Ambiguous windows carry no learnable ZUPT signal but still feed
    the speed head.
    """
    xs, ys, kepts, vs, drivers = [], [], [], [], []
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
                keep = True
            elif not quiet and v >= V_DRIVE_MIN:
                sigma = SIGMA_DRIVE
                keep = True
            else:
                sigma = 0.0  # no ZUPT target for this window
                keep = False
            xs.append(signal[lo:lo + n_win].T)  # (6, 100) channel-first
            ys.append(np.full(3, sigma ** 2))
            kepts.append(keep)
            vs.append(v)
            drivers.append(session.driver)
            count += 1
    kept = np.asarray(kepts, dtype=bool)
    print(f"windows: {len(vs)} total, {int(kept.sum())} kept for ZUPT "
          f"({int((kept & (np.asarray(vs) <= V_STOP_MAX)).sum())} stops, "
          f"{int((kept & (np.asarray(vs) >= V_DRIVE_MIN)).sum())} drives)")
    return (np.asarray(xs, dtype=np.float64),
            np.asarray(ys, dtype=np.float64), kept,
            np.asarray(vs, dtype=np.float64), np.asarray(drivers))


def _log_speed_loss(pred_logspeed, v):
    target = torch.log1p(torch.as_tensor(
        v, dtype=pred_logspeed.dtype, device=pred_logspeed.device))
    return torch.mean((pred_logspeed - target) ** 2)


def _composite_loss(model, x, y_sigma, kept, v):
    """noise log-variance MSE (kept only) + lambda * log-speed MSE (all)."""
    out = model(x)
    speed_loss = _log_speed_loss(out[:, 3], v)
    if kept.any():
        target = torch.as_tensor(y_sigma[kept], dtype=out.dtype,
                                 device=out.device)
        noise = out[kept, :3]
        noise_loss = torch.mean(
            (torch.log(noise) - torch.log(target)) ** 2)
    else:
        noise_loss = torch.as_tensor(0.0, device=out.device)
    return noise_loss + LAMBDA_SPEED * speed_loss, noise_loss, speed_loss


def train_speed_net(train_x, y_sigma, kept, v, val_x, val_y, val_kept,
                    val_v, checkpoint_path, seed=0, epochs=60, batch_size=32,
                    learning_rate=1e-3, patience=12, min_delta=1e-6):
    seed_everything(seed)
    model = NoiseNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_x = torch.as_tensor(train_x, dtype=torch.float32)
    val_x = torch.as_tensor(val_x, dtype=torch.float32)
    if train_x.ndim != 3 or train_x.shape[1:] != (6, 100):
        raise ValueError("train_x must have shape (batch, 6, 100)")
    best = float("inf")
    stale = 0
    history = {"train": [], "validation": []}
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(epochs):
        model.train()
        order = torch.arange(train_x.shape[0])
        train_losses = []
        for start in range(0, train_x.shape[0], batch_size):
            idx = order[start:start + batch_size]
            optimizer.zero_grad()
            loss, _, _ = _composite_loss(
                model, train_x[idx], y_sigma[idx], kept[idx], v[idx])
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            validation, _, _ = _composite_loss(
                model, val_x, val_y, val_kept, val_v)
        train_loss = float(np.mean(train_losses))
        validation_loss = float(validation)
        history["train"].append(train_loss)
        history["validation"].append(validation_loss)
        if validation_loss < best - min_delta:
            best, stale = validation_loss, 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "validation_drift": validation_loss, "seed": seed},
                       checkpoint_path)
        else:
            stale += 1
            if stale >= patience:
                break
    return model, history, checkpoint_path


def fractional_speed_rmse(model, x, v):
    """RMSE(m/s) / mean(v) -- the <10% benchmark gate, all and driving-only."""
    with torch.no_grad():
        pred_v = torch.expm1(model(x)[:, 3]).cpu().numpy()
    v = np.asarray(v)
    rmse_all = float(np.sqrt(np.mean((pred_v - v) ** 2)))
    frac_all = rmse_all / max(float(np.mean(v)), 1e-6)
    drive = v > 1.0
    if drive.any():
        rmse_drive = float(np.sqrt(np.mean((pred_v[drive] - v[drive]) ** 2)))
        frac_drive = rmse_drive / max(float(np.mean(v[drive])), 1e-6)
    else:
        rmse_drive = frac_drive = None
    return {"rmse": rmse_all, "frac_all": frac_all, "rmse_drive": rmse_drive,
            "frac_drive": frac_drive, "mean_v": float(np.mean(v))}


def run(sessions=None, out_weights=WEIGHTS, out_report=REPORT, seed=0,
        epochs=60, windows_cache=WINDOWS_CACHE):
    t0 = time.time()
    cache = Path(windows_cache) if windows_cache else None
    if cache is not None and cache.is_file():
        data = np.load(cache)
        if data["version"] != CACHE_VERSION:
            raise SystemExit(f"cache {cache} is version {data['version']}, "
                             f"expected {CACHE_VERSION} -- rebuild it")
        x, y_sigma, kept, v, drivers = (data["x"], data["y_sigma"],
                                        data["kept"], data["v"],
                                        data["drivers"])
        print(f"loaded {x.shape[0]} windows from cache {cache.name}")
    else:
        if sessions is None:
            sessions = discover_sessions(DATA_ROOT)
        x, y_sigma, kept, v, drivers = build_training_windows(sessions)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, version=CACHE_VERSION, x=x, y_sigma=y_sigma,
                     kept=kept, v=v, drivers=drivers)
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
            model, history, path = train_speed_net(
                train_x, y_sigma[train], kept[train], v[train],
                val_x, y_sigma[val], kept[val], v[val],
                checkpoint, seed=seed, epochs=epochs, patience=12)
        with torch.no_grad():
            test_composite, test_noise, _ = _composite_loss(
                model, torch.as_tensor(test_x, dtype=torch.float32),
                y_sigma[test], kept[test], v[test])
        speed_test = fractional_speed_rmse(
            model, torch.as_tensor(test_x, dtype=torch.float32), v[test])
        result = {"test_driver": test_driver, "val_driver": val_driver,
                  "checkpoint": str(path),
                  "validation_drift": float(history["validation"][-1]),
                  "test_composite": float(test_composite),
                  "test_noise": float(test_noise),
                  "speed_test": speed_test,
                  "windows": {"train": int(train.sum()), "val": int(val.sum()),
                              "test": int(test.sum())},
                  "mean": mean.tolist(), "std": std.tolist()}
        fold_results.append(result)
        print(f"  fold {test_driver}: val={result['validation_drift']:.4f} "
              f"test_noise={test_noise:.4f} speed_frac_all="
              f"{speed_test['frac_all'] * 100:.1f}% "
              f"speed_frac_drive="
              f"{speed_test['frac_drive'] * 100 if speed_test['frac_drive'] is not None else float('nan'):.1f}% "
              f"(train={int(train.sum())}, val={int(val.sum())}, "
              f"test={int(test.sum())})")
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
        "targets": "variance head: sigma = 0.3 m/s (quiet+GT-stopped) or "
                   "20 m/s (busy+GT-driving), target = sigma^2 per axis; "
                   "ambiguous windows masked out of the noise loss. "
                   "speed head: log(1 + v) with v = wheel-encoder speed at "
                   "the window centre; all windows contribute",
        "lambda_speed": LAMBDA_SPEED,
        "window_s": WINDOW_S, "stride_s": STRIDE_S,
        "sessions": len(sessions) if sessions is not None else None,
        "windows": int(x.shape[0]),
        "deployed": best["checkpoint"],
        "deployed_driver": best["test_driver"],
        "deployed_speed_frac_all": best["speed_test"]["frac_all"],
        "deployed_speed_frac_drive": best["speed_test"]["frac_drive"],
        "folds": fold_results, "elapsed_s": round(time.time() - t0, 1),
    }
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"deployed {best['checkpoint']} -> {out_weights}")
    print(f"deployed speed: frac_all={best['speed_test']['frac_all'] * 100:.1f}% "
          f"frac_drive="
          f"{best['speed_test']['frac_drive'] * 100 if best['speed_test']['frac_drive'] is not None else float('nan'):.1f}%")
    print(f"elapsed {report['elapsed_s']}s; report: {out_report}")
    return report


if __name__ == "__main__":
    seed_everything(0)
    run()