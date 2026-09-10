"""Residual-net training (Blueprint Phase 1.2): MSE loss, Adam, LODO folds.

Loss is plain MSE between the predicted residual and the wheel-derived
residual (the plan's loss: "MSE between the network's predicted residual
and the actual ground truth residual"), optimised with Adam, offline on
the desktop, per the project protocol.

Driver-held-out folds: the next driver in sorted order is validation, the
selected driver is test. Resumable per fold (results JSON + checkpoints);
RAM-safe (thread caps before numpy/torch import, streaming cache).

Outputs:
  results/audit/denoise_results.json   per-fold val MSE + deployed fold
  results/audit/denoise_fold_<driver>.pt
  python/ml/weights/denoise.pt         best-fold weights + normalization
"""

import json
import os
import time
from pathlib import Path

# RAM guards -- must run before numpy/torch import (see python/ml/odom/train.py).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np
import torch

from python.ml.denoise.dataset import (CACHE, ROOT, build_all, driver_folds,
                                       normalize_stats)
from python.ml.denoise.model import ResidualTCN

WEIGHTS = ROOT / "python/ml/weights/denoise.pt"
RESULTS = ROOT / "results/audit/denoise_results.json"
FOLD_PT = ROOT / "results/audit/denoise_fold_{driver}.pt"

SEED = 20260909
EPOCHS = 40
BATCH = 32
LR = 1e-3
EARLY_STOP_PATIENCE = 6
TRAIN_STRIDE_K = 2  # training subsample (cache stride is 0.5 s; 1 s effective)


def _predict(model, x, mean, std, batch=256):
    model.eval()
    out = []
    with torch.no_grad():
        for lo in range(0, x.shape[0], batch):
            xb = torch.from_numpy(
                ((x[lo:lo + batch] - mean) / std).astype(np.float32))
            out.append(model(xb[:, None, :]).numpy()[:, 0])
    return np.concatenate(out)


def train_fold(windows, fold, test_driver, epochs=EPOCHS,
               patience=EARLY_STOP_PATIENCE):
    torch.manual_seed(SEED)
    torch.set_num_threads(min(2, os.cpu_count() or 1))
    torch.set_num_interop_threads(1)
    rng = np.random.default_rng(SEED)
    tr = fold["train_idx"][::TRAIN_STRIDE_K]
    va = np.asarray(fold["val_idx"])
    te = np.asarray(fold["test_idx"])
    # OOM guard: never materialise full float32 train/val copies (the 72-
    # session cache is ~195k x 500 float32). Batches are sliced from the
    # cached numpy arrays on demand; validation is scored in chunks.
    x_all, y_all = windows.x, windows.target
    # OOM guard: chunked mean/std over the train subset (normalize_stats
    # would materialise the whole subset as float64 at once).
    _sum = 0.0
    _sq = 0.0
    _cnt = 0
    for _lo in range(0, tr.shape[0], 4096):
        _sub = x_all[tr[_lo:_lo + 4096]].astype(np.float64)
        _sum += float(_sub.sum())
        _sq += float((_sub ** 2).sum())
        _cnt += _sub.size
    mean = _sum / max(_cnt, 1)
    std = float(np.sqrt(max(_sq / max(_cnt, 1) - mean ** 2, 0.0)))
    std = std if std > 1e-9 else 1.0
    model = ResidualTCN()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n_tr = int(tr.shape[0])

    def _batch(sel):
        xb = torch.from_numpy(
            ((x_all[sel] - mean) / std).astype(np.float32))
        yb = torch.from_numpy(y_all[sel].astype(np.float32))
        return xb[:, None, :], yb

    def _val_mse(chunk=256):
        model.eval()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for lo in range(0, va.shape[0], chunk):
                sel = va[lo:lo + chunk]
                xb, yb = _batch(sel)
                tot += float(torch.sum(
                    (model(xb)[:, 0] - yb) ** 2))
                cnt += sel.shape[0]
        return tot / max(cnt, 1)
    best_val, best_state, stale, t0 = float("inf"), None, 0, time.time()
    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(n_tr)
        losses = []
        for lo in range(0, n_tr, BATCH):
            sel = tr[perm[lo:lo + BATCH]]
            xb, yb = _batch(sel)
            opt.zero_grad()
            loss = torch.mean((model(xb)[:, 0] - yb) ** 2)
            loss.backward()
            opt.step()
            losses.append(float(loss))
        val = _val_mse()
        if val < best_val:
            best_val, best_state, stale = val, {
                k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= patience:
                print(f"[{test_driver}] early stop at epoch {epoch + 1} "
                      f"(best val {best_val:.4f})", flush=True)
                break
        print(f"[{test_driver}] epoch {epoch + 1}/{epochs} "
              f"train {np.mean(losses):.4f} val {val:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    model.load_state_dict(best_state)
    ckpt = Path(str(FOLD_PT).format(driver=test_driver.replace(" ", "_")))
    torch.save({"model": model.state_dict(), "mean": mean, "std": std,
                "val_mse": best_val, "test_driver": test_driver}, ckpt)
    # held-out test residual error on the test driver
    test = _predict(model, windows.x[te], mean, std)
    test_mse = float(np.mean((test - windows.target[te]) ** 2))
    return model, mean, std, best_val, test_mse, ckpt


def main(data_root=ROOT / "data/IO-VNBD/Synchronised V abd S datasets",
         epochs=EPOCHS, only_driver=None):
    windows = build_all(data_root, CACHE)
    print(f"windows: {len(windows)} from "
          f"{len(np.unique(windows.segment))} sessions", flush=True)
    folds = driver_folds(windows)
    results = {}
    if RESULTS.is_file():
        results = json.loads(RESULTS.read_text())
    for driver, fold in folds.items():
        if only_driver and driver != only_driver:
            continue
        if driver in results:
            print(f"[{driver}] already trained, skipping", flush=True)
            continue
        model, mean, std, val, test_mse, ckpt = train_fold(
            windows, fold, driver, epochs=epochs)
        results[driver] = {"val_mse": val, "test_mse": test_mse,
                           "checkpoint": str(ckpt), "mean": mean, "std": std}
        RESULTS.write_text(json.dumps(results, indent=2))
        print(f"[{driver}] val MSE {val:.4f} | held-out test MSE "
              f"{test_mse:.4f}", flush=True)
    if not results:
        print("no folds trained (only_driver filter?)")
        return
    # deploy the fold with the lowest validation MSE
    best = min(results, key=lambda d: results[d]["val_mse"])
    torch.save({"model": torch.load(
        results[best]["checkpoint"], map_location="cpu", weights_only=True)
        ["model"],
        "mean": results[best]["mean"], "std": results[best]["std"],
        "fold": best}, WEIGHTS)
    print(f"deployed best fold [{best}] -> {WEIGHTS}", flush=True)


if __name__ == "__main__":
    import sys
    main(only_driver=sys.argv[1] if len(sys.argv) > 1 else None)