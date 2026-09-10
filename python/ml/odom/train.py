"""Phase 4B training + open-loop gate evaluation (AGENTS.md Phase 4B).

Loss: Gaussian NLL on the three heads (displacement dominant by weight):
    L = NLL_disp + NLL_vel + NLL_yaw
Augmentation (train only, seeded): small random mount-pose rotations
(axis-angle <= 3 deg on the vehicle-frame signals), realistic MEMS noise
injection (accel 0.02 m/s^2, gyro 0.001 rad/s), scale jitter (0.5% / 1%).
Log augmented-vs-clean held-out delta in the training report.

Open-loop gate (deployment-shaped):
  * model distance per session: integrate the velocity head sampled at the
    0.5 s stride over all of a held-out session's windows vs wheel distance
  * per-window fractional displacement error, model vs calibrated-INS-alone
    (INS baseline = GT speed at window start + double-integrated calibrated
    forward accel -- the EKF's between-updates propagation, GNSS-seeded v0)
  * report median/p90, all windows and driving-only (disp_gt >= 2 m)

Gate: held-out forward-distance error <= 5% (target 3%) on driving windows,
beats INS-alone per held-out session. Failure -> results/audit/odom_failure.md.

Resumable per fold: checkpoints to results/audit/odom_fold_<driver>.pt.
"""

import json
import os
import time
from pathlib import Path

# RAM guards -- MUST run before numpy/torch import. Cap BLAS/OpenMP pools
# so training coexists with the rest of the system on a 16 GB machine
# (mirrors `nix develop --option max-jobs 1 --option cores 2`).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np
import torch

from python.ml.odom.dataset import (CACHE, ROOT, build_all, driver_folds,
                                    normalize_stats)
from python.ml.odom.model import OdomNet, gaussian_nll

WEIGHTS = ROOT / "python/ml/weights/odomnet.pt"
REPORT = ROOT / "results/audit/odom_training.md"
FAIL_MD = ROOT / "results/audit/odom_failure.md"
FOLD_PT = ROOT / "results/audit/odom_fold_{driver}.pt"

SEED = 20260908
EPOCHS = 40
BATCH = 64
LR = 1e-3
EARLY_STOP_PATIENCE = 6  # epochs without val improvement before stopping
DRIVE_DISP_MIN_M = 2.0
# training subsample: every Kth window (cache stride is 0.5 s; 1 s effective)
TRAIN_STRIDE_K = 2

DRIVING_GATES = {"frac_dist_pct_max": 5.0, "frac_dist_pct_target": 3.0}


def _augment(x, rng):
    """Mount-pose + noise + scale augmentation on a batch (B, 6, W)."""
    b = x.shape[0]
    out = x.copy()
    # random small rotations (mount perturbation), independent per sample
    angles = rng.normal(0.0, np.deg2rad(1.0), size=(b, 3))
    for i in range(b):
        ca, sa = np.cos(angles[i]), np.sin(angles[i])
        rx = np.array([[1, 0, 0], [0, ca[0], -sa[0]], [0, sa[0], ca[0]]])
        ry = np.array([[ca[1], 0, sa[1]], [0, 1, 0], [-sa[1], 0, ca[1]]])
        rz = np.array([[ca[2], -sa[2], 0], [sa[2], ca[2], 0], [0, 0, 1]])
        r = (rz @ ry @ rx).astype(out.dtype)
        out[i, 0:3] = r @ out[i, 0:3]
        out[i, 3:6] = r @ out[i, 3:6]
    out[:, 0:3] += rng.normal(0.0, 0.02, size=out[:, 0:3].shape)
    out[:, 3:6] += rng.normal(0.0, 0.001, size=out[:, 3:6].shape)
    out[:, 0:3] *= (1.0 + rng.normal(0.0, 0.005, size=(b, 1, 1)))
    out[:, 3:6] *= (1.0 + rng.normal(0.0, 0.01, size=(b, 1, 1)))
    return out


def _loss(model, xb, targets):
    out = model(xb)
    l_disp = gaussian_nll(out[:, 0], out[:, 1], targets[:, 0])
    l_vel = gaussian_nll(out[:, 2], out[:, 3], targets[:, 1])
    l_yaw = gaussian_nll(out[:, 4], out[:, 5], targets[:, 2])
    return l_disp + l_vel + l_yaw, (float(l_disp), float(l_vel), float(l_yaw))


def _predict(model, x, mean, std, batch=256):
    model.eval()
    disp, vel, yaw = [], [], []
    with torch.no_grad():
        for lo in range(0, x.shape[0], batch):
            xb = torch.from_numpy(
                ((x[lo:lo + batch] - mean[:, None]) / std[:, None])
                .astype(np.float32))
            out = model(xb).numpy()
            disp.append(out[:, 0])
            vel.append(out[:, 2])
            yaw.append(out[:, 4])
    return np.concatenate(disp), np.concatenate(vel), np.concatenate(yaw)


def _ins_baseline_window(x, v0, disp_gt):
    """Calibrated-INS-alone displacement over each window.

    v0 = wheel speed at window START -- the deployment analog of the GNSS-
    seeded velocity the EKF carries into an outage (measured on the common
    clock, not leaked from the target). Forward accel = calibrated
    vehicle-frame x channel; trapezoid double integration:
        v(t) = v0 + trapz(a),  disp = v0*T + trapz(v)
    """
    dt = 1.0 / 100.0
    a = x[:, 0, :]  # (B, W) forward accel, vehicle frame
    T = a.shape[1] * dt
    v = np.cumsum(a, axis=1) * dt
    trapz_v = (v[:, :-1] + v[:, 1:]).sum(axis=1) * 0.5 * dt
    return v0 * T + trapz_v


def _frac(err, gt, mask=None):
    frac = np.abs(err) / np.maximum(gt, 1e-6)
    if mask is not None:
        frac = frac[mask]
    if frac.size == 0:
        return float("nan"), float("nan")
    return float(np.median(frac)) * 100.0, float(np.percentile(frac, 90)) * 100.0


def evaluate_fold(model, windows, idx, mean, std, eval_stride=5):
    """Open-loop gate metrics on a window index set.

    eval_stride subsamples windows (5 s effective spacing) so the gate
    stays statistically strong without materialising every overlapping
    window of a large fold.
    """
    idx = idx[::eval_stride]
    x = windows.x[idx]
    disp = windows.disp[idx]
    v0 = windows.v0[idx]
    drive = disp >= DRIVE_DISP_MIN_M
    d_pred, v_pred, _ = _predict(model, x, mean, std)
    d_ins = _ins_baseline_window(x, v0, disp)
    out = {
        "n_windows": int(idx.size),
        "n_driving": int(drive.sum()),
        "model_med_all": _frac(d_pred - disp, disp)[0],
        "model_p90_all": _frac(d_pred - disp, disp)[1],
        "model_med_drive": _frac(d_pred - disp, disp, drive)[0],
        "model_p90_drive": _frac(d_pred - disp, disp, drive)[1],
        "ins_med_all": _frac(d_ins - disp, disp)[0],
        "ins_p90_all": _frac(d_ins - disp, disp)[1],
        "ins_med_drive": _frac(d_ins - disp, disp, drive)[0],
        "ins_p90_drive": _frac(d_ins - disp, disp, drive)[1],
    }
    # session-level distance error: integrate the velocity head at stride
    # against the wheel-derived GT distance over the same intervals
    segments = windows.segment[idx]
    t_end = windows.t_end[idx]
    disp_all = windows.disp[idx]
    dist_rows = []
    for seg in np.unique(segments):
        sel = segments == seg
        order = np.argsort(t_end[sel])
        v = v_pred[sel][order]
        t = t_end[sel][order]
        keep = np.ones(t.size, dtype=bool)
        keep[1:] = np.diff(t) > 0.4  # dedupe stride overlaps
        d_model = float(np.trapz(v[keep], dx=0.5))
        d_gt = float(disp_all[sel][order][keep].sum())
        dist_rows.append({"segment": str(seg), "model_m": d_model,
                          "gt_m": d_gt,
                          "frac_pct": 100.0 * abs(d_model - d_gt) / max(d_gt, 1e-6)})
    out["session_dist"] = dist_rows
    fracs = [r["frac_pct"] for r in dist_rows]
    out["session_frac_med"] = float(np.median(fracs)) if fracs else float("nan")
    out["session_frac_max"] = float(np.max(fracs)) if fracs else float("nan")
    return out


def train_fold(windows, fold, test_driver, epochs=EPOCHS,
               patience=EARLY_STOP_PATIENCE):
    torch.manual_seed(SEED)
    # cap intra-op threads (RAM/VCGL guard; torch reads env at first use,
    # but this makes it explicit and covers the case env was already set)
    torch.set_num_threads(min(2, os.cpu_count() or 1))
    rng = np.random.default_rng(SEED)
    tr, va, te = fold["train_idx"], fold["val_idx"], fold["test_idx"]
    tr = tr[::TRAIN_STRIDE_K]
    mean, std = normalize_stats(windows.x, tr)
    model = OdomNet()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    xt = windows.x[tr]
    tt = np.column_stack((windows.disp[tr], windows.vel[tr], windows.yaw[tr]))
    xv = windows.x[va]
    tv = np.column_stack((windows.disp[va], windows.vel[va], windows.yaw[va]))
    best_val, best_state, history = float("inf"), None, []
    stale = 0
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(xt.shape[0])
        losses = []
        for lo in range(0, len(perm), BATCH):
            sel = perm[lo:lo + BATCH]
            xb = _augment(xt[sel], rng)
            targets = tt[sel]
            x_t = torch.from_numpy(
                ((xb - mean[:, None]) / std[:, None]).astype(np.float32))
            y_t = torch.from_numpy(targets.astype(np.float32))
            loss, parts = _loss(model, x_t, y_t)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            losses.append((float(loss), *parts))
        # validation (clean)
        model.eval()
        with torch.no_grad():
            vloss = []
            for lo in range(0, xv.shape[0], BATCH):
                x_v = torch.from_numpy(
                    ((xv[lo:lo + BATCH] - mean[:, None]) / std[:, None])
                    .astype(np.float32))
                y_v = torch.from_numpy(tv[lo:lo + BATCH].astype(np.float32))
                loss, _ = _loss(model, x_v, y_v)
                vloss.append(float(loss))
        val = float(np.mean(vloss))
        if not np.isfinite(val):
            print(f"[{test_driver}] non-finite val at epoch {epoch + 1}; "
                  "restoring best state and stopping", flush=True)
            break
        history.append({"epoch": epoch, "train": float(np.mean([l[0] for l in losses])),
                        "disp": float(np.mean([l[1] for l in losses])),
                        "vel": float(np.mean([l[2] for l in losses])),
                        "yaw": float(np.mean([l[3] for l in losses])),
                        "val": val})
        if val < best_val:
            best_val, best_state = val, {k: v.clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"[{test_driver}] early stop at epoch {epoch + 1} "
                      f"(best val {best_val:.3f})", flush=True)
                break
        print(f"[{test_driver}] epoch {epoch + 1}/{epochs} "
              f"train {history[-1]['train']:.3f} val {val:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    model.load_state_dict(best_state)
    ckpt = Path(str(FOLD_PT).format(driver=test_driver.replace(" ", "_")))
    torch.save({"model": model.state_dict(), "mean": mean, "std": std,
                "val_loss": best_val, "history": history,
                "test_driver": test_driver}, ckpt)
    return model, mean, std, best_val, history, ckpt


def main(data_root=ROOT / "data/IO-VNBD/Synchronised V abd S datasets",
         epochs=EPOCHS, only_driver=None):
    windows = build_all(data_root, CACHE)
    print(f"windows: {len(windows)} from {len(np.unique(windows.segment))} sessions",
          flush=True)
    folds = driver_folds(windows)
    results_path = ROOT / "results/audit/odom_results.json"
    results = {}
    if results_path.is_file():
        results = json.loads(results_path.read_text())
    for driver, fold in folds.items():
        if only_driver and driver != only_driver:
            continue
        if driver in results:
            print(f"[{driver}] already trained, skipping", flush=True)
            continue
        model, mean, std, val, history, ckpt = train_fold(
            windows, fold, driver, epochs=epochs)
        gate = evaluate_fold(model, windows, fold["test_idx"], mean, std)
        results[driver] = {"val_loss": val, "gate": gate,
                           "checkpoint": str(ckpt)}
        results_path.write_text(json.dumps(results, indent=2))
        print(f"[{driver}] val {val:.3f} "
              f"model drive med {gate['model_med_drive']:.1f}% p90 "
              f"{gate['model_p90_drive']:.1f}% | INS-alone drive med "
              f"{gate['ins_med_drive']:.1f}% | session frac med "
              f"{gate['session_frac_med']:.1f}%", flush=True)
    if not results:
        print("no folds trained (only_driver filter?)")
        return
    (ROOT / "results/audit/odom_results.json").write_text(json.dumps(results, indent=2))
    print("done; results/audit/odom_results.json written", flush=True)


if __name__ == "__main__":
    import sys
    only = sys.argv[1] if len(sys.argv) > 1 else None
    main(only_driver=only)
