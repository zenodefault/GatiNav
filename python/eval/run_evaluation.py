"""Outage-window evaluation driver (specs/04 section 6 + specs/03 section 6).

Discovers outage windows per session (bounded: WINDOWS_PER_DURATION per
30/60/90 s duration), runs the four cumulative configurations (raw-INS,
+ZUPT, +CNN, +NHC+matching) through python/eval/engine.py, and emits
results/final_comparison.png (GT green, GNSS-frozen grey, inertial-only
orange, full system blue) and results/final_metrics.md (per-window rows +
ATE/RPE/drift summary by duration). +CNN is the specs/04 "Phase-3
baseline", +NHC+matching the full system.

Resolved choices (VERIFY in specs/01-04): matching uses
OSMMatcher(TEST_PLACE) with a cached OSM graph; failures and unmatched
samples follow specs/04 section 5 (no NHC, recorded).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from python.eval.engine import (CONFIG_KEYS, OUTAGE_DURATIONS_S, _gnss_enu,
                                _match_headings, _pass, _seed,
                                _stationary_mask, compute_metrics,
                                find_windows)

ROOT = Path(__file__).resolve().parents[2]
TEST_PLACE = "Cambridge, United Kingdom"  # specs/04 section 1: VERIFY city
DATA_ROOT = ROOT / "data/IO-VNBD/Synchronised V abd S datasets"
RESULTS_MD = ROOT / "results/final_metrics.md"
RESULTS_PNG = ROOT / "results/final_comparison.png"
LABELS = {"raw": "raw-INS", "zupt": "+ZUPT", "cnn": "+CNN",
          "full": "+NHC+matching"}


@dataclass
class WindowResult:
    window_id: str
    session: str
    duration_s: float
    gt: np.ndarray  # (n, 2) horizontal ground truth
    frozen: np.ndarray  # (n, 2) GNSS frozen at last pre-outage fix
    traj: dict  # config key -> (n, 2) estimated horizontal position
    nhc_applied: int
    fallback: str


def evaluate_window(masked, start_t, end_t, window_id, session, matcher=None,
                    model=None):
    """Run all four configurations on one outage window."""
    t = masked.t
    sel = (t >= start_t) & (t < end_t)
    gt = np.column_stack((masked.gt_pose["x"][sel], masked.gt_pose["y"][sel]))
    gps = _gnss_enu(masked)
    px, py, _, _, valid = gps
    last = np.nonzero(valid & (masked.gnss["t"] < start_t))[0]
    frozen = (np.column_stack((np.full(sel.sum(), px[last[-1]]),
                               np.full(sel.sum(), py[last[-1]])))
              if last.size else gt.copy())
    pos0, vel0 = _seed(masked, px, py, valid, start_t)
    stationary = _stationary_mask(masked.accel, masked.gyro)
    traj = {}
    for mode in CONFIG_KEYS[:3]:
        est, _ = _pass(masked, start_t, end_t, mode, pos0, vel0, model,
                       stationary, gps)
        traj[mode] = est[:, :2]
    headings = _match_headings(matcher, traj["cnn"])
    est, nhc = _pass(masked, start_t, end_t, "full", pos0, vel0, model,
                     stationary, gps, headings=headings)
    traj["full"] = est[:, :2]
    fallback = ("match unavailable" if matcher is None else
                "no road match accepted" if not np.isfinite(headings).any()
                else "")
    return WindowResult(window_id, session, float(end_t - start_t), gt,
                        frozen, traj, nhc, fallback)


def load_model():
    path = ROOT / "python/ml/weights/noisenet.pt"
    if path.exists():
        import torch
        from python.ml.cnn import NoiseNet
        model = NoiseNet()
        model.load_state_dict(torch.load(path, map_location="cpu",
                                         weights_only=True)["model"])
        model.eval()
        return model
    return None


def evaluate_all(windows, matcher=None, model=None):
    results = []
    for i, (session, window) in enumerate(windows):
        results.append(evaluate_window(
            window.masked, window.start_t, window.end_t,
            f"W{i + 1:02d}-{int(window.duration_s)}s",
            session.split("/")[-1], matcher, model))
    return results


def _summary(results, mode):
    m = [compute_metrics(r.traj[mode], r.gt) for r in results]
    return {key: float(np.mean([v[key] for v in m]))
            for key in ("ate", "rpe", "drift", "rel")}


def write_metrics(results, path=RESULTS_MD):
    """specs/04 section 6 per-window rows + specs/03 section 6 summary."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Final evaluation metrics", "",
             "specs/03 section 6 metrics, no alignment: ATE = RMS horizontal "
             "position error, RPE = RMS one-step horizontal displacement "
             "error, drift = final horizontal error (lower is better).", "",
             "## Per-window comparison (specs/04 section 6)", "",
             "| Window | Segment | Outage s | Baseline drift (+CNN) m | "
             "Matching+NHC drift m | Reduction | Match accepted | Fallback |",
             "|---|---|---:|---:|---:|---:|---|---|"]
    for r in sorted(results, key=lambda r: (r.duration_s, r.window_id)):
        base = compute_metrics(r.traj["cnn"], r.gt)
        full = compute_metrics(r.traj["full"], r.gt)
        red = (f"{(1.0 - full['drift'] / base['drift']) * 100:.1f}%"
               if base["drift"] > 0 else "-")
        lines.append(f"| {r.window_id} | {r.session} | {r.duration_s:.0f} | "
                     f"{base['drift']:.2f} | {full['drift']:.2f} | {red} | "
                     f"{'yes' if r.nhc_applied else 'no'} | "
                     f"{r.fallback or '-'} |")
    lines += ["", "## ATE / RPE / drift by outage duration (specs/03 section "
              "6)", "", "| Duration s | Config | ATE m | RPE m | Drift m | "
              "Drift/distance |", "|---|---|---:|---:|---:|---:|"]
    for d in OUTAGE_DURATIONS_S:
        group = [r for r in results if abs(r.duration_s - d) < 1e-9]
        if not group:
            continue
        for mode in CONFIG_KEYS:
            v = _summary(group, mode)
            lines.append(f"| {d:.0f} | {LABELS[mode]} | {v['ate']:.2f} | "
                         f"{v['rpe']:.2f} | {v['drift']:.2f} | {v['rel']:.1f}% |")
        wins = sum(compute_metrics(r.traj["cnn"], r.gt)["drift"]
                   < compute_metrics(r.traj["zupt"], r.gt)["drift"]
                   for r in group)
        lines += [f"| {d:.0f} | +CNN beats +ZUPT (ties not wins) | "
                  f"{wins}/{len(group)} | - | - | - |", ""]
    path.write_text("\n".join(lines) + "\n")


def write_plot(results, path=RESULTS_PNG):
    """GT green, GNSS-frozen grey, inertial-only orange, full system blue."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = max(1, len(results))
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), squeeze=False)
    axs = axes[0]
    for r, ax in zip(results, axs):
        ax.plot(r.gt[:, 0], r.gt[:, 1], color="green", label="GT")
        ax.plot(r.frozen[:, 0], r.frozen[:, 1], color="grey",
                label="GNSS-frozen")
        ax.plot(r.traj["raw"][:, 0], r.traj["raw"][:, 1], color="orange",
                label="inertial-only")
        ax.plot(r.traj["full"][:, 0], r.traj["full"][:, 1], color="blue",
                label="full system")
        ax.set_title(f"{r.window_id} ({r.session}, {r.duration_s:.0f} s) - "
                     f"{r.fallback or 'matched'}")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(fontsize=7)
    for ax in axs[len(results):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run_evaluation(data_root=DATA_ROOT, matcher=None, model=None,
                   out_md=RESULTS_MD, out_png=RESULTS_PNG):
    """Discover windows, run the four configs, write metrics and plot."""
    windows = find_windows(data_root)
    if matcher is None:
        try:
            from python.matching.matcher import OSMMatcher
            matcher = OSMMatcher(TEST_PLACE)
        except Exception:
            matcher = None
    results = evaluate_all(windows, matcher, model or load_model())
    write_metrics(results, out_md)
    write_plot(results, out_png)
    return results


if __name__ == "__main__":
    run_evaluation()
