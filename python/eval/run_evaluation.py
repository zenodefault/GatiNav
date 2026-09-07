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

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from python.eval.engine import (CONFIG_KEYS, OUTAGE_DURATIONS_S, _gnss_enu,
                                _init_state, _match_headings, _pass,
                                _stationary_mask, compute_metrics,
                                find_windows)
from python.matching.matcher import OSMMatcher

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data/IO-VNBD/Synchronised V abd S datasets"
# trust a road match only while the estimate is this close to the matched
# road; beyond it the HMM has snapped to a wrong road and NHC would amplify
MATCH_RESIDUAL_MAX_M = 30.0
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


def _window_bbox(masked):
    """Lat/lon box over the window's valid GNSS fixes (for graph download)."""
    g = masked.gnss
    ok = (np.isfinite(g["lat_deg"]) & np.isfinite(g["lon_deg"])
          & (np.abs(g["lat_deg"]) > 1e-6))
    if ok.sum() < 2:
        return None
    return (float(g["lat_deg"][ok].max()), float(g["lat_deg"][ok].min()),
            float(g["lon_deg"][ok].max()), float(g["lon_deg"][ok].min()))


def matcher_from_window(masked, session):
    """Local drive graph covering this window's GNSS box (specs/04 section 1)."""
    box = _window_bbox(masked)
    if box is None:
        return None
    north, south, east, west = box
    cache = ROOT / "python/matching/graphs" / f"{session}.graphml"
    try:
        return OSMMatcher.from_bbox(north, south, east, west, cache)
    except Exception:
        return None


def _match_with_anchor(matcher, masked, t, start_t, gps, est_xy):
    """Road headings for the outage estimate, anchored on pre-outage GNSS.

    The HMM receives [last ~10 s of valid pre-outage GNSS fixes; estimate]
    so it starts on the true road at the outage instant instead of on an
    already-drifted trajectory. The track is transformed from ENU metres to
    the graph CRS (they never share a frame otherwise). Returns headings
    aligned with est_xy only.
    """
    headings = np.full(est_xy.shape[0], np.nan)
    if matcher is None or est_xy.shape[0] == 0:
        return headings
    if not hasattr(matcher, "graph"):
        # stub matcher in tests: raw ENU input, no anchoring
        return _match_headings(matcher, est_xy)
    px, py, pz, _, valid = gps
    if not valid.any():
        return headings
    i0 = int(np.argmax(valid))
    lat0 = float(masked.gnss["lat_deg"][i0])
    lon0 = float(masked.gnss["lon_deg"][i0])
    pre = np.flatnonzero(valid & (t < start_t))
    tail = (np.column_stack((px[pre[-1000:]], py[pre[-1000:]]))
            if pre.size else np.empty((0, 2)))
    track = np.vstack((tail, est_xy)) if tail.size else est_xy
    crs = matcher.graph.graph.get("crs", "EPSG:3857")
    xy = matcher.enu_to_graph_xy(track, lat0, lon0, crs)
    h_all = _match_headings(matcher, xy, gate_residual_m=MATCH_RESIDUAL_MAX_M)
    return h_all[tail.shape[0]:]


def evaluate_window(masked, start_t, end_t, window_id, session, matcher=None,
                    model=None):
    """Run all four configurations on one outage window."""
    t = masked.t
    sel = (t >= start_t) & (t < end_t)
    gt = np.column_stack((masked.gt_pose["x"][sel], masked.gt_pose["y"][sel]))
    gps = _gnss_enu(masked)
    px, py, pz, _, valid = gps
    last = np.nonzero(valid & (masked.gnss["t"] < start_t))[0]
    frozen = (np.column_stack((np.full(sel.sum(), px[last[-1]]),
                               np.full(sel.sum(), py[last[-1]])))
              if last.size else gt.copy())
    run_start_t = max(float(masked.t[0]), start_t - 60.0)
    pos0, vel0, rot0 = _init_state(masked, px, py, pz, valid, run_start_t)
    stationary = _stationary_mask(masked.accel, masked.gyro)
    traj = {}
    for mode in CONFIG_KEYS[:3]:
        est, _ = _pass(masked, start_t, end_t, mode, pos0, vel0, model,
                       stationary, gps, rotation=rot0,
                       run_start_t=run_start_t)
        traj[mode] = est[:, :2]
    headings = _match_with_anchor(matcher, masked, t, start_t, gps,
                                  traj["cnn"])
    est, nhc = _pass(masked, start_t, end_t, "full", pos0, vel0, model,
                     stationary, gps, headings=headings, rotation=rot0,
                     run_start_t=run_start_t)
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
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = NoiseNet()
        model.load_state_dict(state["model"])
        model.eval()
        # training-fold normalization stats ride along in the checkpoint
        model.mean = np.asarray(state.get("mean", np.zeros(6)),
                                dtype=np.float64)
        model.std = np.asarray(state.get("std", np.ones(6)),
                               dtype=np.float64)
        return model
    return None


def evaluate_all(windows, matcher=None, model=None, make_matcher=None):
    """matcher: shared matcher (tests); make_matcher: per-window builder."""
    results = []
    for i, (session, window) in enumerate(windows):
        window_matcher = matcher
        if window_matcher is None and make_matcher is not None:
            window_matcher = make_matcher(window.masked,
                                          session.split("/")[-1])
        results.append(evaluate_window(
            window.masked, window.start_t, window.end_t,
            f"W{i + 1:02d}-{int(window.duration_s)}s",
            session.split("/")[-1], window_matcher, model))
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


def export_session_trajectories(session_id, trajectories, out_json):
    """Write the per-session trajectory JSON consumed by the Android loader."""
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "session_id": session_id,
        "raw": [[float(x), float(y)] for x, y in trajectories.get("raw", [])],
        "inertial": [[float(x), float(y)] for x, y in trajectories.get("inertial", [])],
        "fused": [[float(x), float(y)] for x, y in trajectories.get("fused", [])],
    }
    out_path.write_text(json.dumps(payload))
    return out_path


def run_evaluation(data_root=DATA_ROOT, model=None,
                   out_md=RESULTS_MD, out_png=RESULTS_PNG):
    """Discover windows, run the four configs, write metrics and plot."""
    windows = find_windows(data_root)
    results = evaluate_all(windows, matcher=None, model=model or load_model(),
                           make_matcher=matcher_from_window)
    write_metrics(results, out_md)
    write_plot(results, out_png)
    return results


if __name__ == "__main__":
    run_evaluation()
