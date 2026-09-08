"""Phase 0 artifacts: frozen held-out split + exact outage windows.

build_split assigns every session to train/val/test at the DRIVER level
(sorted drivers: last -> test, second-to-last -> val, rest -> train), so no
driver, route, or recording session appears on both sides of the split.
Driver names come from the categorised tree ("Vw (Driver E)"); the dataset
does not publish the phone model, so device is recorded as "unknown".

The exact windows used for the 30/60/90 s tests are saved as lightweight
records (session pair paths + interval bounds); load_windows rebuilds the
masked OutageWindow objects bit-for-bit from them, so the evaluation report
is reproducible from one command.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from python.eval.engine import FS, find_windows, session_pairs
from python.io.outages import OutageWindow, _mask_gnss
from python.io.iovnb_loader import load_pair, resample_to_common_clock

ROOT = Path(__file__).resolve().parents[2]
SPLIT_JSON = ROOT / "results/heldout_split.json"
WINDOWS_JSON = ROOT / "results/outage_windows.json"


def session_key(s_path):
    """(driver, route, session, device) inferred from the dataset layout."""
    s_path = Path(s_path)
    driver = "unknown"
    for part in s_path.parts:
        if "(Driver" in part:
            driver = part.split("(", 1)[1].rstrip(")").strip()
            break
    stem = s_path.stem[2:]  # strip "S-"
    route = stem.rstrip("0123456789") or stem
    return {"driver": driver, "route": route, "session": stem,
            "device": "unknown"}


def _split_of(driver, drivers_sorted):
    if driver == "unknown":
        return "train"  # uncategorised copies duplicate categorised sessions
    rank = drivers_sorted.index(driver)
    if rank == len(drivers_sorted) - 1:
        return "test"
    if rank == len(drivers_sorted) - 2:
        return "val"
    return "train"


def build_split(data_root):
    """Frozen held-out split for every S csv under ``data_root``."""
    s_paths = sorted(Path(data_root).rglob("S-*.csv"))
    drivers = sorted({session_key(p)["driver"] for p in s_paths
                      if session_key(p)["driver"] != "unknown"})
    sessions = {}
    for p in s_paths:
        key = session_key(p)
        prior = sessions.get(key["session"])
        # Categorised sessions (known driver) take precedence over the
        # uncategorised duplicates of the same recording.
        if prior is not None and prior["driver"] != "unknown":
            continue
        sessions[key["session"]] = {**key,
                                    "split": _split_of(key["driver"],
                                                       drivers)}
    return {"rule": "driver-level: last sorted driver -> test, "
                    "second-to-last -> val, rest -> train",
            "drivers": drivers, "sessions": sessions}


def save_split(split, path=SPLIT_JSON):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(split, indent=2, sort_keys=True) + "\n")
    return Path(path)


def load_split(path=SPLIT_JSON):
    return json.loads(Path(path).read_text())


def window_records(data_root, windows=None):
    """Lightweight records (paths + bounds) for the exact test windows."""
    pairs = {s_path.stem: (s_path, v_path)
             for s_path, v_path in session_pairs(data_root)}
    if windows is None:
        windows = find_windows(data_root)
    records = []
    for stem, w in windows:
        s_path, v_path = pairs[stem]
        records.append({"session": stem,
                        "s_csv": str(s_path),
                        "v_csv": str(v_path),
                        "start_t": float(w.start_t),
                        "end_t": float(w.end_t),
                        "duration_s": float(w.duration_s),
                        "rejected": list(w.rejected)})
    return records


def save_windows(records, path=WINDOWS_JSON):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(records, indent=2) + "\n")
    return Path(path)


def load_window_records(path=WINDOWS_JSON):
    return json.loads(Path(path).read_text())


def rebuild_window(sess, record):
    """Re-mask one OutageWindow from a saved record (bit-for-bit)."""
    i0 = int(np.searchsorted(sess.t, record["start_t"]))
    i1 = int(np.searchsorted(sess.t, record["end_t"]))
    return OutageWindow(start_t=record["start_t"], end_t=record["end_t"],
                        duration_s=record["duration_s"],
                        masked=_mask_gnss(sess, i0, i1),
                        rejected=list(record.get("rejected", [])))


def load_windows(data_root, records=None, fs=FS):
    """Rebuild (stem, OutageWindow) tuples from saved window records."""
    if records is None:
        records = load_window_records()
    pairs = {s_path.stem: (s_path, v_path)
             for s_path, v_path in session_pairs(data_root)}
    out = []
    for rec in records:
        s_path, v_path = pairs[rec["session"]]
        sess = resample_to_common_clock(load_pair(v_path, s_path), fs=fs)
        out.append((rec["session"], rebuild_window(sess, rec)))
    return out


def reproducible_windows(data_root, path=WINDOWS_JSON):
    """Saved windows if present, else discover + save (one-command run)."""
    if Path(path).is_file():
        return load_windows(data_root)
    windows = find_windows(data_root)
    save_windows(window_records(data_root, windows), path)
    return windows
