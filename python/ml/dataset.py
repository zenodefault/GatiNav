"""Training-window construction and train-only normalization for ML inputs."""

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from python.io.iovnb_loader import load_pair, resample_to_common_clock


@dataclass
class Session:
    driver: str
    segment: str
    sample: object


@dataclass
class WindowSet:
    x: np.ndarray
    driver: np.ndarray
    segment: np.ndarray
    start_t: np.ndarray
    end_t: np.ndarray


def _windows(session, window_s, stride_s, fs):
    n_win = int(round(float(window_s) * fs))
    n_stride = int(round(float(stride_s) * fs))
    if float(window_s) != 1.0 or stride_s <= 0 or n_win <= 0:
        raise ValueError("window_s must be 1.0 and stride_s must be positive")
    if session.sample.t.size < n_win:
        return WindowSet(np.empty((0, n_win, 6)), np.array([]), np.array([]),
                         np.array([]), np.array([]))
    xs, drivers, segments, starts, ends = [], [], [], [], []
    signal = np.column_stack((session.sample.accel, session.sample.gyro))
    for lo in range(0, signal.shape[0] - n_win + 1, max(1, n_stride)):
        hi = lo + n_win
        xs.append(signal[lo:hi])
        drivers.append(session.driver)
        segments.append(session.segment)
        starts.append(float(session.sample.t[lo]))
        ends.append(float(session.sample.t[hi - 1]))
    return WindowSet(np.asarray(xs, dtype=np.float64), np.asarray(drivers),
                     np.asarray(segments), np.asarray(starts),
                     np.asarray(ends))


def _concat(sets):
    sets = [item for item in sets if item.x.shape[0]]
    if not sets:
        return WindowSet(np.empty((0, 0, 6)), np.array([]), np.array([]),
                         np.array([]), np.array([]))
    return WindowSet(np.concatenate([s.x for s in sets]),
                     np.concatenate([s.driver for s in sets]),
                     np.concatenate([s.segment for s in sets]),
                     np.concatenate([s.start_t for s in sets]),
                     np.concatenate([s.end_t for s in sets]))


def normalization_stats(windows):
    if windows.x.shape[0] == 0:
        raise ValueError("cannot compute statistics from empty windows")
    mean = windows.x.reshape(-1, windows.x.shape[-1]).mean(axis=0)
    std = windows.x.reshape(-1, windows.x.shape[-1]).std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    return mean.astype(np.float64), std.astype(np.float64)


def normalize(windows, mean, std):
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    if mean.shape != (6,) or std.shape != (6,) or np.any(std <= 0):
        raise ValueError("mean and std must be positive 6-channel vectors")
    return (windows.x - mean) / std


def denormalize(values, mean, std):
    return np.asarray(values) * np.asarray(std) + np.asarray(mean)


def _save_stats(path, mean, std):
    if path is not None:
        np.savez(path, mean=mean, std=std)


def _load_stats(path):
    if path is None or not Path(path).is_file():
        return None
    data = np.load(path)
    return data["mean"], data["std"]


def build_leave_one_driver_out(sessions, window_s, stride_s, cache_dir=None,
                               fs=100.0):
    """Return one deterministic fold per driver.

    The next driver in sorted order is validation; the selected driver is test.
    All remaining drivers are training drivers.
    """
    sessions = list(sessions)
    drivers = sorted({s.driver for s in sessions})
    if len(drivers) < 3:
        raise ValueError("at least three drivers are required")
    all_windows = {s.segment: _windows(s, window_s, stride_s, fs) for s in sessions}
    folds = {}
    for i, test_driver in enumerate(drivers):
        val_driver = drivers[(i + 1) % len(drivers)]
        train = _concat([w for s, w in all_windows.items()
                         if next(x for x in sessions if x.segment == s).driver
                         not in (test_driver, val_driver)])
        val = _concat([w for s, w in all_windows.items()
                       if next(x for x in sessions if x.segment == s).driver == val_driver])
        test = _concat([w for s, w in all_windows.items()
                        if next(x for x in sessions if x.segment == s).driver == test_driver])
        cache = Path(cache_dir) / f"{test_driver}.npz" if cache_dir else None
        stats = _load_stats(cache)
        if stats is None:
            stats = normalization_stats(train)
            _save_stats(cache, *stats)
        folds[test_driver] = {
            "train": (train, normalize(train, *stats)),
            "val": (val, normalize(val, *stats)),
            "test": (test, normalize(test, *stats)),
            "mean": stats[0],
            "std": stats[1],
            "validation_driver": val_driver,
        }
    return folds


def _vehicle_pair(s_path):
    """Vehicle CSV paired with an S-*.csv, tolerant of V/v naming variants.

    Most categories name the vehicle file V-<stem>.csv, but the Vta/Vtb
    (urban Driver E) categories use a lowercase 'v' (V-vta2.csv), which the
    strict naming misses and silently drops entire sessions.
    """
    strict = s_path.with_name("V-" + s_path.name[2:])
    if strict.is_file():
        return strict
    want = ("v-" + s_path.name[2:]).lower()
    for candidate in s_path.parent.iterdir():
        if candidate.is_file() and candidate.name.lower() == want:
            return candidate
    return None


def discover_sessions(root):
    """Discover paired synchronized CSV sessions and their driver labels."""
    root = Path(root)
    sessions = []
    for s_path in sorted(root.rglob("S-*.csv")):
        v_path = _vehicle_pair(s_path)
        if v_path is None:
            continue
        # categories differ in depth (M (Driver B)/S-M.csv sits one level
        # shallower than S (Driver A)/S1/S-S1.csv), so search the full path
        match = re.search(r"\((Driver [^)]+)\)", str(s_path))
        if match:
            driver = match.group(1)
        else:
            driver = "unknown"
        sample = resample_to_common_clock(load_pair(v_path, s_path))
        sessions.append(Session(driver, s_path.stem[2:], sample))
    if not sessions:
        raise ValueError(f"no paired synchronized sessions found under {root}")
    return sessions
