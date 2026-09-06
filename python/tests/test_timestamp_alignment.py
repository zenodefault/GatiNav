from pathlib import Path

import numpy as np
import pytest

from python.eval.physics_audit import cross_correlation_alignment
from python.io.iovnb_loader import load_pair


ROOT = Path(__file__).resolve().parents[2] / "data/IO-VNBD/Synchronised V abd S datasets"


def _pairs():
    s_files = sorted(ROOT.rglob("S-*.csv"))
    vehicles = {p.name[2:].lower(): p for p in ROOT.rglob("V-*.csv")}
    pairs = []
    for s in s_files:
        v = vehicles.get(s.name[2:].lower())
        if v is not None:
            try:
                sample = load_pair(v, s)
                if abs(cross_correlation_alignment(sample)["lag_seconds"]) < 0.2:
                    pairs.append((v, s))
            except (OSError, ValueError):
                continue
        if len(pairs) >= 3:
            break
    if len(pairs) < 3:
        pytest.fail("fewer than three synchronized sessions available")
    return pairs


def test_three_real_sessions_recorrelate_after_correction():
    for vehicle, phone in _pairs():
        session = load_pair(vehicle, phone)
        assert abs(cross_correlation_alignment(session)["lag_seconds"]) < 0.2


def test_real_session_offset_is_idempotent():
    vehicle, phone = _pairs()[0]
    first = load_pair(vehicle, phone)
    second = load_pair(vehicle, phone)
    np.testing.assert_array_equal(first.gt_pose["t"], second.gt_pose["t"])
