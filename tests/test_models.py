"""Smoke tests: both models train on a tiny synthetic network and return sane quantiles.

LightGBM runs in a subprocess: it must never share a process with PyTorch (two OpenMP runtimes,
see ripple/isolate.py).
"""

import subprocess
import sys
import textwrap

import numpy as np
import pytest

from ripple.data.build import chronological_split
from ripple.data.features import FeatureStore, snapshot_rows
from ripple.data.synthetic import SimConfig, generate


@pytest.fixture(scope="module")
def tiny():
    store = FeatureStore(*generate(SimConfig(days=12, seed=5)))
    return store, chronological_split(store.f, frac=(0.6, 0.2), gap_days=0)


def test_gnn_trains_and_predicts_monotone_quantiles(tiny):
    from ripple.models.gnn import GNNTrainer

    store, split = tiny
    tr = GNNTrainer(store, split, epochs=1, hidden=16, heads=2, batch=8, start_stride=8)
    tr._val_rows = lambda max_t=20: GNNTrainer._val_rows(tr, 20)
    tr.fit()
    rows = snapshot_rows(store, split["test"]).iloc[:200]
    p = tr.predict(store, rows)
    assert p["dep"].shape == (len(rows), 19)
    assert np.all(np.diff(p["dep"], axis=1) > 0) and np.all(np.diff(p["arr"], axis=1) > 0)
    att = tr.attention(float(rows.t.iloc[0]), rows.idx.to_numpy()[:5])
    assert set(att) <= {"rotation", "connection", "airport"}


def test_lgbm_trains_in_its_own_process():
    code = textwrap.dedent("""
        import numpy as np
        from ripple.data.build import chronological_split
        from ripple.data.features import FeatureStore, snapshot_rows
        from ripple.data.synthetic import SimConfig, generate
        from ripple.models.baseline_lgbm import LGBMQuantile
        store = FeatureStore(*generate(SimConfig(days=12, seed=5)))
        split = chronological_split(store.f, frac=(0.6, 0.2), gap_days=0)
        m = LGBMQuantile(rounds=20).fit(store, snapshot_rows(store, split["train"]),
                                       snapshot_rows(store, split["val"]))
        p = m.predict(store, snapshot_rows(store, split["test"]).iloc[:100])
        assert p["dep"].shape == (100, 19) and np.all(np.diff(p["dep"], axis=1) >= 0)
        print("ok")
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip().endswith("ok")
