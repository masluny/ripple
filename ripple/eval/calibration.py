"""Post-hoc quantile recalibration, fitted on the validation split and applied to the test split.

For every quantile level tau we add the offset that makes the validation coverage exact:
    delta_tau = empirical tau-quantile of (y - q_tau) on validation
This is split-conformal calibration applied to each quantile separately. Both models get the same
treatment, so the comparison stays fair. The raw (uncalibrated) metrics are reported too.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ripple.eval.metrics import TAUS


def fit_offsets(y: np.ndarray, q: np.ndarray) -> np.ndarray:
    ok = ~np.isnan(y)
    d = y[ok, None] - q[ok]
    return np.array([np.quantile(d[:, k], t) for k, t in enumerate(TAUS)])


def apply_offsets(q: np.ndarray, off: np.ndarray) -> np.ndarray:
    return np.sort(q + off[None, :], axis=1)


def save(path: Path, offsets: dict[str, np.ndarray]) -> None:
    path.write_text(json.dumps({k: v.tolist() for k, v in offsets.items()}))


def load(path: Path) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    return {k: np.array(v) for k, v in json.loads(path.read_text()).items()}
