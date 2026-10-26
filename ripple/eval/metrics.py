"""Probabilistic forecast metrics on a shared quantile grid."""

from __future__ import annotations

import numpy as np

from ripple import config

TAUS = np.array(config.QUANTILES)


def pinball(y: np.ndarray, q: np.ndarray, taus: np.ndarray = TAUS) -> np.ndarray:
    """Mean pinball loss per row, shape (n,)."""
    d = y[:, None] - q
    return np.mean(np.maximum(taus * d, (taus - 1) * d), axis=1)


def crps(y: np.ndarray, q: np.ndarray) -> float:
    """CRPS approximated by 2 x average pinball loss over the 19 equally spaced quantiles.

    It leaves out the outer 5% tails, where pinball loss is small, so it comes out slightly
    *above* the exact CRPS (about 4% for a Gaussian). Every model is scored the same way, so the
    comparison stays fair.
    """
    return float(2 * pinball(y, q).mean())


def coverage(y: np.ndarray, q: np.ndarray, lo: float, hi: float) -> tuple[float, float]:
    i, j = int(np.argmin(abs(TAUS - lo))), int(np.argmin(abs(TAUS - hi)))
    inside = (y >= q[:, i]) & (y <= q[:, j])
    return float(inside.mean()), float(np.mean(q[:, j] - q[:, i]))


def quantile_calibration(y: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Observed frequency of y <= q_tau for every tau (ideal = tau)."""
    return (y[:, None] <= q).mean(axis=0)


def summarize(y: np.ndarray, q: np.ndarray) -> dict[str, float]:
    ok = ~np.isnan(y)
    y, q = y[ok], q[ok]
    med = q[:, int(np.argmin(abs(TAUS - 0.5)))]
    cov90, w90 = coverage(y, q, 0.05, 0.95)
    cov80, w80 = coverage(y, q, 0.10, 0.90)
    cal = quantile_calibration(y, q)
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(y - med))),
        "crps": crps(y, q),
        "cov80": cov80,
        "cov90": cov90,
        "width80": w80,
        "width90": w90,
        "cal_error": float(np.mean(np.abs(cal - TAUS))),
    }


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 10):
    """Equal-width bins: (mean predicted, observed frequency, count) per bin."""
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    out = []
    for k in range(bins):
        m = b == k
        if m.sum():
            out.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return out


def binary_summary(p: np.ndarray, y: np.ndarray, base_rate: float) -> dict[str, float]:
    bs = brier(p, y)
    ref = brier(np.full_like(p, base_rate), y)
    pos = y == 1
    # AUC via the rank statistic
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    auc = (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / max(pos.sum() * (~pos).sum(), 1)
    return {
        "n": int(len(y)),
        "rate": float(y.mean()),
        "brier": bs,
        "brier_climatology": ref,
        "brier_skill": 1 - bs / ref if ref > 0 else float("nan"),
        "auc": float(auc),
    }
