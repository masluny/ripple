"""P(missed connection) at WAW from marginal delay quantiles.

For an inbound arrival A and outbound departure D with scheduled slack s:

    miss  <=>  arr_delay(A) - dep_delay(D) > s - MCT

We sample both delays from their predicted quantile functions (piecewise-linear inverse CDF,
linear tails) and count. The two marginals are treated as independent. That is wrong when D is
held for A, so a hold makes the true P(miss) lower than this estimate.

What the inbound leg "knows" at prediction time t:
  * already landed            -> its arrival delay is a known number
  * airborne                  -> known departure delay + the empirical en-route deviation
  * still on the ground       -> the model's arrival-delay quantiles for (A, t)
"""

from __future__ import annotations

import numpy as np

from ripple import config
from ripple.eval.metrics import TAUS


def sample_quantiles(q: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Draw samples from rows of a quantile matrix on the uniform TAUS grid."""
    n, k = q.shape
    u = rng.uniform(0, 1, (n, n_samples))
    step = TAUS[1] - TAUS[0]
    pos = u / step - 1  # position on the grid, 0 = TAUS[0]
    i = np.clip(np.floor(pos).astype(int), 0, k - 2)
    frac = pos - i  # may be <0 or >1 in the tails -> linear extrapolation
    lo = np.take_along_axis(q, i, axis=1)
    hi = np.take_along_axis(q, i + 1, axis=1)
    return lo + frac * (hi - lo)


def miss_probability(
    slack: np.ndarray,
    in_arr_q: np.ndarray,
    out_dep_q: np.ndarray,
    n_samples: int = 400,
    mct: float = config.MCT_MIN,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = sample_quantiles(in_arr_q, n_samples, rng)
    d = sample_quantiles(out_dep_q, n_samples, rng)
    return ((a - d) > (slack[:, None] - mct)).mean(axis=1)


def inbound_arrival_quantiles(store, in_idx, t, model_arr_q, enroute_q):
    """Replace model quantiles with known facts for inbound legs that are airborne or landed."""
    q = model_arr_q.copy()
    aa, ad = store.aa[in_idx], store.ad[in_idx]
    landed = aa <= t
    airborne = (ad <= t) & ~landed
    q[landed] = (aa - store.sa[in_idx])[landed, None]
    dep_delay = (ad - store.sd[in_idx])[airborne]
    q[airborne] = dep_delay[:, None] + enroute_q[None, :]
    return q


def enroute_deviation_quantiles(store, train_idx) -> np.ndarray:
    """Empirical quantiles of (arrival delay - departure delay) on training flights."""
    dev = (store.aa[train_idx] - store.sa[train_idx]) - (store.ad[train_idx] - store.sd[train_idx])
    dev = dev[~np.isnan(dev)]
    return np.quantile(dev, TAUS)
