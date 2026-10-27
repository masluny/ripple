"""Where does the graph help? Test CRPS (departure delay) by what is known about the inbound aircraft.

Reads the predictions saved by `ripple evaluate` and writes reports/<dataset>/breakdown.md.
Usage: uv run python scripts/breakdown.py [dataset] [model_a] [model_b]
"""

import sys

import numpy as np
import pandas as pd

from ripple import config
from ripple.data.features import SLOT
from ripple.eval import metrics
from ripple.pipeline import Experiment

dataset = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
a, b = (sys.argv[2], sys.argv[3]) if len(sys.argv) > 3 else ("lgbm", "gnn")
exp = Experiment(dataset)
st = exp.store
hz, _ = exp.eval_requests
x = st.features(hz.idx.to_numpy(), hz.t.to_numpy())
prev_state = x[:, st.names.index("prev_state")]
hub = x[:, st.names.index("hub_dep")]
y = st.targets(hz.idx.to_numpy())[:, 0]
cols = [f"dep_q{int(round(t * 100)):02d}" for t in config.QUANTILES]
want = hz.idx.to_numpy() * 10_000_000 + (hz.t.to_numpy() // SLOT).astype(np.int64)


def quantiles(v: str) -> np.ndarray:
    p = pd.read_parquet(exp.art_dir / v / "test_predictions.parquet")
    have = p.idx.to_numpy() * 10_000_000 + (p.t.to_numpy() // SLOT).astype(np.int64)
    return p[cols].to_numpy()[pd.Index(have).get_indexer(want)]


qa, qb = quantiles(a), quantiles(b)
groups = {
    "all flights": np.ones(len(hz), bool),
    "inbound aircraft not departed yet": prev_state == 0,
    "inbound departed or landed": prev_state >= 2,
    "WAW departures": hub == 1,
}
lines = [f"# CRPS (departure delay, minutes) by inbound-aircraft state - {a} vs {b}", "",
         f"| horizon | subset | n | {a} | {b} | {b} vs {a} |", "|---|---|---|---|---|---|"]  # fmt: skip
for h in config.HORIZONS:
    for name, m in groups.items():
        mm = m & (hz.h.to_numpy() == h)
        ca, cb = metrics.crps(y[mm], qa[mm]), metrics.crps(y[mm], qb[mm])
        lines.append(
            f"| {h}h | {name} | {mm.sum()} | {ca:.2f} | {cb:.2f} | {100 * (cb - ca) / ca:+.1f}% |"
        )
# Day-block bootstrap of the CRPS difference (flights on the same day are not independent).
day = st.f.sched_dep.dt.floor("D").to_numpy()[hz.idx.to_numpy()]
d = 2 * (metrics.pinball(y, qb) - metrics.pinball(y, qa))
rng = np.random.default_rng(0)
lines += [
    "",
    f"Day-block bootstrap, 95% CI of mean CRPS difference ({b} - {a}, minutes; < 0 favours {b}):",
    "",
]
for h in config.HORIZONS:
    m = hz.h.to_numpy() == h
    per_day = pd.DataFrame({"day": day[m], "d": d[m]}).groupby("day").d.agg(["sum", "count"])
    draws = rng.integers(0, len(per_day), (2000, len(per_day)))
    boot = per_day["sum"].to_numpy()[draws].sum(1) / per_day["count"].to_numpy()[draws].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    lines.append(f"- {h}h: {d[m].mean():+.3f} [{lo:+.3f}, {hi:+.3f}] over {len(per_day)} days")
out = exp.report_dir / "breakdown.md"
out.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
