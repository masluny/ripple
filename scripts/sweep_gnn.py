"""Small GNN hyper-parameter sweep on the validation split (results -> reports/<dataset>/gnn_sweep.json).

Usage: uv run python scripts/sweep_gnn.py [dataset]
"""

import json
import logging
import sys
import time

from ripple import config
from ripple.models.gnn import GNNTrainer
from ripple.pipeline import Experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
dataset = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
CONFIGS = {
    # "reg" was run with the first preprocessing (delays clipped at +-6 robust units), kept for the record
    "reg": dict(dropout=0.2, weight_decay=1e-3, lr=1e-3, epochs=10),
    "reg_v2": dict(dropout=0.2, weight_decay=1e-3, lr=1e-3, epochs=10),
    "reg_v2_plr": dict(dropout=0.2, weight_decay=1e-3, lr=1e-3, epochs=10, num_emb="plr"),
}
exp = Experiment(dataset)
out = config.REPORTS / dataset / "gnn_sweep.json"
out.parent.mkdir(parents=True, exist_ok=True)
res = json.loads(out.read_text()) if out.exists() else {}
for name, kw in CONFIGS.items():
    if name in res:
        continue
    t0 = time.time()
    tr = GNNTrainer(exp.store, exp.split, **kw).fit()
    best = min(h["val_crps"] for h in tr.history)
    res[name] = {
        "config": kw,
        "best_val_crps": best,
        "history": tr.history,
        "secs": time.time() - t0,
    }
    tr.save(config.ARTIFACTS / dataset / f"sweep_{name}")
    out.write_text(json.dumps(res, indent=1))
    logging.info("%s: best val CRPS %.3f", name, best)
