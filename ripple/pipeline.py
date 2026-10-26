"""Train / evaluate orchestration shared by the CLI and the API."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd

from ripple import config
from ripple.data.build import chronological_split, load
from ripple.data.features import SLOT, FeatureStore, snapshot_rows
from ripple.eval import metrics
from ripple.eval.connections import (
    enroute_deviation_quantiles,
    inbound_arrival_quantiles,
    miss_probability,
)

log = logging.getLogger(__name__)


def _key(idx, t) -> np.ndarray:
    """Unique int64 key for a (flight index, prediction time) pair."""
    return np.asarray(idx, dtype=np.int64) * 10_000_000 + (np.asarray(t) // SLOT).astype(np.int64)


# Model variants: name -> (kind, options). Ablations are just variants.
VARIANTS = {
    "lgbm": ("lgbm", {}),
    "lgbm_noweather": ("lgbm", {"exclude": ("wx_", "fc_")}),
    "gnn": ("gnn", {}),
    "gnn_norot": ("gnn", {"drop_edges": ("rotation",)}),
    "gnn_noconn": ("gnn", {"drop_edges": ("connection",)}),
    "gnn_noapt": ("gnn", {"drop_edges": ("airport",)}),
    "gnn_noweather": ("gnn", {"exclude": ("wx_", "fc_")}),
    "gnn_nogru": ("gnn", {"temporal": False}),
}


@dataclass
class Experiment:
    dataset: str = "synthetic"
    max_train_rows: int = 800_000
    seed: int = 0

    @cached_property
    def store(self) -> FeatureStore:
        f, wx, fc = load(self.dataset)
        return FeatureStore(f, wx, fc)

    @cached_property
    def split(self) -> dict[str, np.ndarray]:
        return chronological_split(self.store.f)

    def rows(self, part: str, max_rows: int | None = None) -> pd.DataFrame:
        r = snapshot_rows(self.store, self.split[part])
        if max_rows and len(r) > max_rows:
            r = r.sample(max_rows, random_state=self.seed).sort_values(["t", "idx"])
        return r.reset_index(drop=True)

    @property
    def art_dir(self):
        return config.ARTIFACTS / self.dataset

    @property
    def report_dir(self):
        return config.REPORTS / self.dataset

    # --- models ------------------------------------------------------------------------------
    def train(self, variant: str, **overrides):
        kind, opts = VARIANTS[variant]
        opts = {**opts, **overrides}
        t0 = time.time()
        if kind == "lgbm":
            from ripple.models.baseline_lgbm import LGBMQuantile

            model = LGBMQuantile(exclude=opts.get("exclude", ()))
            model.fit(
                self.store, self.rows("train", self.max_train_rows), self.rows("val", 200_000)
            )
        else:
            from ripple.models.gnn import GNNTrainer

            model = GNNTrainer(self.store, self.split, **opts)
            model.fit()
        model.save(self.art_dir / variant)
        log.info("trained %s in %.0fs", variant, time.time() - t0)
        return model

    def load_model(self, variant: str):
        kind, _ = VARIANTS[variant]
        if kind == "lgbm":
            from ripple.models.baseline_lgbm import LGBMQuantile

            return LGBMQuantile.load(self.art_dir / variant)
        from ripple.models.gnn import GNNTrainer

        return GNNTrainer.load(self.art_dir / variant, self.store, self.split)

    # --- evaluation ------------------------------------------------------------------------
    @cached_property
    def eval_requests(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Test-set prediction requests: horizon rows and connection-pair rows."""
        st, test = self.store, self.split["test"]
        idx = np.where(test & ~np.isnan(st.ad))[0]
        hz = []
        for h in config.HORIZONS:
            t = np.floor((st.sd[idx] - 60 * h) / SLOT) * SLOT
            hz.append(pd.DataFrame({"idx": idx, "t": t, "h": h}))
        hz = pd.concat(hz, ignore_index=True)

        pairs = st.pairs.copy()
        pairs["in_idx"] = st.id2idx.loc[pairs.in_id].to_numpy()
        pairs["out_idx"] = st.id2idx.loc[pairs.out_id].to_numpy()
        pairs = pairs[test[pairs.out_idx] & test[pairs.in_idx]]
        cp = []
        for h in config.HORIZONS:
            t = np.floor((st.sd[pairs.out_idx] - 60 * h) / SLOT) * SLOT
            cp.append(pairs.assign(t=t, h=h))
        cp = pd.concat(cp, ignore_index=True)
        mct = config.MCT_MIN
        cp["miss"] = ((st.ad[cp.out_idx] - st.aa[cp.in_idx]) < mct).astype(float)
        return hz, cp

    def predict_requests(self, model) -> tuple[pd.DataFrame, dict]:
        hz, cp = self.eval_requests
        req = pd.concat([
            hz[["idx", "t"]],
            cp[["out_idx", "t"]].rename(columns={"out_idx": "idx"}),
            cp[["in_idx", "t"]].rename(columns={"in_idx": "idx"}),
        ]).drop_duplicates().sort_values(["t", "idx"]).reset_index(drop=True)  # fmt: skip
        pred = model.predict(self.store, req)
        return req, pred

    def evaluate(self, variants: list[str]) -> dict:
        st = self.store
        hz, cp = self.eval_requests
        train_idx = np.where(self.split["train"])[0]
        enroute_q = enroute_deviation_quantiles(st, train_idx)
        base_rate = None
        results: dict = {"dataset": self.dataset, "models": {}}
        self.report_dir.mkdir(parents=True, exist_ok=True)
        pred_store = {}
        for v in variants:
            path = self.art_dir / v
            if not path.exists():
                log.warning("skip %s: not trained", v)
                continue
            model = self.load_model(v)
            t0 = time.time()
            req, pred = self.predict_requests(model)
            log.info("predicted %s: %d requests in %.0fs", v, len(req), time.time() - t0)
            index = pd.Index(_key(req.idx, req.t))

            def take(idx, t, target, _index=index, _pred=pred):
                pos = _index.get_indexer(_key(idx, t))
                assert (pos >= 0).all(), "missing prediction"
                return _pred[target][pos]

            res: dict = {"horizons": {}, "connections": {}}
            y = st.targets(hz.idx.to_numpy())
            for h in config.HORIZONS:
                m = (hz.h == h).to_numpy()
                res["horizons"][h] = {
                    tgt: metrics.summarize(y[m, k], take(hz.idx[m], hz.t[m], tgt))
                    for k, tgt in enumerate(("dep", "arr"))
                }
            p_all = np.zeros(len(cp))
            for h in config.HORIZONS:
                m = (cp.h == h).to_numpy()
                c = cp[m]
                in_q = take(c.in_idx, c.t, "arr")
                in_q = inbound_arrival_quantiles(st, c.in_idx.to_numpy(), c.t.to_numpy(), in_q,
                                                 enroute_q)  # fmt: skip
                out_q = take(c.out_idx, c.t, "dep")
                p = miss_probability(c.slack.to_numpy(), in_q, out_q)
                p_all[m] = p
                if base_rate is None:
                    tr = st.pairs[self.split["train"][st.id2idx.loc[st.pairs.out_id].to_numpy()]]
                    base_rate = float(
                        ((st.ad[st.id2idx.loc[tr.out_id]] - st.aa[st.id2idx.loc[tr.in_id]])
                         < config.MCT_MIN).mean()
                    )  # fmt: skip
                res["connections"][h] = metrics.binary_summary(p, c.miss.to_numpy(), base_rate)
                res["connections"][h]["reliability"] = metrics.reliability(p, c.miss.to_numpy())
            # quantile calibration curve over all horizons (departure)
            qd = take(hz.idx, hz.t, "dep")
            res["calibration_dep"] = metrics.quantile_calibration(y[:, 0], qd).tolist()
            results["models"][v] = res
            pred_store[v] = (req, pred, p_all)
            self._save_predictions(v, req, pred, p_all)
        (self.report_dir / "metrics.json").write_text(json.dumps(results, indent=1))
        return results

    def _save_predictions(self, v, req, pred, p_conn) -> None:
        out = req.copy()
        for tgt in ("dep", "arr"):
            for j, tau in enumerate(config.QUANTILES):
                out[f"{tgt}_q{int(round(tau * 100)):02d}"] = pred[tgt][:, j].astype(np.float32)
        d = self.art_dir / v
        out.to_parquet(d / "test_predictions.parquet", index=False)
        hz, cp = self.eval_requests
        cp.assign(p_miss=p_conn).to_parquet(d / "test_connections.parquet", index=False)
