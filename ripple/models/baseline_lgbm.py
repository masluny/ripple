"""LightGBM quantile-regression baseline: one booster per (target, quantile).

It sees exactly the same as-of features as the GNN (FeatureStore), including hand-made
propagation features (projected reactionary delay along the rotation, feeder delays). The GNN has
to beat it on information it extracts from the graph, not on better inputs.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from ripple import config
from ripple.data.features import CATEGORICAL, FeatureStore

log = logging.getLogger(__name__)
TARGETS = ("dep", "arr")

PARAMS = dict(
    objective="quantile",
    learning_rate=0.06,
    num_leaves=63,
    min_child_samples=200,
    feature_fraction=0.8,
    bagging_fraction=0.7,
    bagging_freq=1,
    lambda_l2=1.0,
    max_bin=127,
    verbose=-1,
    num_threads=0,
)


class LGBMQuantile:
    def __init__(self, exclude: tuple[str, ...] = (), rounds: int = 600):
        self.exclude = exclude
        self.rounds = rounds
        self.models: dict[tuple[str, float], lgb.Booster] = {}
        self.names: list[str] = []

    def _x(self, store: FeatureStore, rows: pd.DataFrame) -> pd.DataFrame:
        x = store.features(rows.idx.to_numpy(), rows.t.to_numpy())
        df = pd.DataFrame(x, columns=store.names)
        keep = (
            [c for c in df.columns if not c.startswith(self.exclude)] if self.exclude else list(df)
        )
        self.names = keep
        return df[keep]

    def fit(self, store: FeatureStore, train: pd.DataFrame, val: pd.DataFrame) -> LGBMQuantile:
        xtr, xva = self._x(store, train), self._x(store, val)
        ytr, yva = store.targets(train.idx.to_numpy()), store.targets(val.idx.to_numpy())
        cats = [c for c in CATEGORICAL if c in self.names]
        for k, target in enumerate(TARGETS):
            dtr = lgb.Dataset(xtr, ytr[:, k], categorical_feature=cats, free_raw_data=False)
            dva = lgb.Dataset(xva, yva[:, k], categorical_feature=cats, reference=dtr)
            for tau in config.QUANTILES:
                t0 = time.time()
                b = lgb.train(
                    {**PARAMS, "alpha": tau},
                    dtr,
                    num_boost_round=self.rounds,
                    valid_sets=[dva],
                    callbacks=[lgb.early_stopping(40, verbose=False)],
                )
                self.models[(target, tau)] = b
                log.info(
                    "%s q%.2f: %d rounds, %.1fs", target, tau, b.best_iteration, time.time() - t0
                )
        return self

    def predict(self, store: FeatureStore, rows: pd.DataFrame) -> dict[str, np.ndarray]:
        x = self._x(store, rows)
        out = {}
        for target in TARGETS:
            q = np.column_stack(
                [
                    self.models[(target, tau)].predict(
                        x, num_iteration=self.models[(target, tau)].best_iteration
                    )
                    for tau in config.QUANTILES
                ]  # fmt: skip
            )
            out[target] = np.sort(q, axis=1)  # rearrangement fixes quantile crossing
        return out

    def importance(self) -> pd.Series:
        b = self.models[("dep", 0.5)]
        return pd.Series(b.feature_importance("gain"), index=b.feature_name()).sort_values(
            ascending=False
        )

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        for (target, tau), b in self.models.items():
            b.save_model(str(path / f"{target}_{tau:.2f}.txt"), num_iteration=b.best_iteration)
        (path / "features.txt").write_text("\n".join(self.names))
        (path / "exclude.txt").write_text("\n".join(self.exclude))

    @classmethod
    def load(cls, path: Path) -> LGBMQuantile:
        excl = tuple(x for x in (path / "exclude.txt").read_text().splitlines() if x)
        m = cls(exclude=excl)
        for target in TARGETS:
            for tau in config.QUANTILES:
                b = lgb.Booster(model_file=str(path / f"{target}_{tau:.2f}.txt"))
                b.best_iteration = b.current_iteration()
                m.models[(target, tau)] = b
        return m
