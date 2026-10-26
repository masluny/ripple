"""Run model code in a fresh process.

LightGBM (Homebrew libomp) and PyTorch (bundled libomp) load two different OpenMP runtimes. Inside
one process that ends in a segfault or a deadlock on macOS. Every train/predict call for a model
kind therefore runs in its own spawned process, and the parent never imports either library.
"""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor


def run(fn, *args, **kwargs):
    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as ex:
        return ex.submit(fn, *args, **kwargs).result()


# --- picklable task functions --------------------------------------------------------------------
def _setup_logging():
    import logging

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True
    )


def train_variant(dataset: str, variant: str, overrides: dict) -> None:
    _setup_logging()
    from ripple.pipeline import Experiment

    Experiment(dataset).train(variant, **overrides)


def predict_variant(dataset: str, variant: str, rows):
    from ripple.pipeline import Experiment

    exp = Experiment(dataset)
    return exp.load_model(variant).predict(exp.store, rows)


def attribute_variant(dataset: str, variant: str, rows):
    from ripple.eval.explain import attribute
    from ripple.pipeline import VARIANTS, Experiment

    exp = Experiment(dataset)
    return attribute(exp, exp.load_model(variant), VARIANTS[variant][0], rows)


def attention_variant(dataset: str, variant: str, t: float, idx):
    from ripple.pipeline import Experiment

    exp = Experiment(dataset)
    return exp.load_model(variant).attention(t, idx)
