"""Assemble a dataset (flights + weather) under data/processed/<name>/."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ripple import config
from ripple.data import sources
from ripple.data.schema import validate_flights

log = logging.getLogger(__name__)


def dataset_dir(name: str) -> Path:
    return config.PROCESSED / name


def build_synthetic(days: int = 365, seed: int = 7, start: str = "2025-01-01") -> Path:
    from ripple.data.synthetic import SimConfig, generate

    flights, wx, fc = generate(SimConfig(start=start, days=days, seed=seed))
    return _save("synthetic", validate_flights(flights), wx, fc)


def build_real() -> Path:
    parts = []
    adb = sources.load_aerodatabox()
    if len(adb):
        parts.append(sources.enrich_with_opensky(adb))
    ec = sources.load_eurocontrol()
    if len(ec):
        parts.append(ec)
    if not parts:
        raise FileNotFoundError(
            "No scheduled-time data yet. Add AERODATABOX_KEY or EUROCONTROL files, see "
            "docs/data_sources.md"
        )
    flights = sources.to_canonical(pd.concat(parts, ignore_index=True))
    wx = sources.load_metar()
    fc = sources.load_taf()
    return _save("real", validate_flights(flights), wx, fc)


def _save(name: str, flights: pd.DataFrame, wx: pd.DataFrame, fc: pd.DataFrame) -> Path:
    d = dataset_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    flights.to_parquet(d / "flights.parquet", index=False)
    wx.to_parquet(d / "weather_obs.parquet", index=False)
    fc.to_parquet(d / "weather_fcst.parquet", index=False)
    log.info("saved %s: %d flights, %d obs, %d fcst", name, len(flights), len(wx), len(fc))
    return d


def load(name: str):
    d = dataset_dir(name)
    if not (d / "flights.parquet").exists():
        raise FileNotFoundError(f"run `ripple build --source {name}` first")
    return (
        pd.read_parquet(d / "flights.parquet"),
        pd.read_parquet(d / "weather_obs.parquet"),
        pd.read_parquet(d / "weather_fcst.parquet"),
    )


def chronological_split(flights: pd.DataFrame, frac=(0.70, 0.15), gap_days: int = 1):
    """Train / val / test by scheduled departure date with a gap so rotations do not straddle."""
    days = flights.sched_dep.dt.floor("D")
    u = days.drop_duplicates().sort_values().reset_index(drop=True)
    n = len(u)
    d1, d2 = u[int(n * frac[0])], u[int(n * (frac[0] + frac[1]))]
    gap = pd.Timedelta(days=gap_days)
    return {
        "train": (days < d1 - gap).to_numpy(),
        "val": ((days >= d1) & (days < d2 - gap)).to_numpy(),
        "test": (days >= d2).to_numpy(),
    }
