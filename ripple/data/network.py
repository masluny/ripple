"""Rotations, connections and airport/time neighbourhoods - the three edge types of the graph."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ripple import config

_EPOCH = pd.Timestamp("2000-01-01", tz="UTC")
MAX_GROUND_H = 18  # a gap longer than this is a night stop / maintenance, not a rotation


def _minutes(s: pd.Series) -> np.ndarray:
    return ((s - _EPOCH).dt.total_seconds() / 60).to_numpy()


def add_rotations(df: pd.DataFrame) -> pd.DataFrame:
    """``prev_id``: previous leg flown by the same tail that ends where this leg starts."""
    df = df.sort_values(["tail", "sched_dep"]).copy()
    g = df.groupby("tail", sort=False)
    prev_id = g["flight_id"].shift(1)
    prev_arr_ap = g["arr_airport"].shift(1)
    prev_sched_arr = g["sched_arr"].shift(1)
    gap_h = (df.sched_dep - prev_sched_arr).dt.total_seconds() / 3600
    ok = (prev_arr_ap == df.dep_airport) & (gap_h > 0) & (gap_h < MAX_GROUND_H)
    df["prev_id"] = prev_id.where(ok).astype("Int64")
    df["turn_buffer"] = ((df.sched_dep - prev_sched_arr).dt.total_seconds() / 60).where(ok)
    return df.sort_values("sched_dep").reset_index(drop=True)


def connections(df: pd.DataFrame, window=config.CONNECTION_WINDOW_MIN) -> pd.DataFrame:
    """Candidate passenger connections at the hub: arrival -> departure with slack in ``window``.

    We have no passenger data, so this is the *schedule-feasible* set of connections. It is an
    upper bound on real itineraries: it includes pairs nobody actually books.
    """
    arr = df.loc[df.arr_airport == config.HUB, ["flight_id", "sched_arr", "dep_airport", "tail"]]
    dep = df.loc[df.dep_airport == config.HUB, ["flight_id", "sched_dep", "arr_airport", "tail"]]
    arr = arr.sort_values("sched_arr")
    dep = dep.sort_values("sched_dep")
    a_t = _minutes(arr.sched_arr)
    rows_in, rows_out = [], []
    for d in dep.itertuples():
        t = (d.sched_dep - _EPOCH).total_seconds() / 60
        lo, hi = np.searchsorted(a_t, [t - window[1], t - window[0]], side="left")
        rows_in.append(arr.flight_id.to_numpy()[lo:hi])
        rows_out.append(np.full(hi - lo, d.flight_id))
    pairs = pd.DataFrame(
        {"in_id": np.concatenate(rows_in or [[]]), "out_id": np.concatenate(rows_out or [[]])}
    ).astype(int)
    pairs = pairs.merge(arr[["flight_id", "sched_arr", "dep_airport", "tail"]], left_on="in_id",
                        right_on="flight_id").drop(columns="flight_id")  # fmt: skip
    pairs = pairs.merge(dep[["flight_id", "sched_dep", "arr_airport", "tail"]], left_on="out_id",
                        right_on="flight_id", suffixes=("_in", "_out")).drop(columns="flight_id")  # fmt: skip
    # A same-tail pair is a through flight, not a connection; an O&D that returns home is not one either.
    pairs = pairs[(pairs.tail_in != pairs.tail_out) & (pairs.dep_airport != pairs.arr_airport)]
    pairs["slack"] = (pairs.sched_dep - pairs.sched_arr).dt.total_seconds() / 60
    return pairs[["in_id", "out_id", "slack"]].reset_index(drop=True)


def realised_miss(pairs: pd.DataFrame, df: pd.DataFrame, mct: float = config.MCT_MIN) -> pd.Series:
    """1 if the arriving flight landed later than (outbound actual departure - MCT)."""
    idx = df.set_index("flight_id")
    a = idx.loc[pairs.in_id, "actual_arr"].to_numpy()
    d = idx.loc[pairs.out_id, "actual_dep"].to_numpy()
    gap = (d - a) / np.timedelta64(1, "m")
    return pd.Series((gap < mct).astype(float), index=pairs.index)
