"""Precompute replay data for the dashboard (no model code runs inside the API server).

For a few test-set days and each model: every 30 minutes, predictions for all not-yet-departed
flights, P(miss) for WAW connections departing in the next 6 hours, and occlusion attribution.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ripple import config, isolate
from ripple.data.features import EPOCH, SLOT, snapshot_rows
from ripple.eval import calibration
from ripple.eval.connections import (
    enroute_deviation_quantiles,
    inbound_arrival_quantiles,
    miss_probability,
)
from ripple.eval.explain import GROUPS, describe_cause
from ripple.pipeline import Experiment

log = logging.getLogger(__name__)
WAW = ZoneInfo("Europe/Warsaw")
Q_OUT = [0, 4, 9, 14, 18]  # indices of q05, q25, q50, q75, q95


def demo_dir(dataset: str) -> Path:
    return config.ARTIFACTS / dataset / "demo"


def pick_days(exp: Experiment, n: int = 4) -> list[str]:
    """Worst, a weather-heavy, a median and a calm day from the test split."""
    f = exp.store.f[exp.split["test"]].copy()
    f["day"] = f.sched_dep.dt.tz_convert(WAW).dt.date.astype(str)
    f["dd"] = (f.actual_dep - f.sched_dep).dt.total_seconds() / 60
    by = f.groupby("day").dd.mean().iloc[1:-1]  # skip partial edge days
    picks = [by.idxmax(), by.index[len(by) // 2], by.idxmin()]
    if "cause_weather" in f:
        w = f.groupby("day")[["cause_weather", "cause_atfm"]].sum().sum(1).iloc[1:-1]
        picks.insert(1, w.drop(picks, errors="ignore").idxmax())
    return list(dict.fromkeys(picks))[:n]


def _day_rows(exp: Experiment, day: str) -> tuple[pd.DataFrame, np.ndarray]:
    st = exp.store
    start = pd.Timestamp(day, tz=WAW).replace(hour=4).tz_convert("UTC")
    t0 = (start - EPOCH).total_seconds() / 60
    times = t0 + SLOT * np.arange(40)  # 04:00 - 23:30 local
    rows = snapshot_rows(st, np.ones(st.n, bool))
    rows = rows[rows.t.isin(times)].reset_index(drop=True)
    return rows, times


def build_day(exp: Experiment, day: str, variant: str) -> dict:
    st = exp.store
    rows, times = _day_rows(exp, day)
    # connections whose outbound leaves within 6 h of each frame
    pairs = st.pairs.copy()
    pairs["in_idx"] = st.id2idx.loc[pairs.in_id].to_numpy()
    pairs["out_idx"] = st.id2idx.loc[pairs.out_id].to_numpy()
    creq = []
    for t in times:
        lead = st.sd[pairs.out_idx] - t
        sel = pairs[(lead > 0) & (lead <= 360) & (st.ad[pairs.out_idx] > t)]
        creq.append(sel.assign(t=t))
    creq = pd.concat(creq, ignore_index=True)
    extra = pd.concat([creq[["in_idx", "t"]].rename(columns={"in_idx": "idx"}), rows[["idx", "t"]]])
    req = extra.drop_duplicates().sort_values(["t", "idx"]).reset_index(drop=True)
    pred = isolate.run(isolate.predict_variant, exp.dataset, variant, req)
    off = calibration.load(exp.art_dir / variant / "calibration.json")
    if off is not None:  # same validation-fitted recalibration as in the reported metrics
        pred = {tgt: calibration.apply_offsets(pred[tgt], off[tgt]) for tgt in pred}
    key = pd.Index(req.idx.to_numpy() * 10_000_000 + (req.t.to_numpy() // SLOT).astype(np.int64))

    def take(idx, t, tgt):
        return pred[tgt][
            key.get_indexer(np.asarray(idx) * 10_000_000 + (np.asarray(t) // SLOT).astype(np.int64))
        ]

    enroute_q = enroute_deviation_quantiles(st, np.where(exp.split["train"])[0])
    in_q = inbound_arrival_quantiles(st, creq.in_idx.to_numpy(), creq.t.to_numpy(),
                                     take(creq.in_idx, creq.t, "arr"), enroute_q)  # fmt: skip
    creq["p"] = miss_probability(creq.slack.to_numpy(), in_q, take(creq.out_idx, creq.t, "dep"))

    attr = isolate.run(isolate.attribute_variant, exp.dataset, variant, rows[["idx", "t"]])
    attr_key = pd.Index(
        attr.idx.to_numpy() * 10_000_000 + (attr.t.to_numpy() // SLOT).astype(np.int64)
    )

    # flights shown on this day: anything scheduled between 00:00 and +30 h local
    d0 = (pd.Timestamp(day, tz=WAW).tz_convert("UTC") - EPOCH).total_seconds() / 60
    shown = np.where((st.sd >= d0 - 600) & (st.sd <= d0 + 1800))[0]
    f = st.f
    ap = {k: v.iata for k, v in config.AIRPORTS.items()}

    def m(x):
        return None if np.isnan(x) else round(float(x - d0), 1)

    flights = [
        dict(id=int(i), no=f.flight_no.iat[i], tail=f["tail"].iat[i], type=f.ac_type.iat[i],
             dep=ap.get(f.dep_airport.iat[i]), arr=ap.get(f.arr_airport.iat[i]),
             sd=m(st.sd[i]), sa=m(st.sa[i]), ad=m(st.ad[i]), aa=m(st.aa[i]),
             prev=int(st.prev[i]) if st.prev[i] >= 0 else None,
             truth=f.true_cause.iat[i] if "true_cause" in f else None)
        for i in shown
    ]  # fmt: skip
    frames = []
    groups = list(GROUPS)
    for t in times:
        r = rows[rows.t == t]
        qd, qa = take(r.idx, r.t, "dep"), take(r.idx, r.t, "arr")
        ai = attr_key.get_indexer(r.idx.to_numpy() * 10_000_000 + int(t // SLOT))
        a = attr.iloc[ai]
        preds = []
        for j, i in enumerate(r.idx.to_numpy()):
            cause = a.cause.iat[j]
            deltas = [round(float(a[f"d_{g}"].iat[j]), 1) for g in groups]
            pf = None
            if cause == "rotation" and st.prev[i] >= 0:
                k = key.get_indexer([st.prev[i] * 10_000_000 + int(t // SLOT)])[0]
                pf = float(pred["arr"][k, 9]) if k >= 0 else None
            text = describe_cause(exp, int(i), t, cause, max(deltas), pf) if cause != "none" else ""
            preds.append([int(i), *[round(float(v), 1) for v in qd[j, Q_OUT]],
                          *[round(float(v), 1) for v in qa[j, Q_OUT]], cause, deltas, text])  # fmt: skip
        c = creq[creq.t == t].sort_values("p", ascending=False).head(60)
        conns = [[int(x.in_idx), int(x.out_idx), round(float(x.slack)), round(float(x.p), 3)]
                 for x in c.itertuples()]  # fmt: skip
        frames.append({"t": round(float(t - d0), 1), "preds": preds, "conns": conns})
    return {"day": day, "model": variant, "groups": groups, "flights": flights, "frames": frames,
            "pred_cols": ["id", "d05", "d25", "d50", "d75", "d95", "a05", "a25", "a50", "a75", "a95",
                          "cause", "deltas", "text"]}  # fmt: skip


def build_demo(dataset: str = "synthetic", variants=("gnn", "lgbm"), n_days: int = 4) -> Path:
    exp = Experiment(dataset)
    out = demo_dir(dataset)
    out.mkdir(parents=True, exist_ok=True)
    days = pick_days(exp, n_days)
    index = {"dataset": dataset, "days": days, "models": [], "airports": [
        dict(icao=a.icao, iata=a.iata, name=a.name, lat=a.lat, lon=a.lon) for a in config.AIRPORTS.values()]}  # fmt: skip
    for v in variants:
        if not (exp.art_dir / v).exists():
            continue
        index["models"].append(v)
        for day in days:
            log.info("demo %s %s", v, day)
            (out / f"{day}_{v}.json").write_text(json.dumps(build_day(exp, day, v)))
    metrics_path = exp.report_dir / "metrics.json"
    if metrics_path.exists():
        index["metrics"] = json.loads(metrics_path.read_text())
    (out / "index.json").write_text(json.dumps(index))
    return out
