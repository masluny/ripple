"""Where does a predicted delay come from?

Occlusion attribution, the same for both models: for a target flight at time t, remove one group of
evidence and see how much the predicted median departure delay drops.

    rotation    - what is known about the inbound aircraft (+ rotation edges into the flight)
    connection  - what is known about feeder flights at WAW (+ connection edges)
    weather     - weather observations and forecasts at both ends (set to clear conditions)
    airport     - scheduled slot load and recent delays at the airports (+ airport/time edges)

delta_g = median(full) - median(without g). The largest positive delta is the reported cause.
On synthetic data the true cause is known, so attribution accuracy can be measured.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ripple import config

GROUPS = {
    "rotation": ["prev_state", "prev_delay", "chain_depth", "chain_delay", "chain_slack", "proj_reactionary"],
    "connection": ["feed_mean", "feed_max", "feed_frac_late", "feed_break"],
    "weather": ["wx_dep_severity", "wx_dep_vis_m", "wx_dep_ceil_ft", "wx_dep_wind_kt", "wx_dep_gust_kt",
                "wx_dep_ts", "wx_dep_sn", "wx_dep_fg", "wx_arr_severity", "wx_arr_vis_m", "wx_arr_ts",
                "wx_arr_sn", "fc_dep_sev", "fc_arr_sev"],
    "airport": ["dep_load", "arr_load", "dep_ap_recent_dep", "dep_ap_recent_n", "arr_ap_recent_arr",
                "hub_recent_dep", "hub_recent_arr"],
}  # fmt: skip
EDGE_OF_GROUP = {"rotation": "rotation", "connection": "connection", "airport": "airport"}
CLEAR = {"vis_m": 9999.0, "ceil_ft": 5000.0, "wind_kt": 8.0, "gust_kt": 8.0}
# which explanation counts as correct for each simulated cause
TRUE_TO_GROUP = {
    "reactionary": {"rotation"},
    "hold": {"connection"},
    "weather": {"weather"},
    "atfm": {"weather", "airport"},
    "congestion": {"airport"},
}


def _counterfactual(names: list[str], group: str, low_load: float = 4.0):
    cols = [names.index(c) for c in GROUPS[group] if c in names]

    def apply(x: np.ndarray, rows_mask: np.ndarray | None = None) -> np.ndarray:
        x = x.copy()
        m = slice(None) if rows_mask is None else rows_mask
        for c in cols:
            nm = names[c]
            if group == "weather":
                key = next((k for k in CLEAR if nm.endswith(k)), None)
                x[m, c] = CLEAR[key] if key else 0.0
            elif nm == "prev_state":
                x[m, c] = 0.0
            elif nm in ("dep_load", "arr_load"):
                x[m, c] = low_load
            elif nm == "dep_ap_recent_n":
                x[m, c] = 0.0
            else:
                x[m, c] = np.nan
        return x

    return apply


def attribute(exp, model, kind: str, rows: pd.DataFrame) -> pd.DataFrame:
    """Occlusion deltas per group for (idx, t) rows. ``kind`` is 'lgbm' or 'gnn'."""
    st = exp.store
    names = st.feature_names()
    full = model.predict(st, rows)["dep"][:, 9]
    out = pd.DataFrame({"idx": rows.idx.to_numpy(), "t": rows.t.to_numpy(), "q50": full})
    for g in GROUPS:
        cf = _counterfactual(names, g)
        if kind == "lgbm":
            orig = st.features

            def patched(idx, t, _orig=orig, _cf=cf):
                return _cf(_orig(idx, t))

            st.features = patched
            try:
                occ = model.predict(st, rows)["dep"][:, 9]
            finally:
                st.features = orig
        else:
            targets = rows.idx.to_numpy()

            def feat(x, idx, _cf=cf, _targets=targets):
                return _cf(x, np.isin(idx, _targets))

            spec = {"features": feat, "nodes": targets,
                    "drop_into": (EDGE_OF_GROUP[g],) if g in EDGE_OF_GROUP else ()}  # fmt: skip
            occ = model.predict(st, rows, occlude=spec)["dep"][:, 9]
        out[f"d_{g}"] = full - occ
    d = out[[f"d_{g}" for g in GROUPS]].to_numpy()
    out["cause"] = np.where(d.max(1) > 2.0, np.array(list(GROUPS))[d.argmax(1)], "none")
    return out


def attribution_accuracy(exp, attr: pd.DataFrame) -> dict:
    f = exp.store.f
    if "true_cause" not in f:
        return {}
    truth = f.true_cause.to_numpy()[attr.idx]
    res = {}
    ok_all = []
    for cause, allowed in TRUE_TO_GROUP.items():
        m = truth == cause
        if m.sum() == 0:
            continue
        hit = attr.cause.to_numpy()[m]
        acc = np.isin(hit, list(allowed)).mean()
        res[cause] = {"n": int(m.sum()), "accuracy": float(acc)}
        ok_all.append(np.isin(hit, list(allowed)))
    res["overall"] = float(np.concatenate(ok_all).mean()) if ok_all else float("nan")
    return res


def describe_cause(exp, idx: int, t: float, cause: str, delta: float,
                   prev_forecast: float | None = None) -> str:  # fmt: skip
    """Human-readable cause line for the dashboard. ``prev_forecast`` is the predicted arrival
    delay of the inbound leg, used when nothing has been observed about it yet."""
    st = exp.store
    f = st.f
    ap = {k: v.iata for k, v in config.AIRPORTS.items()}
    if cause == "rotation" and st.prev[idx] >= 0:
        p = st.prev[idx]
        state, dl = st.known_delay(np.array([p]), np.array([t]))
        status = {0: "not yet departed", 1: "overdue", 2: "airborne", 3: "landed"}[int(state[0])]
        if not np.isnan(dl[0]):
            dtxt = f"{dl[0]:+.0f} min"
        elif prev_forecast is not None:
            dtxt = f"forecast {prev_forecast:+.0f} min on arrival"
        else:
            dtxt = "delay unknown"
        return (f"Inherited from the previous leg of {f['tail'].iat[idx]}: {f.flight_no.iat[p]} "
                f"from {ap.get(f.dep_airport.iat[p], f.dep_airport.iat[p])} ({status}, {dtxt}); "
                f"adds ~{delta:.0f} min")  # fmt: skip
    if cause == "connection":
        return f"Waiting for late connecting feeders at WAW; adds ~{delta:.0f} min"
    if cause == "weather":
        return (f"Weather at {ap.get(f.dep_airport.iat[idx])} / {ap.get(f.arr_airport.iat[idx])} "
                f"(observed or forecast); adds ~{delta:.0f} min")  # fmt: skip
    if cause == "airport":
        return f"Congestion and recent delays at {ap.get(f.dep_airport.iat[idx])}; adds ~{delta:.0f} min"
    return "No single dominant driver"
