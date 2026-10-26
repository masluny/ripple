"""Leakage-free features for any (flight, prediction time t) pair.

Everything a feature uses must be knowable at time t:

* schedule facts (scheduled times, route, aircraft type, scheduled slot loads) - known in advance;
* actual times of *other* flights only if they happened at or before t (departed/arrived);
* a flight that should have left but has not is at least ``t - sched_dep`` late ("overdue");
* weather: the latest observation at or before t and the latest forecast issued at or before t.

The same ``FeatureStore.features(idx, t)`` call feeds LightGBM rows and GNN node features, so the two
models see exactly the same information about each flight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ripple import config
from ripple.data.network import add_rotations, connections

SLOT = 30  # minutes between prediction times
MIN_TURN = {"B788": 90, "B789": 110, "B38M": 45, "B738": 45, "E195": 40, "E190": 40, "E175": 35,
            "E170": 35, "E75S": 35, "E75L": 35, "DH8D": 30, "BCS3": 40}  # fmt: skip
DEFAULT_TURN = 40
TYPE_CODES = sorted(MIN_TURN)
AIRPORTS = list(config.AIRPORTS)
AP_IDX = {a: i for i, a in enumerate(AIRPORTS)}
WX_GRID_FIELDS = ["severity", "vis_m", "ceil_ft", "wind_kt", "gust_kt", "ts", "sn", "fg"]
EPOCH = pd.Timestamp("2000-01-01", tz="UTC")


def to_min(s: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """Timestamps -> float minutes since EPOCH (NaT -> NaN)."""
    s = pd.Series(s)
    out = (s - EPOCH).dt.total_seconds().to_numpy(dtype=float) / 60
    return out


@dataclass
class FeatureStore:
    flights: pd.DataFrame
    wx_obs: pd.DataFrame
    wx_fcst: pd.DataFrame | None = None
    feeder_max_slack: float = 150.0
    names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        f = add_rotations(self.flights)
        if "dep_delay" not in f:
            f["dep_delay"] = (f.actual_dep - f.sched_dep).dt.total_seconds() / 60
            f["arr_delay"] = (f.actual_arr - f.sched_arr).dt.total_seconds() / 60
        self.f = f
        self.n = len(f)
        self.id2idx = pd.Series(np.arange(self.n), index=f.flight_id.to_numpy())
        self.sd, self.sa = to_min(f.sched_dep), to_min(f.sched_arr)
        self.ad, self.aa = to_min(f.actual_dep), to_min(f.actual_arr)
        self.dep_ap = f.dep_airport.map(AP_IDX).fillna(-1).astype(int).to_numpy()
        self.arr_ap = f.arr_airport.map(AP_IDX).fillna(-1).astype(int).to_numpy()
        self.type_code = (
            f.ac_type.map({t: i for i, t in enumerate(TYPE_CODES)}).fillna(-1).to_numpy()
        )
        self.min_turn = f.ac_type.map(MIN_TURN).fillna(DEFAULT_TURN).to_numpy(dtype=float)
        prev = f.prev_id.map(self.id2idx)
        self.prev = prev.fillna(-1).astype(int).to_numpy()
        self.buffer = f.turn_buffer.to_numpy(dtype=float)
        hub = AP_IDX[config.HUB]
        self.hub_dep = (self.dep_ap == hub).astype(float)
        self.hub_arr = (self.arr_ap == hub).astype(float)
        lh = np.array([config.AIRPORTS[a].longhaul for a in AIRPORTS] + [False])
        self.longhaul = (lh[self.dep_ap] | lh[self.arr_ap]).astype(float)
        self.block = self.sa - self.sd
        local = f.sched_dep.dt.tz_convert(ZoneInfo("Europe/Warsaw"))
        self.hour = (local.dt.hour + local.dt.minute / 60).to_numpy(dtype=float)
        self.dow = local.dt.day_of_week.to_numpy(dtype=float)
        self.dep_load = self._slot_load(self.dep_ap, self.sd)
        self.arr_load = self._slot_load(self.arr_ap, self.sa)
        self._build_feeders()
        self._build_slot_grid()

    # --- static helpers ------------------------------------------------------------------------
    @staticmethod
    def _slot_load(ap: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Scheduled movements at the same airport within +-30 min (schedule-only, no leakage)."""
        out = np.zeros(len(t))
        for a in np.unique(ap):
            m = np.where(ap == a)[0]
            ts = np.sort(t[m])
            out[m] = np.searchsorted(ts, t[m] + 30, "right") - np.searchsorted(
                ts, t[m] - 30, "left"
            )
        return out

    def _build_feeders(self) -> None:
        pairs = connections(self.f)
        pairs = pairs[pairs.slack <= self.feeder_max_slack]
        self.pairs = pairs
        out_idx = self.id2idx.loc[pairs.out_id].to_numpy()
        in_idx = self.id2idx.loc[pairs.in_id].to_numpy()
        order = np.argsort(out_idx, kind="stable")
        self.feed_in = in_idx[order]
        counts = np.bincount(out_idx, minlength=self.n)
        self.feed_ptr = np.concatenate([[0], np.cumsum(counts)])
        self.feed_slack = pairs.slack.to_numpy()[order]

    def _build_slot_grid(self) -> None:
        """Per airport and 30-min slot: recent observed delays and the latest weather report."""
        lo = np.nanmin(self.sd) - 24 * 60
        hi = np.nanmax(self.sa) + 24 * 60
        self.t0 = np.floor(lo / SLOT) * SLOT
        n_slots = int((hi - self.t0) // SLOT) + 2
        n_ap = len(AIRPORTS)
        self.n_slots = n_slots

        def recent(ap: np.ndarray, t_event: np.ndarray, val: np.ndarray):
            ok = ~np.isnan(t_event) & (ap >= 0)
            # event observed at t_event belongs to the first slot boundary >= t_event
            s = np.ceil((t_event[ok] - self.t0) / SLOT).astype(int)
            s_sum = np.zeros((n_slots + 1, n_ap))
            s_cnt = np.zeros((n_slots + 1, n_ap))
            np.add.at(s_sum, (np.clip(s, 0, n_slots), ap[ok]), val[ok])
            np.add.at(s_cnt, (np.clip(s, 0, n_slots), ap[ok]), 1)
            # rolling 2 slots = last 60 minutes
            r_sum = s_sum + np.vstack([np.zeros((1, n_ap)), s_sum[:-1]])
            r_cnt = s_cnt + np.vstack([np.zeros((1, n_ap)), s_cnt[:-1]])
            with np.errstate(invalid="ignore", divide="ignore"):
                return r_sum / r_cnt, r_cnt

        dd = self.ad - self.sd
        ar = self.aa - self.sa
        self.g_dep_mean, self.g_dep_cnt = recent(self.dep_ap, self.ad, dd)
        self.g_arr_mean, self.g_arr_cnt = recent(self.arr_ap, self.aa, ar)

        # Weather: latest observation at or before each slot boundary.
        slot_times = EPOCH + pd.to_timedelta(self.t0 + SLOT * np.arange(n_slots), unit="min")
        self.wx = np.full((n_slots, n_ap, len(WX_GRID_FIELDS)), np.nan, dtype=np.float32)
        grid = pd.DataFrame({"time": slot_times})
        obs = self.wx_obs.copy()
        obs["time"] = pd.to_datetime(obs["time"], utc=True).astype("datetime64[ns, UTC]")
        grid["time"] = grid["time"].astype("datetime64[ns, UTC]")
        for icao, g in obs.groupby("icao"):
            if icao not in AP_IDX:
                continue
            m = pd.merge_asof(grid, g.sort_values("time"), on="time", direction="backward",
                              tolerance=pd.Timedelta("3h"))  # fmt: skip
            self.wx[:, AP_IDX[icao], :] = m[WX_GRID_FIELDS].to_numpy(dtype=np.float32)

        # Forecasts: fc[issue_slot, airport, lead_hour] where lead is counted from the issue slot,
        # plus fc_issue[slot, airport] = latest issue slot <= slot (forward-filled index).
        self.fc = None
        self.fc_lead = 36
        if self.wx_fcst is not None and len(self.wx_fcst):
            fc = self.wx_fcst[self.wx_fcst.icao.isin(AP_IDX)].copy()
            s_iss = np.ceil((to_min(fc.issue_time) - self.t0) / SLOT).astype(int)
            lead = np.round((to_min(fc.valid_time) - (self.t0 + s_iss * SLOT)) / 60).astype(int)
            ok = (s_iss >= 0) & (s_iss < n_slots) & (lead >= 0) & (lead <= self.fc_lead)
            arr = np.full((n_slots, n_ap, self.fc_lead + 1), np.nan, dtype=np.float32)
            ap = fc.icao.map(AP_IDX).to_numpy()
            arr[s_iss[ok], ap[ok], lead[ok]] = fc.severity.to_numpy()[ok]
            have = ~np.isnan(arr).all(axis=2)
            issue = np.where(have, np.arange(n_slots)[:, None], -1)
            self.fc_issue = np.maximum.accumulate(issue, axis=0)
            self.fc = arr

    # --- dynamic features ------------------------------------------------------------------------
    def slot_of(self, t: np.ndarray) -> np.ndarray:
        return np.clip(((t - self.t0) // SLOT).astype(int), 0, self.n_slots - 1)

    def known_delay(self, j: np.ndarray, t: np.ndarray):
        """Latest delay information about flight j at time t.

        Returns (state, delay) with state 0 = nothing known, 1 = overdue (not departed, late),
        2 = departed, 3 = arrived. ``delay`` is the departure delay for 1/2 (lower bound for 1)
        and the arrival delay for 3.
        """
        ok = j >= 0
        jj = np.where(ok, j, 0)
        sd, ad, aa, sa = self.sd[jj], self.ad[jj], self.aa[jj], self.sa[jj]
        arrived = ok & (aa <= t)
        departed = ok & (ad <= t) & ~arrived
        overdue = ok & ~(ad <= t) & (t > sd)
        state = np.select([arrived, departed, overdue], [3, 2, 1], 0)
        delay = np.select([arrived, departed, overdue], [aa - sa, ad - sd, t - sd], np.nan)
        return state, delay

    def features(self, idx: np.ndarray, t: np.ndarray) -> np.ndarray:
        idx = np.asarray(idx, dtype=int)
        t = np.asarray(t, dtype=float)
        if t.ndim == 0:
            t = np.full(len(idx), float(t))
        cols: dict[str, np.ndarray] = {}
        s = self.slot_of(t)
        da, aa_ = self.dep_ap[idx], self.arr_ap[idx]

        # static schedule facts
        cols["dep_ap"] = da.astype(float)
        cols["arr_ap"] = aa_.astype(float)
        cols["ac_type"] = self.type_code[idx].astype(float)
        cols["hub_dep"] = self.hub_dep[idx]
        cols["hub_arr"] = self.hub_arr[idx]
        cols["longhaul"] = self.longhaul[idx]
        cols["block"] = self.block[idx]
        cols["hour"] = self.hour[idx]
        cols["dow"] = self.dow[idx]
        # No month/season feature: with a chronological split the test months never occur in
        # training, so the models could only extrapolate it blindly.
        cols["dep_load"] = self.dep_load[idx]
        cols["arr_load"] = self.arr_load[idx]
        cols["lead"] = self.sd[idx] - t

        # own status (non-zero only for context nodes in the GNN; targets are never departed)
        st, dl = self.known_delay(idx, t)
        cols["own_state"] = st.astype(float)
        cols["own_delay"] = dl

        # rotation: previous leg and the first leg up the chain with known delay
        p = self.prev[idx]
        cols["has_prev"] = (p >= 0).astype(float)
        cols["turn_slack"] = self.buffer[idx] - self.min_turn[idx]
        pst, pdl = self.known_delay(p, t)
        cols["prev_state"] = pst.astype(float)
        cols["prev_delay"] = pdl
        cols["prev_lead"] = np.where(p >= 0, self.sd[np.maximum(p, 0)] - t, np.nan)
        depth = np.zeros(len(idx))
        slack_sum = np.where(p >= 0, cols["turn_slack"], np.nan)
        chain_delay = np.full(len(idx), np.nan)
        cur = p.copy()
        found = np.zeros(len(idx), bool)
        for k in range(1, 7):
            stc, dlc = self.known_delay(cur, t)
            hit = ~found & (stc > 0)
            chain_delay[hit] = dlc[hit]
            depth[hit] = k
            found |= hit
            nxt = np.where(cur >= 0, self.prev[np.maximum(cur, 0)], -1)
            more = ~found & (cur >= 0) & (nxt >= 0)
            slack_sum = np.where(more, slack_sum + self.buffer[np.maximum(cur, 0)]
                                 - self.min_turn[np.maximum(cur, 0)], slack_sum)  # fmt: skip
            cur = np.where(found, -1, nxt)
        cols["chain_depth"] = np.where(found, depth, np.nan)
        cols["chain_delay"] = chain_delay
        cols["chain_slack"] = np.where(found, slack_sum, np.nan)
        cols["proj_reactionary"] = np.maximum(0, chain_delay - np.nan_to_num(slack_sum))

        # connections into hub departures
        cnt = self.feed_ptr[idx + 1] - self.feed_ptr[idx]
        cols["n_feeders"] = cnt.astype(float)
        f_mean = np.full(len(idx), np.nan)
        f_max = np.full(len(idx), np.nan)
        f_late = np.full(len(idx), np.nan)
        f_risk = np.full(len(idx), np.nan)
        has = cnt > 0
        if has.any():
            rows = np.repeat(np.where(has)[0], cnt[has])
            starts = self.feed_ptr[idx[has]]
            offs = np.arange(len(rows)) - np.repeat(np.cumsum(cnt[has]) - cnt[has], cnt[has])
            fi = self.feed_in[np.repeat(starts, cnt[has]) + offs]
            fsl = self.feed_slack[np.repeat(starts, cnt[has]) + offs]
            fst, fdl = self.known_delay(fi, t[rows])
            known = fst > 0
            v = np.where(known, fdl, 0.0)
            seg = np.concatenate([[0], np.cumsum(cnt[has])[:-1]])
            n_known = np.add.reduceat(known.astype(float), seg)
            sums = np.add.reduceat(v, seg)
            mx = np.maximum.reduceat(np.where(known, fdl, -np.inf), seg)
            late = np.add.reduceat((known & (fdl > 15)).astype(float), seg)
            # delay that would break the connection: known delay - (slack - MCT)
            risk = np.maximum.reduceat(np.where(known, fdl - (fsl - config.MCT_MIN), -np.inf), seg)
            with np.errstate(invalid="ignore"):
                f_mean[has] = np.where(n_known > 0, sums / np.maximum(n_known, 1), np.nan)
                f_late[has] = np.where(n_known > 0, late / np.maximum(n_known, 1), np.nan)
            f_max[has] = np.where(np.isfinite(mx), mx, np.nan)
            f_risk[has] = np.where(np.isfinite(risk), risk, np.nan)
        cols["feed_mean"] = f_mean
        cols["feed_max"] = f_max
        cols["feed_frac_late"] = f_late
        cols["feed_break"] = f_risk

        # airport state over the last hour
        cols["dep_ap_recent_dep"] = self.g_dep_mean[s, da]
        cols["dep_ap_recent_n"] = self.g_dep_cnt[s, da]
        cols["arr_ap_recent_arr"] = self.g_arr_mean[s, aa_]
        hub = AP_IDX[config.HUB]
        cols["hub_recent_dep"] = self.g_dep_mean[s, hub]
        cols["hub_recent_arr"] = self.g_arr_mean[s, hub]

        # weather now
        for k, nm in enumerate(WX_GRID_FIELDS):
            cols[f"wx_dep_{nm}"] = self.wx[s, da, k]
        for nm in ["severity", "vis_m", "ts", "sn"]:
            cols[f"wx_arr_{nm}"] = self.wx[s, aa_, WX_GRID_FIELDS.index(nm)]
        # weather forecast for the scheduled hours
        if self.fc is not None:
            cols["fc_dep_sev"] = self._forecast(s, da, self.sd[idx])
            cols["fc_arr_sev"] = self._forecast(s, aa_, self.sa[idx])
        else:
            cols["fc_dep_sev"] = np.full(len(idx), np.nan)
            cols["fc_arr_sev"] = np.full(len(idx), np.nan)

        self.names = list(cols)
        return np.column_stack(list(cols.values())).astype(np.float32)

    def _forecast(self, s: np.ndarray, ap: np.ndarray, valid: np.ndarray) -> np.ndarray:
        si = self.fc_issue[s, ap]
        lead = np.round((valid - (self.t0 + si * SLOT)) / 60)
        ok = (si >= 0) & (lead >= 0) & (lead <= self.fc_lead)
        out = np.full(len(s), np.nan, dtype=np.float32)
        out[ok] = self.fc[si[ok], ap[ok], lead[ok].astype(int)]
        return out

    def feature_names(self) -> list[str]:
        if not self.names:
            self.features(np.array([0]), np.array([self.sd[0] - 60]))
        return self.names

    def targets(self, idx: np.ndarray) -> np.ndarray:
        return np.column_stack([self.ad[idx] - self.sd[idx], self.aa[idx] - self.sa[idx]])


CATEGORICAL = ["dep_ap", "arr_ap", "ac_type"]
WEATHER_COLS_PREFIX = ("wx_", "fc_")


def horizon_rows(store: FeatureStore, horizons=config.HORIZONS) -> pd.DataFrame:
    """(flight, horizon) evaluation rows: prediction time t = floor_30(sched_dep - h)."""
    ok = ~np.isnan(store.ad) & ~np.isnan(store.aa)
    idx = np.where(ok)[0]
    out = []
    for h in horizons:
        t = np.floor((store.sd[idx] - 60 * h) / SLOT) * SLOT
        out.append(pd.DataFrame({"idx": idx, "h": h, "t": t}))
    return pd.concat(out, ignore_index=True)


MAX_LEAD = 390  # minutes ahead (6.5 h) for which a flight is a prediction target
MAX_OVERDUE = 180  # keep predicting a flight up to 3 h after its scheduled departure if not gone


def snapshot_rows(store: FeatureStore, mask: np.ndarray, stride: int = SLOT) -> pd.DataFrame:
    """Every (flight, t) with t on the 30-min grid where the flight is still a prediction target:
    not departed yet and scheduled within [t - MAX_OVERDUE, t + MAX_LEAD]."""
    idx = np.where(mask & ~np.isnan(store.ad) & ~np.isnan(store.aa))[0]
    sd, ad = store.sd[idx], store.ad[idx]
    first = np.ceil((sd - MAX_LEAD) / stride) * stride
    last = np.minimum(
        np.ceil(ad / stride) * stride - stride, np.floor((sd + MAX_OVERDUE) / stride) * stride
    )
    n = np.maximum(((last - first) // stride + 1).astype(int), 0)
    rep = np.repeat(np.arange(len(idx)), n)
    k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    return pd.DataFrame({"idx": idx[rep], "t": first[rep] + stride * k})
