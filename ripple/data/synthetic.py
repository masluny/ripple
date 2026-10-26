"""Synthetic LOT-like hub network with a known delay-generating process.

Why: real scheduled times need an API key (docs/data_sources.md), and even with them we never observe
*why* a flight was late. The simulator gives (a) a dataset to build and test the whole pipeline today
and (b) ground-truth causes, so the explainability part can be scored instead of eyeballed.

Everything here is invented but shaped after public facts: WAW bank structure, LOT fleet mix
(787 / 737 MAX / E-jets / Q400), real airports and great-circle block times. Results on this data
say whether the models can recover a propagation mechanism. They say nothing about LOT's actual
punctuality.

Delay process per leg (minutes), in scheduled-departure order:
    reactionary = max(0, inbound actual arrival + min turn - scheduled departure)
    primary     = random ground/technical delay (more at congested airports, long-haul)
    weather_dep = departure-airport weather severity x de-icing / ATC slowdown
    atfm        = arrival-airport weather + regional ATC capacity days (ground delay at origin)
    congestion  = excess scheduled movements in the 30-min slot, capacity shrinks with weather
    hold        = WAW departure held for a late heavy feeder (connection protection)
    dep_delay   = max(reactionary, primary) + weather_dep + atfm + congestion, then hold
    arr_delay   = dep_delay + (actual block - scheduled block) + arrival holding
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ripple import config
from ripple.data.schema import severity

WAW_TZ = ZoneInfo("Europe/Warsaw")

TYPES = {
    # type: aircraft count, cruise km/h, min turn at outstation / hub (min), destinations
    "B789": dict(n=0, speed=870, turn_out=120, turn_hub=110),
    "B38M": dict(n=18, speed=800, turn_out=45, turn_hub=50),
    "E195": dict(n=26, speed=760, turn_out=35, turn_hub=40),
    "E175": dict(n=12, speed=740, turn_out=30, turn_hub=35),
    "DH8D": dict(n=8, speed=520, turn_out=25, turn_hub=30),
}
DESTS = {
    "B38M": [
        "EGLL",
        "LGAV",
        "LEBL",
        "LEMD",
        "LPPT",
        "LLBG",
        "UGTB",
        "UDYZ",
        "LCLK",
        "EIDW",
        "EGCC",
        "LFMN",
        "LIRF",
        "LFPG",
        "EHAM",
        "EDDF",
    ],
    "E195": [
        "LFPG",
        "EDDF",
        "EDDM",
        "EHAM",
        "EBBR",
        "LSZH",
        "LSGG",
        "LOWW",
        "EKCH",
        "ESSA",
        "ENGM",
        "EFHK",
        "LIMC",
        "LIRF",
        "LHBP",
        "LKPR",
        "LROP",
        "LBSF",
        "LYBE",
        "EDDB",
        "EDDL",
        "EDDH",
        "EGLL",
    ],
    "E175": [
        "EYVI",
        "EVRA",
        "EETN",
        "LKPR",
        "EDDB",
        "EDDH",
        "EDDL",
        "LUKK",
        "LHBP",
        "LOWW",
        "EPGD",
        "EPKK",
        "EPWR",
    ],
    "DH8D": ["EPKK", "EPGD", "EPWR", "EPKT", "EPPO", "EPSC", "EPRZ", "EPBY", "EPSY"],
}
# Long-haul: (destination, WAW local departure) - one daily frequency each.
LONGHAUL = [
    ("KJFK", "13:00"), ("KEWR", "16:10"), ("KORD", "12:50"), ("CYYZ", "13:30"), ("KMIA", "12:15"),
    ("KLAX", "15:20"), ("RKSI", "15:45"), ("RJTT", "16:00"), ("VIDP", "21:45"), ("UTTT", "21:20"),
]  # fmt: skip
# WAW departure banks (local off-block), aligned with real LOT take-off peaks in OpenSky Nov 2022
WAVES = ["07:15", "10:10", "13:05", "15:05", "16:50", "19:50", "22:40"]

REGIONS = {
    "PL": "EPWA EPKK EPGD EPWR EPKT EPPO EPSC EPRZ EPBY EPSY",
    "CE": "EDDB EDDM LKPR LOWW LHBP EDDF",
    "WE": "EGLL LFPG EHAM EBBR EDDL EGCC EIDW LSZH LSGG LFMN",
    "NE": "EKCH ESSA ENGM EFHK EYVI EVRA EETN EDDH",
    "SE": "LGAV LIRF LIMC LEBL LEMD LPPT LROP LBSF LYBE LUKK LCLK LLBG UGTB UDYZ",
    "US": "KJFK KEWR KORD CYYZ KMIA KLAX",
    "AS": "RKSI RJTT VIDP UTTT",
}
REGION_OF = {ap: r for r, aps in REGIONS.items() for ap in aps.split()}
CONGESTED = {"EGLL", "LFPG", "EDDF", "EHAM", "KJFK", "KEWR", "EDDM"}

# Monthly event rates per region per day (fog, snow, thunderstorm, wind storm). Index 0 = January.
_FOG = np.array([0.18, 0.15, 0.10, 0.05, 0.03, 0.02, 0.02, 0.04, 0.08, 0.16, 0.22, 0.22])
_SNOW = np.array([0.30, 0.28, 0.15, 0.03, 0, 0, 0, 0, 0, 0.02, 0.10, 0.25])
_TS = np.array([0, 0, 0.02, 0.06, 0.20, 0.35, 0.40, 0.33, 0.12, 0.03, 0, 0])
_WIND = np.array([0.15, 0.13, 0.12, 0.06, 0.03, 0.02, 0.02, 0.03, 0.06, 0.12, 0.14, 0.16])
REGION_SCALE = {  # multipliers for (fog, snow, ts, wind)
    "PL": (1.0, 1.0, 1.0, 0.8), "CE": (1.0, 0.7, 1.0, 0.8), "WE": (0.9, 0.25, 0.6, 1.4),
    "NE": (0.8, 1.2, 0.5, 1.2), "SE": (0.4, 0.1, 0.9, 0.6), "US": (0.5, 0.6, 1.1, 0.8),
    "AS": (0.7, 0.3, 0.8, 0.5),
}  # fmt: skip


@dataclass
class SimConfig:
    start: str = "2025-01-01"
    days: int = 365
    seed: int = 7
    p_primary: float = 0.25
    p_atc_day: float = 0.025


def _hav_km(a: str, b: str) -> float:
    p, q = config.AIRPORTS[a], config.AIRPORTS[b]
    la1, la2 = math.radians(p.lat), math.radians(q.lat)
    dla, dlo = la2 - la1, math.radians(q.lon - p.lon)
    h = math.sin(dla / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlo / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))


def nominal_block(a: str, b: str, ac_type: str) -> float:
    """Unpadded gate-to-gate minutes; westbound legs slower (jet stream)."""
    d = _hav_km(a, b)
    west = config.AIRPORTS[b].lon < config.AIRPORTS[a].lon
    wind = 1.06 if west else 0.96
    return (22 + d / TYPES[ac_type]["speed"] * 60 * wind) if d > 0 else 0.0


def sched_block(a: str, b: str, ac_type: str) -> int:
    nb = nominal_block(a, b, ac_type)
    return int(
        5 * math.ceil(nb * 1.02 / 5)
    )  # ~2% schedule padding (plus rounding up to 5 min), rounded up to 5 min


def _hm(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


# ---------------------------------------------------------------------------------------------
# Schedule template
# ---------------------------------------------------------------------------------------------
def build_lines(rng: np.random.Generator) -> list[dict]:
    """Daily 'lines' of flying in WAW-local minutes after midnight. One line = one tail."""
    lines: list[dict] = []
    tails = iter(
        f"SP-L{a}{b}" for a in "ABCDEFGHIJKLMNOPRSTUVWZ" for b in "ABCDEFGHIJKLMNOPRSTUVWZ"
    )
    waves = [_hm(w) for w in WAVES]

    for dest, dep in LONGHAUL:
        out_b, back_b = sched_block("EPWA", dest, "B789"), sched_block(dest, "EPWA", "B789")
        cycle = out_b + TYPES["B789"]["turn_out"] + 30 + back_b + TYPES["B789"]["turn_hub"] + 30
        n_ac = math.ceil(cycle / 1440)
        for k in range(n_ac):
            legs = [("EPWA", dest, _hm(dep))]
            ret = _hm(dep) + out_b + TYPES["B789"]["turn_out"] + 30
            legs.append((dest, "EPWA", ret))
            lines.append(
                dict(
                    tail=next(tails),
                    ac_type="B789",
                    legs=legs,
                    every=n_ac,
                    phase=k,
                    nightstop=False,
                )
            )

    for ac_type, spec in TYPES.items():
        if ac_type == "B789":
            continue
        for _ in range(spec["n"]):
            legs: list[tuple[str, str, int]] = []
            nightstop = rng.random() < 0.3
            t = waves[0] + int(rng.integers(-4, 9)) * 5
            if nightstop:  # first leg of the day comes into WAW from the outstation
                x = str(rng.choice(DESTS[ac_type]))
                b = sched_block(x, "EPWA", ac_type)
                legs.append((x, "EPWA", waves[0] - 40 - b))
                t = waves[1] + int(rng.integers(0, 12)) * 5
            while True:
                x = str(rng.choice(DESTS[ac_type]))
                out_b, back_b = sched_block("EPWA", x, ac_type), sched_block(x, "EPWA", ac_type)
                slack_out = int(rng.integers(0, 4)) * 5
                ret = t + out_b + spec["turn_out"] + slack_out
                arr_back = ret + back_b
                if arr_back > _hm("23:50"):
                    if nightstop and t <= _hm("23:20"):
                        legs.append(("EPWA", x, t))  # stays overnight at x, flies x->WAW at dawn
                        legs[0] = (x, "EPWA", waves[0] - 40 - sched_block(x, "EPWA", ac_type))
                    break
                legs += [("EPWA", x, t), (x, "EPWA", ret)]
                nxt = [w for w in waves if w >= arr_back + spec["turn_hub"] + 5]
                if not nxt:
                    break
                t = nxt[0] + int(rng.integers(0, 12)) * 5
            if nightstop and legs[-1][1] == "EPWA":  # never got to overnight away: drop dawn leg
                nightstop = False
                legs = legs[1:]
            if len(legs) >= 2:
                lines.append(
                    dict(tail=next(tails), ac_type=ac_type, legs=legs, every=1, phase=0,
                         nightstop=nightstop)
                )  # fmt: skip
    return lines


def expand_schedule(lines: list[dict], cfg: SimConfig, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    numbers: dict[tuple[str, str, int], str] = {}
    start = pd.Timestamp(cfg.start)
    for d in range(cfg.days):
        day = start + pd.Timedelta(days=d)
        base = pd.Timestamp(day.date(), tz=WAW_TZ).tz_convert("UTC")
        saturday = day.day_of_week == 5
        for li, line in enumerate(lines):
            if (d - line["phase"]) % line["every"]:
                continue
            if saturday and line["ac_type"] in ("E195", "E175", "DH8D") and li % 6 == 0:
                continue  # thinner Saturday programme
            for a, b, t in line["legs"]:
                key = (a, b, t)
                if key not in numbers:
                    numbers[key] = f"LO{100 + len(numbers):03d}"
                dep = base + pd.Timedelta(minutes=t)
                rows.append(
                    dict(flight_no=numbers[key], tail=line["tail"], ac_type=line["ac_type"],
                         dep_airport=a, arr_airport=b, sched_dep=dep,
                         sched_arr=dep + pd.Timedelta(minutes=sched_block(a, b, line["ac_type"])))
                )  # fmt: skip
    df = pd.DataFrame(rows).sort_values("sched_dep").reset_index(drop=True)
    df.insert(0, "flight_id", np.arange(len(df)))
    return df


# ---------------------------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------------------------
def simulate_weather(cfg: SimConfig, rng: np.random.Generator):
    """Hourly METAR-like observations plus 6-hourly TAF-like forecasts for every airport."""
    airports = list(config.AIRPORTS)
    hours = pd.date_range(cfg.start, periods=cfg.days * 24 + 48, freq="h", tz="UTC")
    n_h = len(hours)
    hod = hours.hour.values
    events: list[tuple[str, str, int, int]] = []  # (airport, kind, start_hour_idx, duration)

    for region, aps in REGIONS.items():
        aps = aps.split()
        sc = REGION_SCALE[region]
        for d in range(cfg.days + 1):
            m = (pd.Timestamp(cfg.start) + pd.Timedelta(days=d)).month - 1
            for kind, rate, scale, (h0, h1), (dur0, dur1) in (
                ("fg", _FOG[m], sc[0], (2, 8), (2, 7)),
                ("sn", _SNOW[m], sc[1], (0, 23), (3, 12)),
                ("ts", _TS[m], sc[2], (12, 19), (1, 4)),
                ("wind", _WIND[m], sc[3], (0, 23), (6, 18)),
            ):
                if rng.random() < rate * scale:
                    start = d * 24 + int(rng.integers(h0, h1 + 1))
                    dur = int(rng.integers(dur0, dur1 + 1))
                    share = 0.8 if kind in ("wind", "sn") else 0.45
                    for ap in aps:
                        if rng.random() < share:
                            off = int(rng.integers(-2, 3))
                            events.append((ap, kind, start + off, dur + int(rng.integers(-1, 2))))
    ev = pd.DataFrame(events, columns=["icao", "kind", "h", "dur"])

    obs = {}
    for ap in airports:
        lat = config.AIRPORTS[ap].lat
        seasonal = -np.cos(2 * np.pi * (hours.day_of_year.values - 20) / 365)
        temp = 30 - 0.45 * lat + 10 * seasonal + 4 * np.sin(2 * np.pi * (hod - 9) / 24)
        temp += rng.normal(0, 2.5, n_h)
        f = dict(
            vis_m=np.full(n_h, 9999.0),
            ceil_ft=rng.choice([1500, 2500, 3500, 5000], n_h, p=[0.15, 0.2, 0.25, 0.4]).astype(float),
            wind_kt=np.clip(8 + 4 * np.sin(2 * np.pi * (hod - 14) / 24) + rng.normal(0, 3, n_h), 0, None),
            temp_c=temp,
            ts=np.zeros(n_h), sn=np.zeros(n_h), fg=np.zeros(n_h),
            ra=(rng.random(n_h) < 0.08).astype(float),
        )  # fmt: skip
        for e in ev[ev.icao == ap].itertuples():
            sl = slice(max(e.h, 0), min(e.h + max(e.dur, 1), n_h))
            if e.kind == "fg":
                f["vis_m"][sl] = rng.choice([150, 300, 500, 800, 1200])
                f["ceil_ft"][sl] = rng.choice([100, 200, 300])
                f["fg"][sl] = 1
            elif e.kind == "sn" and temp[sl].mean() < 4:
                f["vis_m"][sl] = np.minimum(f["vis_m"][sl], rng.choice([600, 1200, 2500, 4000]))
                f["ceil_ft"][sl] = np.minimum(f["ceil_ft"][sl], rng.choice([300, 600, 1200]))
                f["sn"][sl] = 1
            elif e.kind == "ts":
                f["ts"][sl], f["ra"][sl] = 1, 1
                f["vis_m"][sl] = np.minimum(f["vis_m"][sl], rng.choice([2000, 4000, 7000]))
                f["wind_kt"][sl] = np.maximum(f["wind_kt"][sl], 18)
            elif e.kind == "wind":
                f["wind_kt"][sl] = np.maximum(f["wind_kt"][sl], rng.integers(24, 38))
                f["ra"][sl] = 1
        f["gust_kt"] = np.where(f["wind_kt"] > 15, f["wind_kt"] * 1.5, f["wind_kt"])
        obs[ap] = pd.DataFrame(f).assign(icao=ap, time=hours)
    wx = pd.concat(obs.values(), ignore_index=True)
    wx["ceil_ft"] = wx["ceil_ft"].astype(float)
    wx["severity"] = severity(wx)
    fc = _forecasts(wx, rng)
    return wx, fc


def _forecasts(wx: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """TAF-like severity forecasts issued every 6 h for the next 30 h.

    Forecast = truth smoothed in time (timing error) + noise that grows with lead time, plus
    occasional missed events and false alarms.
    """
    out = []
    for ap, g in wx.groupby("icao", sort=False):
        sev = g["severity"].to_numpy()
        times = g["time"].to_numpy()
        n = len(sev)
        smooth = np.convolve(sev, np.ones(3) / 3, mode="same")
        for i0 in range(0, n - 30, 6):
            lead = np.arange(1, 31)
            truth = smooth[i0 + lead]
            miss = rng.random(30) < 0.1 + lead / 120
            fa = (rng.random(30) < 0.02) * rng.choice([0.4, 0.7, 0.9], 30)
            f = np.where(miss, 0.0, truth) + fa + rng.normal(0, 0.05 + lead / 150, 30)
            out.append(
                pd.DataFrame(dict(icao=ap, issue_time=times[i0], valid_time=times[i0 + lead],
                                  severity=np.clip(f, 0, None)))
            )  # fmt: skip
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------------------------
# Delay process
# ---------------------------------------------------------------------------------------------
def simulate_delays(
    flights: pd.DataFrame, wx: pd.DataFrame, cfg: SimConfig, rng: np.random.Generator
) -> pd.DataFrame:
    df = flights.copy()
    t0 = pd.Timestamp(cfg.start, tz="UTC")
    ap_idx = {a: i for i, a in enumerate(config.AIRPORTS)}
    sev = wx.pivot(index="time", columns="icao", values="severity")[
        list(config.AIRPORTS)
    ].to_numpy()
    snow = wx.pivot(index="time", columns="icao", values="sn")[list(config.AIRPORTS)].to_numpy()
    n_h = sev.shape[0]

    def hour_of(ts_min: float) -> int:
        return int(min(max(ts_min // 60, 0), n_h - 1))

    sd = ((df.sched_dep - t0).dt.total_seconds() / 60).to_numpy()
    sa = ((df.sched_arr - t0).dt.total_seconds() / 60).to_numpy()
    dep_ap = df.dep_airport.map(ap_idx).to_numpy()
    arr_ap = df.arr_airport.map(ap_idx).to_numpy()
    types = df.ac_type.to_numpy()
    dep_codes, arr_codes = df.dep_airport.to_numpy(), df.arr_airport.to_numpy()
    nom_blk = np.array(
        [nominal_block(a, b, t) for a, b, t in zip(dep_codes, arr_codes, types, strict=True)]
    )
    n = len(df)

    # Previous leg of the same tail (rotation) - only if it ends where this one starts.
    df["prev_id"] = df.groupby("tail")["flight_id"].shift(1)
    prev_arr_ap = df.groupby("tail")["arr_airport"].shift(1)
    df.loc[prev_arr_ap != df.dep_airport, "prev_id"] = np.nan
    prev = df["prev_id"].fillna(-1).astype(int).to_numpy()

    # Scheduled movements per 30-min slot and airport (known from the schedule).
    slot_dep = pd.Series(1, index=pd.MultiIndex.from_arrays([dep_ap, (sd // 30).astype(int)]))
    dep_load = slot_dep.groupby(level=[0, 1]).transform("sum").to_numpy()
    slot_arr = pd.Series(1, index=pd.MultiIndex.from_arrays([arr_ap, (sa // 30).astype(int)]))
    arr_load = slot_arr.groupby(level=[0, 1]).transform("sum").to_numpy()
    hub = ap_idx[config.HUB]
    cap = np.where(dep_ap == hub, 14, 4)
    cap_arr = np.where(arr_ap == hub, 14, 4)

    # Regional ATC capacity days.
    days = cfg.days + 2
    atc = {r: rng.random(days) < cfg.p_atc_day * (2.0 if r == "WE" else 1.0) for r in REGIONS}
    atc_win = {r: (rng.integers(5, 12, days) * 60, rng.integers(4, 10, days) * 60) for r in REGIONS}
    region_idx = [REGION_OF[a] for a in config.AIRPORTS]

    # Feeders for WAW departures: arrivals 35-120 min before. "Heavy" pairs get protected.
    is_hub_dep = dep_ap == hub
    arr_hub_ids = np.where(arr_ap == hub)[0]
    arr_hub_sa = sa[arr_hub_ids]
    order = np.argsort(arr_hub_sa)
    arr_hub_ids, arr_hub_sa = arr_hub_ids[order], arr_hub_sa[order]
    longhaul = np.array([config.AIRPORTS[a].longhaul for a in config.AIRPORTS])
    is_lh = longhaul[dep_ap] | longhaul[arr_ap]

    actual_dep = np.zeros(n)
    actual_arr = np.zeros(n)
    comps = np.zeros((n, 6))  # reactionary, primary, weather, atfm, congestion, hold
    sched_blk = sa - sd

    for i in range(n):
        typ = TYPES[types[i]]
        # 1. reactionary
        react = 0.0
        if prev[i] >= 0:
            turn = typ["turn_hub"] if dep_ap[i] == hub else typ["turn_out"]
            ready = actual_arr[prev[i]] + turn * rng.uniform(0.95, 1.12)
            react = max(0.0, ready - sd[i])
        # 2. primary
        p = cfg.p_primary + 0.08 * (dep_codes[i] in CONGESTED)
        p += 0.10 * is_lh[i]
        primary = 0.0
        if rng.random() < p:
            primary = rng.lognormal(2.3, 0.75)
        if rng.random() < 0.012:
            primary += 20 + rng.exponential(70)
        # 3. weather at departure airport
        hd = hour_of(sd[i])
        s_dep = sev[hd, dep_ap[i]]
        weather = s_dep * rng.uniform(8, 30) + 12 * snow[hd, dep_ap[i]]
        # 4. ATFM: arrival weather + ATC capacity days
        ha = hour_of(sa[i])
        s_arr = sev[ha, arr_ap[i]]
        atfm = s_arr * rng.uniform(10, 35) if rng.random() < min(0.9, 0.6 * s_arr) else 0.0
        day = int(sd[i] // 1440)
        for r in {region_idx[dep_ap[i]], region_idx[arr_ap[i]]}:
            if atc[r][day]:
                w0, wl = atc_win[r][0][day], atc_win[r][1][day]
                mod = sd[i] % 1440
                if w0 <= mod <= w0 + wl:
                    atfm += rng.uniform(15, 55)
        # 5. congestion (capacity shrinks with weather)
        eff_cap = cap[i] * (1 - 0.35 * min(s_dep, 1.5))
        congestion = 1.5 * max(0.0, dep_load[i] - eff_cap)
        base = max(react, primary)
        dep_delay = base + weather + atfm + congestion + rng.normal(-1.5, 2.0)
        # 6. connection hold at WAW
        hold = 0.0
        if is_hub_dep[i]:
            lo = np.searchsorted(arr_hub_sa, sd[i] - 120)
            hi = np.searchsorted(arr_hub_sa, sd[i] - 35)
            for j in arr_hub_ids[lo:hi]:
                heavy = is_lh[i] or is_lh[j] or (i * 31 + j * 17) % 5 == 0
                if not heavy or j >= i:
                    continue
                need = actual_arr[j] + 30 - (sd[i] + dep_delay)
                if 0 < need <= 20 and rng.random() < 0.6:
                    hold = max(hold, need)
            dep_delay += hold
        dep_delay = max(dep_delay, -8.0)
        actual_dep[i] = sd[i] + dep_delay

        # Arrival: nominal block with noise, airborne holding at destination
        blk = nom_blk[i] * (1 + rng.normal(0, 0.025)) + rng.normal(0, 2.5)
        ha2 = hour_of(actual_dep[i] + blk)
        s_arr2 = sev[ha2, arr_ap[i]]
        blk += s_arr2 * rng.uniform(4, 18)
        blk += 1.0 * max(0.0, arr_load[i] - cap_arr[i] * (1 - 0.35 * min(s_arr2, 1.5)))
        actual_arr[i] = actual_dep[i] + max(blk, 0.6 * sched_blk[i])

        react_part = react if react >= primary else 0.0
        prim_part = primary if primary > react else 0.0
        comps[i] = (react_part, prim_part, weather, atfm, congestion, hold)

    df["actual_dep"] = t0 + pd.to_timedelta(actual_dep, unit="min")
    df["actual_arr"] = t0 + pd.to_timedelta(actual_arr, unit="min")
    names = ["reactionary", "primary", "weather", "atfm", "congestion", "hold"]
    for k, nm in enumerate(names):
        df[f"cause_{nm}"] = comps[:, k]
    tot = comps.sum(1)
    df["true_cause"] = np.where(tot < 5, "none", np.array(names)[comps.argmax(1)])
    return df.drop(columns=["prev_id"])


def generate(cfg: SimConfig | None = None):
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed)
    lines = build_lines(rng)
    flights = expand_schedule(lines, cfg, rng)
    wx, fc = simulate_weather(cfg, rng)
    flights = simulate_delays(flights, wx, cfg, rng)
    return flights, wx, fc
