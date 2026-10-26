"""Real-data adapters -> canonical tables (see schema.py).

* AeroDataBox raw JSON (data/raw/aerodatabox)  -> flights with scheduled + actual times
* OpenSky /flights parquet (data/raw/opensky_flights) -> fills missing actual times (take-off/landing)
* OpenSky anonymous states (data/raw/opensky_states) -> WAW take-off/landing events (EDA, enrichment)
* EUROCONTROL R&D archive CSV (data/raw/eurocontrol) -> historical flights (column names follow the
  published metadata; not yet verified against a downloaded file)
* METAR parquet (data/raw/metar), TAF JSONL (data/raw/taf) -> weather_obs / weather_fcst
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ripple import config
from ripple.data.metar import decode_frame
from ripple.data.schema import FLIGHT_COLS, severity

MODEL_TO_TYPE = [
    (r"787-9", "B789"), (r"787-8", "B788"), (r"737 MAX 8|737-8 MAX|737 8", "B38M"),
    (r"737-800", "B738"), (r"Embraer 195|E195|ERJ-195", "E195"), (r"Embraer 190|E190", "E190"),
    (r"Embraer 175|E175", "E175"), (r"Embraer 170|E170", "E170"),
    (r"Dash 8|DHC-8-4|Q400", "DH8D"), (r"A220-300|BCS3", "BCS3"),
]  # fmt: skip


def model_to_type(model: str | None) -> str | None:
    if not model:
        return None
    for pat, code in MODEL_TO_TYPE:
        if re.search(pat, model, re.I):
            return code
    return None


# --- flights --------------------------------------------------------------------------------------
def load_aerodatabox(raw_dir: Path | None = None) -> pd.DataFrame:
    from ripple.collect.schedules import normalise

    raw_dir = raw_dir or config.RAW / "aerodatabox"
    frames = [normalise(json.loads(p.read_text())) for p in sorted(raw_dir.glob("*.json"))]
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("sched_dep").drop_duplicates(["flight_no", "sched_dep"], keep="last")
    for c in ("sched_dep", "sched_arr", "actual_dep", "actual_arr", "runway_dep", "runway_arr"):
        df[c] = pd.to_datetime(df[c], utc=True)
    # gate times when available, otherwise runway times (take-off/landing) - flagged per leg
    df["time_basis"] = np.where(df.actual_dep.notna(), "gate", "runway")
    df["actual_dep"] = df.actual_dep.fillna(df.runway_dep)
    df["actual_arr"] = df.actual_arr.fillna(df.runway_arr)
    df["ac_type"] = df.ac_model.map(model_to_type)
    return df[~df.cancelled]


def enrich_with_opensky(df: pd.DataFrame, raw_dir: Path | None = None) -> pd.DataFrame:
    """Fill missing actual times from OpenSky firstSeen/lastSeen matched on icao24 and time."""
    raw_dir = raw_dir or config.RAW / "opensky_flights"
    files = sorted(raw_dir.glob("*.parquet"))
    if not files or df.empty or "icao24" not in df:
        return df
    osk = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    osk["takeoff"] = pd.to_datetime(osk.firstSeen, unit="s", utc=True)
    osk["landing"] = pd.to_datetime(osk.lastSeen, unit="s", utc=True)
    osk = osk.dropna(subset=["icao24"]).sort_values("takeoff")
    left = df.dropna(subset=["icao24"]).sort_values("sched_dep")
    m = pd.merge_asof(
        left, osk[["icao24", "takeoff", "landing"]], left_on="sched_dep", right_on="takeoff",
        by="icao24", direction="nearest", tolerance=pd.Timedelta("4h"),
    )  # fmt: skip
    m = m.set_index(left.index)
    df = df.copy()
    df.loc[m.index, "actual_dep"] = df.loc[m.index, "actual_dep"].fillna(m.takeoff)
    df.loc[m.index, "actual_arr"] = df.loc[m.index, "actual_arr"].fillna(m.landing)
    return df


def load_eurocontrol(raw_dir: Path | None = None) -> pd.DataFrame:
    """EUROCONTROL Aviation Data Repository for Research - Flights_*.csv(.gz) files.

    Filed off-block time is the *last filed* EOBT, which can already contain delay updates (DLA
    messages). Delays measured against it are therefore lower bounds - say so when reporting.
    """
    raw_dir = raw_dir or config.RAW / "eurocontrol"
    files = sorted(raw_dir.glob("Flights_*.csv*"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    df = df[df["AC Operator"] == config.AIRLINE_ICAO]
    fmt = "%d-%m-%Y %H:%M:%S"
    out = pd.DataFrame({
        "flight_no": df["ECTRL ID"].astype(str),
        "tail": df["AC Registration"],
        "ac_type": df["AC Type"],
        "dep_airport": df["ADEP"],
        "arr_airport": df["ADES"],
        "sched_dep": pd.to_datetime(df["FILED OFF BLOCK TIME"], format=fmt, utc=True),
        "sched_arr": pd.to_datetime(df["FILED ARRIVAL TIME"], format=fmt, utc=True),
        "actual_dep": pd.to_datetime(df["ACTUAL OFF BLOCK TIME"], format=fmt, utc=True),
        "actual_arr": pd.to_datetime(df["ACTUAL ARRIVAL TIME"], format=fmt, utc=True),
        "source": "eurocontrol",
    })  # fmt: skip
    return out


def to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["tail", "sched_dep", "sched_arr", "dep_airport", "arr_airport"]).copy()
    df = df[df.sched_arr > df.sched_dep]
    df["ac_type"] = df["ac_type"].fillna("UNKN")
    df = df.sort_values("sched_dep").reset_index(drop=True)
    df["flight_id"] = np.arange(len(df))
    extra = [c for c in ("time_basis", "source", "icao24") if c in df]
    return df[FLIGHT_COLS + extra]


# --- OpenSky anonymous states -> WAW movements ---------------------------------------------------
def states_to_movements(raw_dir: Path | None = None, callsign_prefix: str | None = "LOT"):
    """Detect take-offs and landings at WAW from the 4.5-minute anonymous state snapshots.

    Times are estimated from altitude / vertical rate at the first airborne (or last airborne)
    snapshot and bounded by the neighbouring snapshot, so the error is a few minutes at most.
    """
    raw_dir = raw_dir or config.RAW / "opensky_states"
    files = sorted(raw_dir.glob("*.jsonl"))
    if not files:
        return pd.DataFrame()
    st = pd.concat([pd.read_json(p, lines=True) for p in files], ignore_index=True)
    if callsign_prefix:
        st = st[st.callsign.fillna("").str.startswith(callsign_prefix)]
    st = st.sort_values(["icao24", "snapshot_time"])
    events = []
    for icao24, g in st.groupby("icao24"):
        g = g.reset_index(drop=True)
        for i in range(1, len(g)):
            a, b = g.iloc[i - 1], g.iloc[i]
            if b.snapshot_time - a.snapshot_time > 900:
                continue
            if a.on_ground and not b.on_ground:
                alt, vr = b.baro_altitude or 0, b.vertical_rate or 0
                dt = alt / vr if vr and vr > 0 else (b.snapshot_time - a.snapshot_time) / 2
                t = max(a.snapshot_time, b.snapshot_time - dt)
                events.append((icao24, b.callsign, "takeoff", t))
            elif not a.on_ground and b.on_ground:
                alt, vr = a.baro_altitude or 0, a.vertical_rate or 0
                dt = alt / -vr if vr and vr < 0 else (b.snapshot_time - a.snapshot_time) / 2
                t = min(b.snapshot_time, a.snapshot_time + dt)
                events.append((icao24, a.callsign, "landing", t))
    ev = pd.DataFrame(events, columns=["icao24", "callsign", "event", "time"])
    ev["time"] = pd.to_datetime(ev.time, unit="s", utc=True)
    return ev


# --- weather --------------------------------------------------------------------------------------
def load_metar(raw_dir: Path | None = None) -> pd.DataFrame:
    raw_dir = raw_dir or config.RAW / "metar"
    files = sorted(raw_dir.glob("*.parquet"))
    if not files:
        return pd.DataFrame(columns=["icao", "time", "severity"])
    return decode_frame(pd.concat([pd.read_parquet(p) for p in files], ignore_index=True))


def _taf_group_fields(g: dict) -> dict:
    vis = g.get("visib")
    if isinstance(vis, str):
        vis_m = 9999.0 if vis.endswith("+") or not vis else float(vis.rstrip("+")) * 1609
    else:
        vis_m = float(vis) * 1609 if vis is not None else np.nan
    ceil = [c.get("base") for c in g.get("clouds") or [] if c.get("cover") in ("BKN", "OVC", "VV")]
    wx = g.get("wxString") or ""
    return dict(
        vis_m=min(vis_m, 9999.0) if not np.isnan(vis_m) else np.nan,
        ceil_ft=float(min(ceil)) if ceil and min(ceil) is not None else np.nan,
        wind_kt=g.get("wspd"), gust_kt=g.get("wgst"),
        ts=float("TS" in wx), sn=float(any(k in wx for k in ("SN", "SG", "PL", "FZ"))),
        fg=float(any(k in wx for k in ("FG", "BR"))), ra=float(any(k in wx for k in ("RA", "DZ"))),
        temp_c=np.nan,
    )  # fmt: skip


def load_taf(raw_dir: Path | None = None) -> pd.DataFrame:
    """AWC TAF JSON archive -> hourly expected severity per (icao, issue_time, valid_time).

    Base/FM/BECMG groups set the prevailing conditions; TEMPO and PROB groups add their severity
    weighted by probability (TEMPO = 0.5).
    """
    raw_dir = raw_dir or config.RAW / "taf"
    rows = []
    for p in sorted(raw_dir.glob("*.jsonl")):
        for line in p.read_text().splitlines():
            rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame(columns=["icao", "issue_time", "valid_time", "severity"])
    seen, out = set(), []
    for r in rows:
        key = (r["icaoId"], r["issueTime"])
        if key in seen:
            continue
        seen.add(key)
        issue = pd.Timestamp(r["issueTime"]).tz_convert("UTC") if pd.Timestamp(r["issueTime"]).tz \
            else pd.Timestamp(r["issueTime"], tz="UTC")  # fmt: skip
        hours = pd.date_range(
            pd.Timestamp(r["validTimeFrom"], unit="s", tz="UTC"),
            pd.Timestamp(r["validTimeTo"], unit="s", tz="UTC"), freq="h", inclusive="left",
        )  # fmt: skip
        if len(hours) == 0:
            continue
        hv = ((hours - pd.Timestamp(0, tz="UTC")).total_seconds()).to_numpy()
        base = pd.DataFrame([_taf_group_fields({})] * len(hours))
        extra = np.zeros(len(hours))
        for g in r.get("fcsts", []):
            m = (hv >= g["timeFrom"]) & (hv < g["timeTo"])
            fields = _taf_group_fields(g)
            if g.get("fcstChange") in (None, "FM", "BECMG"):
                for k, v in fields.items():
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        base.loc[m, k] = v
            else:
                p_ = (g.get("probability") or 50) / 100
                sev = severity(pd.DataFrame([fields]))[0]
                extra[m] = np.maximum(extra[m], p_ * sev)
        sev = severity(base).to_numpy() + extra
        out.append(pd.DataFrame(dict(icao=r["icaoId"], issue_time=issue, valid_time=hours,
                                     severity=sev)))  # fmt: skip
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# --- OpenSky Zenodo flight lists (2019-2022) ------------------------------------------------------
def load_zenodo_lot(path: Path, out: Path | None = None) -> pd.DataFrame:
    """Filter one monthly ``flightlist_*.csv.gz`` to LOT flights and cache it as parquet.

    These lists have real take-off/landing times, icao24, registration and type, but no schedule,
    so they are used for rotation structure and turnaround analysis, not for delay labels.
    """
    out = out or config.INTERIM / f"zenodo_lot_{path.name.split('_')[1]}.parquet"
    if out.exists():
        return pd.read_parquet(out)
    chunks = []
    for ch in pd.read_csv(path, chunksize=500_000, usecols=range(10)):
        chunks.append(ch[ch.callsign.fillna("").str.startswith(config.AIRLINE_ICAO)])
    df = pd.concat(chunks, ignore_index=True)
    for c in ("firstseen", "lastseen", "day"):
        df[c] = pd.to_datetime(df[c], utc=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return df
