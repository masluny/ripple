"""Canonical tables shared by the real and the synthetic pipelines.

flights (one row per operated leg, all times tz-aware UTC):
    flight_id     int, unique
    flight_no     "LO379"
    tail          registration ("SP-LWA") - the rotation key
    ac_type       ICAO type designator ("E195")
    dep_airport   ICAO
    arr_airport   ICAO
    sched_dep, sched_arr, actual_dep, actual_arr   (actual_* NaT if unknown)

weather_obs (METAR-like, one row per airport per report):
    icao, time, vis_m, ceil_ft, wind_kt, gust_kt, temp_c, ts, sn, fg, ra, severity

weather_fcst (TAF-like): icao, issue_time, valid_time (hourly), same fields as weather_obs.
"""

from __future__ import annotations

import pandas as pd

FLIGHT_COLS = [
    "flight_id", "flight_no", "tail", "ac_type", "dep_airport", "arr_airport",
    "sched_dep", "sched_arr", "actual_dep", "actual_arr",
]  # fmt: skip
WX_FIELDS = ["vis_m", "ceil_ft", "wind_kt", "gust_kt", "temp_c", "ts", "sn", "fg", "ra"]
TIME_COLS = ["sched_dep", "sched_arr", "actual_dep", "actual_arr"]


def validate_flights(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(FLIGHT_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"flights table misses {missing}")
    for c in TIME_COLS:
        if str(df[c].dtype) != "datetime64[ns, UTC]":
            df[c] = pd.to_datetime(df[c], utc=True).astype("datetime64[ns, UTC]")
    if df.flight_id.duplicated().any():
        raise ValueError("flight_id must be unique")
    bad = df.sched_arr <= df.sched_dep
    if bad.any():
        raise ValueError(f"{bad.sum()} legs arrive before they depart")
    return df.sort_values("sched_dep").reset_index(drop=True)


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Delay labels in minutes (positive = late)."""
    df = df.copy()
    df["dep_delay"] = (df.actual_dep - df.sched_dep).dt.total_seconds() / 60
    df["arr_delay"] = (df.actual_arr - df.sched_arr).dt.total_seconds() / 60
    return df


def severity(wx: pd.DataFrame) -> pd.Series:
    """Hand-made operational weather severity in [0, ~3]. Same formula for METARs and the simulator."""
    vis = wx.vis_m.fillna(9999)
    ceil = wx.ceil_ft.fillna(5000)
    gust = wx.gust_kt.fillna(0)
    wind = wx.wind_kt.fillna(0)
    return (
        0.4 * (vis < 1500)
        + 0.4 * (vis < 550)
        + 0.3 * (ceil < 500)
        + 0.3 * (ceil < 200)
        + 0.9 * wx.ts.fillna(0)
        + 0.7 * wx.sn.fillna(0)
        + 0.2 * wx.fg.fillna(0)
        + 0.1 * wx.ra.fillna(0)
        + 0.3 * (gust >= 35)
        + 0.3 * (wind >= 25)
    ).astype(float)
