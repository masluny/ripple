"""Minimal METAR / TAF decoding into the canonical weather fields.

Deliberately small: we only need visibility, ceiling, wind, temperature and a few weather flags.
Trend groups (TEMPO/BECMG/NOSIG) are cut off before decoding the main body and summarised
separately as ``trend_adverse`` because they are a 2-hour forecast that is known at report time.
"""

from __future__ import annotations

import re
from fractions import Fraction

import numpy as np
import pandas as pd

from ripple.data.schema import WX_FIELDS, severity

_WIND = re.compile(r"\b(\d{3}|VRB)(\d{2,3})(?:G(\d{2,3}))?(KT|MPS)\b")
_VIS = re.compile(r"(?<=\s)(\d{4})(?=\s)")
_CAVOK = re.compile(r"\bCAVOK\b")
_SM = re.compile(r"\b(\d+)?\s?(\d/\d)?SM\b")
_CLOUD = re.compile(r"\b(BKN|OVC|VV)(\d{3}|///)")
_TEMP = re.compile(r"\b(M?\d{2})/(M?\d{2})?\b")
_TREND = re.compile(r"\b(TEMPO|BECMG|NOSIG|RMK)\b")


_WX_TOKEN = re.compile(
    r"^(?:\+|-|VC)?(?:MI|PR|BC|DR|BL|SH|TS|FZ)?"
    r"(?:DZ|RA|SN|SG|IC|PL|GR|GS|UP|BR|FG|FU|VA|DU|SA|HZ|PO|SQ|FC|SS|DS)*$"
)


def _wx_flags(body: str) -> dict[str, int]:
    tokens = [t for t in body.split() if len(t) >= 2 and _WX_TOKEN.match(t)]

    def has(*codes: str) -> int:
        return int(any(c in t for t in tokens for c in codes))

    return {
        "ts": has("TS"),
        "sn": has("SN", "SG", "PL", "FZRA", "FZDZ"),
        "fg": has("FG", "BR"),
        "ra": has("RA", "DZ", "SH"),
    }


def decode(raw: str) -> dict[str, float]:
    raw = f" {raw.strip()} "
    m = _TREND.search(raw)
    body, trend = (raw[: m.start()], raw[m.start() :]) if m else (raw, "")
    out: dict[str, float] = dict.fromkeys(WX_FIELDS, np.nan)

    if w := _WIND.search(body):
        k = 1.94384 if w.group(4) == "MPS" else 1.0
        out["wind_kt"] = int(w.group(2)) * k
        out["gust_kt"] = int(w.group(3)) * k if w.group(3) else out["wind_kt"]
    if _CAVOK.search(body):
        out["vis_m"], out["ceil_ft"] = 9999, 5000
    elif v := _VIS.search(body):
        out["vis_m"] = int(v.group(1))
    elif sm := _SM.search(body):  # US stations
        whole = float(sm.group(1) or 0)
        frac = float(Fraction(sm.group(2))) if sm.group(2) else 0.0
        out["vis_m"] = min(9999, (whole + frac) * 1609)
    clouds = [int(h) * 100 for _, h in _CLOUD.findall(body) if h != "///"]
    if clouds:
        out["ceil_ft"] = min(min(clouds), 5000)
    elif np.isnan(out["ceil_ft"]):
        out["ceil_ft"] = 5000
    if t := _TEMP.search(body):
        out["temp_c"] = float(t.group(1).replace("M", "-"))
    out.update(_wx_flags(body))

    out["trend_adverse"] = _trend_adverse(trend)
    return out


def _trend_adverse(trend: str) -> float:
    """1 if a TEMPO/BECMG group forecasts TS, snow, visibility < 1500 m or ceiling < 500 ft."""
    trend = trend.split(" RMK")[0]
    if not trend.strip() or trend.lstrip().startswith(("NOSIG", "RMK")):
        return 0.0
    flags = _wx_flags(trend)
    vis = [int(x) for x in _VIS.findall(trend + " ")]
    ceil = [int(h) * 100 for _, h in _CLOUD.findall(trend) if h != "///"]
    adverse = flags["ts"] or flags["sn"] or (vis and min(vis) < 1500) or (ceil and min(ceil) < 500)
    return float(bool(adverse))


def decode_frame(df: pd.DataFrame) -> pd.DataFrame:
    """IEM METAR frame (icao, time, metar) -> canonical weather_obs."""
    dec = pd.DataFrame([decode(m) for m in df["metar"].astype(str)], index=df.index)
    out = pd.concat([df[["icao", "time"]], dec], axis=1)
    out["severity"] = severity(out)
    return out
