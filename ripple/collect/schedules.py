"""Scheduled vs actual times for flights at WAW.

Why not the Chopin Airport website? Checked on 2026-10-26: robots.txt allows crawling, but the
arrivals/departures board (``/pl/przyloty-i-odloty/``) sits behind an F5 bot-defence challenge with a
CAPTCHA. That is a clear "no automated access" signal, so Ripple does not scrape it. See
docs/data_sources.md for the decision and the paid alternatives.

The default provider is AeroDataBox (FIDS endpoint, withLeg=true gives scheduled / revised / runway
times at *both* ends of each leg plus the aircraft registration and Mode-S address). One call covers
up to 12 hours, so WAW costs 2 calls x 2 units per day.

Licence note: AeroDataBox terms (art. 5.5) cap caching of raw API contents at 7 days unless the plan
or a written exemption allows more. Ask them for an academic exemption before keeping raw files for
months; ``--retain-days`` below enforces the cap if you do not have one.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from ripple import config
from ripple.collect.http import client, get_with_retry

log = logging.getLogger(__name__)

HOSTS = {
    "rapidapi": "https://aerodatabox.p.rapidapi.com",
    "apimarket": "https://prod.api.market/api/v1/aedbx/aerodatabox",
    "direct": "https://api.aerodatabox.com",
}


class AeroDataBox:
    def __init__(self, key: str | None = None, channel: str | None = None):
        self.key = key or os.environ.get("AERODATABOX_KEY")
        self.channel = channel or os.environ.get("AERODATABOX_CHANNEL", "rapidapi")
        if not self.key:
            raise PermissionError("Set AERODATABOX_KEY (and AERODATABOX_CHANNEL) in .env")
        self.base = HOSTS[self.channel]

    def _headers(self) -> dict[str, str]:
        if self.channel == "rapidapi":
            return {"X-RapidAPI-Key": self.key, "X-RapidAPI-Host": "aerodatabox.p.rapidapi.com"}
        if self.channel == "apimarket":
            return {"x-api-market-key": self.key}
        return {"X-Api-Key": self.key}

    def fids(self, airport: str, from_local: datetime, to_local: datetime) -> dict:
        url = (
            f"{self.base}/flights/airports/icao/{airport}/"
            f"{from_local:%Y-%m-%dT%H:%M}/{to_local:%Y-%m-%dT%H:%M}"
        )
        params = dict(
            direction="Both",
            withLeg="true",
            withCancelled="true",
            withCodeshared="false",
            withCargo="false",
            withPrivate="false",
            withLocation="false",
        )
        with client(headers=self._headers()) as c:
            r = get_with_retry(c, url, params=params)
        if r.status_code == 204:
            return {"departures": [], "arrivals": []}
        r.raise_for_status()
        return r.json()


def collect_day(day: str, out_dir: Path | None = None, provider: AeroDataBox | None = None) -> Path:
    """Fetch the WAW board for one local calendar day (two 12 h windows) and save raw JSON."""
    out_dir = out_dir or config.RAW / "aerodatabox"
    out_dir.mkdir(parents=True, exist_ok=True)
    provider = provider or AeroDataBox()
    d = datetime.fromisoformat(day)
    payload = {"departures": [], "arrivals": []}
    for start in (d, d + timedelta(hours=12)):
        part = provider.fids(config.HUB, start, start + timedelta(hours=11, minutes=59))
        for k in payload:
            payload[k] += part.get(k, [])
        time.sleep(1.1)  # 1 request / second on the free plan
    payload["fetched_at"] = datetime.now(UTC).isoformat()
    path = out_dir / f"{day}.json"
    path.write_text(json.dumps(payload))
    return path


def purge_older_than(days: int, out_dir: Path | None = None) -> int:
    out_dir = out_dir or config.RAW / "aerodatabox"
    cutoff = time.time() - days * 86400
    n = 0
    for p in out_dir.glob("*.json"):
        if p.stat().st_mtime < cutoff:
            p.unlink()
            n += 1
    return n


def _t(m: dict | None, key: str) -> pd.Timestamp:
    v = ((m or {}).get(key) or {}).get("utc")
    return pd.Timestamp(v) if v else pd.NaT


def normalise(payload: dict) -> pd.DataFrame:
    """AeroDataBox FIDS JSON -> one row per LOT leg in the canonical flight schema."""
    rows = []
    for direction in ("departures", "arrivals"):
        for f in payload.get(direction, []):
            airline = (f.get("airline") or {}).get("icao") or ""
            number = (f.get("number") or "").replace(" ", "")
            if airline != config.AIRLINE_ICAO and not number.startswith(config.AIRLINE_IATA):
                continue
            dep, arr = f.get("departure") or {}, f.get("arrival") or {}
            if direction == "departures":
                dep_ap, arr_ap = config.HUB, (arr.get("airport") or {}).get("icao")
            else:
                dep_ap, arr_ap = (dep.get("airport") or {}).get("icao"), config.HUB
            ac = f.get("aircraft") or {}
            rows.append(
                dict(
                    flight_no=number,
                    callsign=f.get("callSign"),
                    dep_airport=dep_ap,
                    arr_airport=arr_ap,
                    sched_dep=_t(dep, "scheduledTime"),
                    sched_arr=_t(arr, "scheduledTime"),
                    # Prefer gate times (revised) when runway times are also given; both are kept.
                    actual_dep=_t(dep, "revisedTime"),
                    actual_arr=_t(arr, "revisedTime"),
                    runway_dep=_t(dep, "runwayTime"),
                    runway_arr=_t(arr, "runwayTime"),
                    tail=ac.get("reg"),
                    icao24=(ac.get("modeS") or "").lower() or None,
                    ac_model=ac.get("model"),
                    status=f.get("status"),
                    cancelled=f.get("status") in ("Canceled", "CanceledUncertain"),
                    source="aerodatabox",
                )
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.drop_duplicates(["flight_no", "sched_dep"]).reset_index(drop=True)
