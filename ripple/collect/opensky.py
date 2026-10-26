"""OpenSky Network REST client.

Two modes:

* anonymous - only ``/states/all`` (current state vectors). 400 API credits per day; a bounding box
  under 25 square degrees costs 1 credit, so the WAW poller can run every ~4 minutes.
* authenticated - OAuth2 client credentials (create an API client in your OpenSky account page and put
  ``OPENSKY_CLIENT_ID`` / ``OPENSKY_CLIENT_SECRET`` in ``.env``). Unlocks ``/flights/*`` which gives
  firstSeen/lastSeen and estimated airports per flight - the best free source of actual times.

Historical bulk access (Trino / Impala shell) needs an academic application, see docs/opensky_application.md.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd

from ripple import config
from ripple.collect.http import client, get_with_retry

log = logging.getLogger(__name__)

API = "https://opensky-network.org/api"
TOKEN_URL = (
    "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
)

STATE_FIELDS = [
    "icao24",
    "callsign",
    "origin_country",
    "time_position",
    "last_contact",
    "longitude",
    "latitude",
    "baro_altitude",
    "on_ground",
    "velocity",
    "true_track",
    "vertical_rate",
    "sensors",
    "geo_altitude",
    "squawk",
    "spi",
    "position_source",
    "category",
]

FLIGHT_FIELDS = [
    "icao24",
    "firstSeen",
    "estDepartureAirport",
    "lastSeen",
    "estArrivalAirport",
    "callsign",
    "estDepartureAirportHorizDistance",
    "estDepartureAirportVertDistance",
    "estArrivalAirportHorizDistance",
    "estArrivalAirportVertDistance",
    "departureAirportCandidatesCount",
    "arrivalAirportCandidatesCount",
]


@dataclass
class _Token:
    value: str
    expires_at: float


class OpenSky:
    def __init__(self, client_id: str | None = None, client_secret: str | None = None):
        self.client_id = client_id or os.environ.get("OPENSKY_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get("OPENSKY_CLIENT_SECRET")
        self._token: _Token | None = None
        self._http = client()

    @property
    def authenticated(self) -> bool:
        return bool(self.client_id and self.client_secret)

    def _headers(self) -> dict[str, str]:
        if not self.authenticated:
            return {}
        if self._token is None or time.time() > self._token.expires_at - 60:
            r = self._http.post(
                TOKEN_URL,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                },
            )
            r.raise_for_status()
            body = r.json()
            self._token = _Token(body["access_token"], time.time() + body.get("expires_in", 1800))
        return {"Authorization": f"Bearer {self._token.value}"}

    def _get(self, path: str, **params) -> httpx.Response:
        r = get_with_retry(self._http, f"{API}{path}", params=params, headers=self._headers())
        remaining = r.headers.get("X-Rate-Limit-Remaining")
        if remaining is not None:
            log.info("OpenSky credits remaining: %s", remaining)
        return r

    # --- state vectors -------------------------------------------------------------------------
    def states(self, bbox: dict[str, float] | None = None) -> dict:
        r = self._get("/states/all", **(bbox or {}))
        if r.status_code == 429:
            raise RateLimited(r.headers.get("X-Rate-Limit-Retry-After-Seconds", "?"))
        r.raise_for_status()
        return r.json()

    # --- flights (authenticated only) ----------------------------------------------------------
    def _flights(self, path: str, **params) -> pd.DataFrame:
        if not self.authenticated:
            raise PermissionError(
                "OpenSky /flights endpoints need OPENSKY_CLIENT_ID/SECRET in .env"
            )
        r = self._get(path, **params)
        if r.status_code == 404:  # OpenSky returns 404 for "no flights in interval"
            return pd.DataFrame(columns=FLIGHT_FIELDS)
        r.raise_for_status()
        return pd.DataFrame(r.json(), columns=FLIGHT_FIELDS)

    def departures(self, airport: str, begin: int, end: int) -> pd.DataFrame:
        return self._flights("/flights/departure", airport=airport, begin=begin, end=end)

    def arrivals(self, airport: str, begin: int, end: int) -> pd.DataFrame:
        return self._flights("/flights/arrival", airport=airport, begin=begin, end=end)

    def aircraft_flights(self, icao24: str, begin: int, end: int) -> pd.DataFrame:
        return self._flights("/flights/aircraft", icao24=icao24, begin=begin, end=end)


class RateLimited(RuntimeError):
    pass


def poll_states_once(out_dir: Path | None = None) -> int:
    """Append one snapshot of every aircraft in the WAW box to a daily JSONL file.

    We keep all operators, not only LOT: other airlines' movements are the congestion signal.
    """
    out_dir = out_dir or config.RAW / "opensky_states"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = OpenSky(client_id="", client_secret="").states(config.WAW_BBOX)  # anonymous on purpose
    ts = data.get("time") or int(time.time())
    rows = [dict(zip(STATE_FIELDS, s, strict=False)) for s in (data.get("states") or [])]
    for row in rows:
        row["callsign"] = (row["callsign"] or "").strip()
        row["snapshot_time"] = ts
    day = datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%d")
    with open(out_dir / f"{day}.jsonl", "a") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    return len(rows)


def fetch_hub_flights(day: str, out_dir: Path | None = None, sky: OpenSky | None = None) -> Path:
    """Download all EPWA departures and arrivals for one UTC day (needs credentials)."""
    out_dir = out_dir or config.RAW / "opensky_flights"
    out_dir.mkdir(parents=True, exist_ok=True)
    sky = sky or OpenSky()
    begin = int(pd.Timestamp(day, tz="UTC").timestamp())
    end = begin + 86400
    dep = sky.departures(config.HUB, begin, end).assign(direction="departure")
    arr = sky.arrivals(config.HUB, begin, end).assign(direction="arrival")
    df = pd.concat([dep, arr], ignore_index=True)
    df["callsign"] = df["callsign"].fillna("").str.strip()
    path = out_dir / f"{day}.parquet"
    df.to_parquet(path, index=False)
    return path
