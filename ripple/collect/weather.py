"""Weather collectors.

* METAR history: Iowa Environmental Mesonet ASOS archive (global METARs, free, no key).
* TAF: IEM only archives US TAFs, so we archive TAFs ourselves going forward from the
  aviationweather.gov Data API (NOAA, public). Historical TAFs before the collector started are
  simply missing; the models fall back to METAR trend groups (TEMPO/BECMG) in that case.
"""

from __future__ import annotations

import io
import json
import logging
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd

from ripple import config
from ripple.collect.http import client, get_with_retry

log = logging.getLogger(__name__)

IEM_ASOS = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
AWC_TAF = "https://aviationweather.gov/api/data/taf"


def fetch_metar(start: date, end: date, stations: list[str] | None = None) -> pd.DataFrame:
    """Raw METARs (routine + specials) for [start, end) in UTC."""
    stations = stations or list(config.AIRPORTS)
    params: list[tuple[str, str]] = [("station", s) for s in stations]
    params += [
        ("data", "metar"),
        ("year1", str(start.year)),
        ("month1", str(start.month)),
        ("day1", str(start.day)),
        ("year2", str(end.year)),
        ("month2", str(end.month)),
        ("day2", str(end.day)),
        ("tz", "Etc/UTC"),
        ("format", "onlycomma"),
        ("latlon", "no"),
        ("missing", "M"),
        ("trace", "T"),
        ("direct", "no"),
        ("report_type", "3"),
        ("report_type", "4"),
    ]
    with client(timeout=300) as c:
        r = get_with_retry(c, IEM_ASOS, params=params)
        r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df = df.rename(columns={"station": "icao", "valid": "time"})
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.drop_duplicates(["icao", "time", "metar"]).reset_index(drop=True)


def update_metar_archive(days_back: int = 3, out_dir: Path | None = None) -> Path:
    """Refresh the monthly METAR parquet files covering the last ``days_back`` days."""
    out_dir = out_dir or config.RAW / "metar"
    out_dir.mkdir(parents=True, exist_ok=True)
    end = datetime.now(UTC).date() + timedelta(days=1)
    start = end - timedelta(days=days_back + 1)
    return backfill_metar(start, end, out_dir)


def backfill_metar(start: date, end: date, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or config.RAW / "metar"
    out_dir.mkdir(parents=True, exist_ok=True)
    cur = date(start.year, start.month, 1)
    path = out_dir
    while cur < end:
        nxt = date(cur.year + cur.month // 12, cur.month % 12 + 1, 1)
        lo, hi = max(cur, start), min(nxt, end)
        log.info("METAR %s .. %s", lo, hi)
        new = fetch_metar(lo, hi)
        path = out_dir / f"{cur:%Y-%m}.parquet"
        if path.exists():
            new = pd.concat([pd.read_parquet(path), new]).drop_duplicates(["icao", "time", "metar"])
        new.sort_values(["icao", "time"]).to_parquet(path, index=False)
        cur = nxt
    return path


def poll_taf(out_dir: Path | None = None, stations: list[str] | None = None) -> int:
    """Append the current TAFs for every network airport to a daily JSONL archive."""
    out_dir = out_dir or config.RAW / "taf"
    out_dir.mkdir(parents=True, exist_ok=True)
    stations = stations or list(config.AIRPORTS)
    rows: list[dict] = []
    with client() as c:
        for i in range(0, len(stations), 40):
            r = get_with_retry(
                c, AWC_TAF, params={"ids": ",".join(stations[i : i + 40]), "format": "json"}
            )
            r.raise_for_status()
            rows += r.json() if r.content else []
    now = datetime.now(UTC)
    with open(out_dir / f"{now:%Y-%m-%d}.jsonl", "a") as f:
        for row in rows:
            row["fetched_at"] = now.isoformat()
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    return len(rows)
