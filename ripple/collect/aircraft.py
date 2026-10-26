"""OpenSky aircraft database -> LOT fleet table (icao24, registration, type)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ripple import config
from ripple.collect.http import client

URL = "https://s3.opensky-network.org/data-samples/metadata/aircraftDatabase.csv"


def update_fleet(out: Path | None = None) -> Path:
    out = out or config.RAW / "fleet.parquet"
    tmp = config.RAW / "aircraftDatabase.csv"
    with client(timeout=600) as c, c.stream("GET", URL) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    db = pd.read_csv(tmp, dtype=str, on_bad_lines="skip", quotechar="'")
    db.columns = [c.strip("'") for c in db.columns]
    lot = db[
        (db["operatoricao"] == config.AIRLINE_ICAO)
        | db["registration"].fillna("").str.match(r"^SP-L[A-Z]{2}$")
    ]
    keep = [
        "icao24",
        "registration",
        "typecode",
        "model",
        "manufacturername",
        "operatoricao",
        "built",
    ]
    lot = lot[[c for c in keep if c in lot.columns]].drop_duplicates("icao24")
    lot.to_parquet(out, index=False)
    tmp.unlink()
    return out


def load_fleet() -> pd.DataFrame:
    return pd.read_parquet(config.RAW / "fleet.parquet")
