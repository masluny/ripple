import json

import pandas as pd

from ripple import config
from ripple.collect import opensky
from ripple.collect.schedules import normalise


def test_quantile_grid_and_airports():
    assert (
        config.QUANTILES[0] == 0.05 and config.QUANTILES[-1] == 0.95 and len(config.QUANTILES) == 19
    )
    assert config.HUB in config.AIRPORTS
    assert all(-90 < a.lat < 90 and -180 < a.lon < 180 for a in config.AIRPORTS.values())
    assert len({a.iata for a in config.AIRPORTS.values()}) == len(config.AIRPORTS)


def _mv(icao, sched, revised=None, runway=None):
    m = {"airport": {"icao": icao}, "scheduledTime": {"utc": sched}}
    if revised:
        m["revisedTime"] = {"utc": revised}
    if runway:
        m["runwayTime"] = {"utc": runway}
    return m


def test_aerodatabox_normalise_keeps_lot_and_orients_legs():
    payload = {
        "departures": [
            {
                "number": "LO 333",
                "callSign": "LOT333",
                "status": "Departed",
                "airline": {"icao": "LOT"},
                "aircraft": {"reg": "SP-LWA", "modeS": "48AD01", "model": "Embraer 195"},
                "departure": {
                    "scheduledTime": {"utc": "2026-10-20 05:00Z"},
                    "revisedTime": {"utc": "2026-10-20 05:12Z"},
                },
                "arrival": _mv("LFPG", "2026-10-20 07:20Z", "2026-10-20 07:31Z"),
            },
            {"number": "W6 1", "airline": {"icao": "WZZ"}, "departure": {}, "arrival": {}},
        ],
        "arrivals": [
            {
                "number": "LO 334",
                "status": "Arrived",
                "airline": {"icao": "LOT"},
                "aircraft": {"reg": "SP-LWA"},
                "departure": _mv("LFPG", "2026-10-20 08:10Z"),
                "arrival": {"scheduledTime": {"utc": "2026-10-20 10:20Z"}},
            }
        ],
    }
    df = normalise(json.loads(json.dumps(payload)))
    assert list(df.flight_no) == ["LO333", "LO334"]
    d = df.iloc[0]
    assert (d.dep_airport, d.arr_airport, d.icao24) == ("EPWA", "LFPG", "48ad01")
    assert (d.actual_dep - d.sched_dep) == pd.Timedelta(minutes=12)
    assert df.iloc[1].dep_airport == "LFPG" and df.iloc[1].arr_airport == "EPWA"
    assert pd.isna(df.iloc[1].actual_arr)


def test_poll_states_writes_jsonl(tmp_path, monkeypatch):
    fake = {
        "time": 1793050100,
        "states": [
            [
                "48af10",
                "LOT304  ",
                "Poland",
                1,
                2,
                20.97,
                52.16,
                None,
                True,
                7.7,
                337.5,
                None,
                None,
                None,
                "1000",
                False,
                0,
            ]
        ],
    }
    monkeypatch.setattr(opensky.OpenSky, "states", lambda self, bbox=None: fake)
    assert opensky.poll_states_once(tmp_path) == 1
    row = json.loads((tmp_path / "2026-10-26.jsonl").read_text())
    assert (
        row["callsign"] == "LOT304"
        and row["on_ground"] is True
        and row["snapshot_time"] == 1793050100
    )
