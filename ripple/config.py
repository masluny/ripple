"""Paths, network definition and global constants."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA = Path(os.environ.get("RIPPLE_DATA", ROOT / "data"))
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
LOGS = DATA / "logs"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"

HUB = "EPWA"
AIRLINE_ICAO = "LOT"
AIRLINE_IATA = "LO"

# Prediction lead times (hours) and the quantile grid shared by every model.
HORIZONS = (1, 3, 6)
QUANTILES = tuple(round(q / 20, 2) for q in range(1, 20))  # 0.05 ... 0.95

# Minimum connecting time at WAW (minutes). LOT publishes 25 min for Schengen-Schengen and
# longer for non-Schengen; we use one conservative figure because we do not know passenger flows.
MCT_MIN = 35
CONNECTION_WINDOW_MIN = (MCT_MIN, 240)


@dataclass(frozen=True)
class Airport:
    icao: str
    iata: str
    name: str
    lat: float
    lon: float
    tz: str
    schengen: bool = True
    longhaul: bool = False


# LOT network (approximate winter 2026/27 shape). Coordinates rounded to ~1 km.
AIRPORTS: dict[str, Airport] = {
    a.icao: a
    for a in [
        Airport("EPWA", "WAW", "Warsaw Chopin", 52.166, 20.967, "Europe/Warsaw"),
        # Domestic
        Airport("EPKK", "KRK", "Kraków", 50.078, 19.785, "Europe/Warsaw"),
        Airport("EPGD", "GDN", "Gdańsk", 54.378, 18.466, "Europe/Warsaw"),
        Airport("EPWR", "WRO", "Wrocław", 51.103, 16.886, "Europe/Warsaw"),
        Airport("EPKT", "KTW", "Katowice", 50.474, 19.080, "Europe/Warsaw"),
        Airport("EPPO", "POZ", "Poznań", 52.421, 16.826, "Europe/Warsaw"),
        Airport("EPSC", "SZZ", "Szczecin", 53.585, 14.902, "Europe/Warsaw"),
        Airport("EPRZ", "RZE", "Rzeszów", 50.110, 22.019, "Europe/Warsaw"),
        Airport("EPBY", "BZG", "Bydgoszcz", 53.097, 17.978, "Europe/Warsaw"),
        Airport("EPSY", "SZY", "Olsztyn-Mazury", 53.482, 20.938, "Europe/Warsaw"),
        # Europe
        Airport("EGLL", "LHR", "London Heathrow", 51.470, -0.454, "Europe/London", False),
        Airport("LFPG", "CDG", "Paris CDG", 49.010, 2.548, "Europe/Paris"),
        Airport("EDDF", "FRA", "Frankfurt", 50.033, 8.571, "Europe/Berlin"),
        Airport("EDDM", "MUC", "Munich", 48.354, 11.786, "Europe/Berlin"),
        Airport("EDDB", "BER", "Berlin", 52.367, 13.503, "Europe/Berlin"),
        Airport("EDDL", "DUS", "Düsseldorf", 51.289, 6.767, "Europe/Berlin"),
        Airport("EDDH", "HAM", "Hamburg", 53.630, 9.988, "Europe/Berlin"),
        Airport("EHAM", "AMS", "Amsterdam", 52.309, 4.764, "Europe/Amsterdam"),
        Airport("EBBR", "BRU", "Brussels", 50.901, 4.484, "Europe/Brussels"),
        Airport("LSZH", "ZRH", "Zurich", 47.458, 8.548, "Europe/Zurich"),
        Airport("LSGG", "GVA", "Geneva", 46.238, 6.109, "Europe/Zurich"),
        Airport("LOWW", "VIE", "Vienna", 48.110, 16.570, "Europe/Vienna"),
        Airport("LKPR", "PRG", "Prague", 50.101, 14.260, "Europe/Prague"),
        Airport("LHBP", "BUD", "Budapest", 47.437, 19.256, "Europe/Budapest"),
        Airport("EKCH", "CPH", "Copenhagen", 55.618, 12.656, "Europe/Copenhagen"),
        Airport("ESSA", "ARN", "Stockholm Arlanda", 59.652, 17.919, "Europe/Stockholm"),
        Airport("ENGM", "OSL", "Oslo", 60.194, 11.100, "Europe/Oslo"),
        Airport("EFHK", "HEL", "Helsinki", 60.317, 24.963, "Europe/Helsinki"),
        Airport("EYVI", "VNO", "Vilnius", 54.634, 25.286, "Europe/Vilnius"),
        Airport("EVRA", "RIX", "Riga", 56.924, 23.971, "Europe/Riga"),
        Airport("EETN", "TLL", "Tallinn", 59.413, 24.833, "Europe/Tallinn"),
        Airport("LROP", "OTP", "Bucharest", 44.571, 26.085, "Europe/Bucharest", False),
        Airport("LBSF", "SOF", "Sofia", 42.697, 23.411, "Europe/Sofia", False),
        Airport("LGAV", "ATH", "Athens", 37.936, 23.947, "Europe/Athens"),
        Airport("LIRF", "FCO", "Rome Fiumicino", 41.800, 12.239, "Europe/Rome"),
        Airport("LIMC", "MXP", "Milan Malpensa", 45.630, 8.723, "Europe/Rome"),
        Airport("LEBL", "BCN", "Barcelona", 41.297, 2.078, "Europe/Madrid"),
        Airport("LEMD", "MAD", "Madrid", 40.472, -3.561, "Europe/Madrid"),
        Airport("LPPT", "LIS", "Lisbon", 38.774, -9.134, "Europe/Lisbon"),
        Airport("EIDW", "DUB", "Dublin", 53.421, -6.270, "Europe/Dublin", False),
        Airport("EGCC", "MAN", "Manchester", 53.354, -2.275, "Europe/London", False),
        Airport("LFMN", "NCE", "Nice", 43.658, 7.216, "Europe/Paris"),
        Airport("LYBE", "BEG", "Belgrade", 44.818, 20.309, "Europe/Belgrade", False),
        Airport("LUKK", "KIV", "Chișinău", 46.927, 28.931, "Europe/Chisinau", False),
        Airport("UGTB", "TBS", "Tbilisi", 41.669, 44.955, "Asia/Tbilisi", False),
        Airport("UDYZ", "EVN", "Yerevan", 40.147, 44.396, "Asia/Yerevan", False),
        Airport("LLBG", "TLV", "Tel Aviv", 32.011, 34.887, "Asia/Jerusalem", False),
        Airport("LCLK", "LCA", "Larnaca", 34.875, 33.625, "Asia/Nicosia", False),
        # Long-haul
        Airport("KJFK", "JFK", "New York JFK", 40.640, -73.779, "America/New_York", False, True),
        Airport("KEWR", "EWR", "Newark", 40.690, -74.174, "America/New_York", False, True),
        Airport("KORD", "ORD", "Chicago O'Hare", 41.978, -87.905, "America/Chicago", False, True),
        Airport("CYYZ", "YYZ", "Toronto", 43.677, -79.631, "America/Toronto", False, True),
        Airport("KMIA", "MIA", "Miami", 25.793, -80.291, "America/New_York", False, True),
        Airport("KLAX", "LAX", "Los Angeles", 33.942, -118.408, "America/Los_Angeles", False, True),
        Airport("RKSI", "ICN", "Seoul Incheon", 37.460, 126.441, "Asia/Seoul", False, True),
        Airport("RJTT", "HND", "Tokyo Haneda", 35.552, 139.780, "Asia/Tokyo", False, True),
        Airport("VIDP", "DEL", "Delhi", 28.566, 77.103, "Asia/Kolkata", False, True),
        Airport("UTTT", "TAS", "Tashkent", 41.258, 69.281, "Asia/Tashkent", False, True),
    ]
}

IATA_TO_ICAO = {a.iata: a.icao for a in AIRPORTS.values()}

# Bounding box around EPWA used by the anonymous OpenSky state-vector poller (< 25 sq deg = 1 credit).
WAW_BBOX = dict(lamin=51.95, lomin=20.55, lamax=52.40, lomax=21.40)


def ensure_dirs() -> None:
    for d in (RAW, INTERIM, PROCESSED, LOGS, ARTIFACTS, REPORTS):
        d.mkdir(parents=True, exist_ok=True)
