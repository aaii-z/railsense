from pathlib import Path

import pandas as pd
from rapidfuzz import process


MAX_STATIONS = 3

_LONDON_TERMINALS = {"EUS", "VIC", "PAD", "KGX", "LST", "CHX", "WAT", "MYB", "STP"}

# RTJP gives "Any London Terminal" fares so all terminals show the same price.
# Pick the actual terminal for this origin so the booking link works.
_ORIGIN_TO_LONDON_TERMINAL: dict[str, str] = {
    # Avanti West Coast → Euston
    "LIV": "EUS", "LPY": "EUS",
    "MAN": "EUS", "MCV": "EUS",
    "BHM": "EUS",
    "GLC": "EUS",                                    # Glasgow Central (Avanti)
    # LNER → King's Cross (faster/more frequent than Avanti for these origins)
    "EDB": "KGX",                                    # Edinburgh (LNER primary)
    "GLQ": "KGX",                                    # Glasgow Queen Street via Edinburgh
    "LDS": "KGX", "SHF": "KGX", "NCL": "KGX", "YRK": "KGX",
    # Chiltern Railways → Marylebone
    "BMO": "MYB",                                    # Birmingham Moor Street
    "BSW": "MYB",                                    # Birmingham Snow Hill
    # Greater Anglia → Liverpool Street
    "NRW": "LST", "CBG": "LST", "IPS": "LST", "COL": "LST",
    # GWR → Paddington
    "BRI": "PAD", "BPW": "PAD", "OXF": "PAD",
    "CDF": "PAD", "SWA": "PAD", "RDG": "PAD", "BTH": "PAD",
    # South Western Railway → Waterloo
    "SOU": "WAT", "BMH": "WAT", "PMS": "WAT", "PMH": "WAT", "SAL": "WAT",
    # Southern / Gatwick Express → Victoria
    "BTN": "VIC", "GTW": "VIC",
    # East Midlands Railway → St Pancras
    "NOT": "STP", "LEI": "STP", "DBY": "STP",
}

# Some CRS codes have no direct long-distance service. Remap to the main
# terminal so the National Rail booking link finds a real train.
CANONICAL_FOR_BOOKING: dict[str, str] = {
    "LPY": "LIV",  # Liverpool South Parkway → Liverpool Lime Street (Avanti)
    "MCV": "MAN",  # Manchester Victoria → Manchester Piccadilly (Avanti)
    "BMO": "BHM",  # Birmingham Moor Street → Birmingham New Street
    "BSW": "BHM",  # Birmingham Snow Hill → Birmingham New Street
    "BPW": "BRI",  # Bristol Parkway → Bristol Temple Meads (GWR)
    "GLQ": "GLC",  # Glasgow Queen Street → Glasgow Central
}

# Big cities have many stations in the CSV. Pin the useful terminals first.
_CITY_PRIORITY: dict[str, list[str]] = {
    "london":       ["LST", "VIC", "PAD", "KGX", "EUS", "CHX", "WAT"],
    "birmingham":   ["BHM", "BMO", "BSW"],
    "manchester":   ["MAN", "MCV"],
    "edinburgh":    ["EDB"],
    "glasgow":      ["GLC", "GLQ"],
    "bristol":      ["BRI", "BPW"],
    "leeds":        ["LDS"],
    "sheffield":    ["SHF"],
    "liverpool":    ["LIV", "LPY"],
    "newcastle":    ["NCL"],
    "norwich":      ["NRW"],
    "cambridge":    ["CBG"],
    "oxford":       ["OXF"],
    "york":         ["YRK"],
}


def _load_station_lookup(
    csv_path: str = str(Path(__file__).resolve().parents[2] / "data" / "station_cities.csv")
) -> dict[str, list[str]]:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    lookup: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        city = str(row["city"]).strip().lower()
        name = str(row["stationName"]).strip().lower()
        crs  = str(row["crsCode"]).strip().upper()
        for key in (city, name):
            lookup.setdefault(key, [])
            if crs not in lookup[key]:
                lookup[key].append(crs)
    return lookup


STATION_LOOKUP = _load_station_lookup()


class StationAmbiguous(Exception):
    def __init__(self, candidates: list[str]):
        self.candidates = candidates


def resolve_station(user_input: str) -> list[str]:
    """Fuzzy match user input to a station/city name. Returns CRS codes.

    >= 90: accept
    70-89: ambiguous, raises StationAmbiguous
    < 70:  not found, returns []
    """
    query = user_input.strip().lower()

    if query in _CITY_PRIORITY:
        return _CITY_PRIORITY[query][:MAX_STATIONS]

    if query in STATION_LOOKUP:
        return STATION_LOOKUP[query][:MAX_STATIONS]

    matches = process.extract(query, STATION_LOOKUP.keys(), limit=3)
    if not matches or matches[0][1] < 70:
        return []

    top_name, top_score, _ = matches[0]

    # Typos like 'londn' shouldn't trigger a confirmation prompt for big cities.
    if top_name in _CITY_PRIORITY and top_score >= 80:
        return _CITY_PRIORITY[top_name][:MAX_STATIONS]

    if top_score >= 90:
        return STATION_LOOKUP[top_name][:1]

    # 70-89: ambiguous — show top match plus anything within 10 points.
    candidates = [matches[0][0].title()]
    for name, score, _ in matches[1:]:
        if score >= top_score - 10 and name.title() not in candidates:
            candidates.append(name.title())
    raise StationAmbiguous(candidates)


def london_terminal_for(counterpart_crs: str) -> str:
    """Return the London terminal that serves the given origin/destination CRS."""
    return _ORIGIN_TO_LONDON_TERMINAL.get(counterpart_crs, "EUS")


def is_london(codes: list[str]) -> bool:
    """True if all CRS codes in the list are London terminals."""
    return bool(codes) and all(c in _LONDON_TERMINALS for c in codes)
