from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from itertools import product
from zoneinfo import ZoneInfo
import pandas as pd
from rapidfuzz import process
from zeep.helpers import serialize_object

from rtjp.client import create_client
from llm.client import chat_json


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
MAX_STATIONS   = 3

# RTJP gives "Any London Terminal" fares so all terminals show the same price.
# Pick the actual terminal for this origin so the booking link works.
_LONDON_TERMINALS = {"EUS", "VIC", "PAD", "KGX", "LST", "CHX", "WAT", "MYB", "STP"}
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
_CANONICAL_FOR_BOOKING: dict[str, str] = {
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


class StationAmbiguous(Exception):
    def __init__(self, candidates: list[str]):
        self.candidates = candidates


def resolve_station(user_input: str) -> list[str]:
    """Fuzzy match user input to a station/city. Returns CRS codes.

    >= 90: accept
    70-89: ambiguous, ask user to confirm
    < 70:  give up
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

    # 70-89: ambiguous. Show top match plus anything within 10 points.
    candidates = [matches[0][0].title()]
    for name, score, _ in matches[1:]:
        if score >= top_score - 10 and name.title() not in candidates:
            candidates.append(name.title())
    raise StationAmbiguous(candidates)



TICKET_KEYS = ["origin", "destination", "departure_time", "return_time", "next_question"]


def default_ticket_state() -> dict[str, Any]:
    return {
        "origin":         None,
        "destination":    None,
        "departure_time": None,
        "return_time":    None,
    }



_UK_TZ = ZoneInfo("Europe/London")


def _now_uk() -> datetime:
    return datetime.now(_UK_TZ)


def _now_str() -> str:
    return _now_uk().strftime("%A %d %B %Y, %H:%M %Z")



def _extract_ticket_fields(user_input: str, state: dict[str, Any]) -> dict[str, Any]:
    prompt = (
        f"Current date and time: {_now_str()}\n\n"
        "Extract ticket search fields from the user message and reply with a single JSON object.\n"
        "Always copy origin and destination EXACTLY as the user wrote them, even if the word looks unusual or misspelled. Do NOT leave them null just because the name is unfamiliar.\n"
        "departure_time and return_time must be ISO 8601 (e.g. 2026-05-10T09:00:00). "
        "Use the current date above to resolve relative times like 'tomorrow', 'in 2 hours', 'next Friday'. "
        "Treat 'morning' as 09:00, 'afternoon' as 14:00, 'evening' as 18:00 when no specific time is given.\n"
        "If origin, destination, or departure_time is genuinely absent from the message AND not already in current state, "
        "put one short follow-up question in next_question; otherwise next_question must be null.\n\n"
        "Example input: \"I want to go from Norwich to London tomorrow morning\"\n"
        "Example output: {\"origin\": \"Norwich\", \"destination\": \"London\", \"departure_time\": \"2026-05-18T09:00:00\", \"return_time\": null, \"next_question\": null}\n\n"
        f"Current state: {state}\n"
        f'User: "{user_input}"'
    )
    return chat_json(
        [
            {"role": "system", "content": "You extract structured ticket booking fields from user messages. Reply with JSON only."},
            {"role": "user", "content": prompt},
        ],
        expected_keys=TICKET_KEYS,
    )


def _update_state(state: dict[str, Any], extracted: dict[str, Any]) -> None:
    for key in ("origin", "destination", "departure_time", "return_time"):
        value = extracted.get(key)
        if value and str(value).lower() not in ("none", "null", ""):
            state[key] = value


def _is_complete(state: dict[str, Any]) -> bool:
    return all(state.get(k) for k in ("origin", "destination", "departure_time"))



def _parse_time(time_str: str | None) -> tuple[datetime | None, bool]:
    """Returns (datetime, assumed). assumed=True if we made up the time."""
    if not time_str:
        return None, False
    s = str(time_str).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
        if dt.hour == 0 and dt.minute == 0:
            # midnight is almost certainly "no time given", default to 9am
            dt = dt.replace(hour=9)
            return dt, True
        return dt, False
    except ValueError:
        # bad input, fall back to tomorrow 9am
        fallback = (_now_uk() + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
        return fallback, True



def _booking_link(origin: str, destination: str, depart_by: datetime, is_return: bool = False) -> str:
    """Build a National Rail journey planner URL.

    Quirks found by testing:
      - leavingDate must be DDMMYY (6 digits), not DDMMYYYY.
      - leavingMin must be 00/15/30/45 or the page comes back empty.
        Round down so the requested train still shows up.
    """
    rounded_min = (depart_by.minute // 15) * 15
    return (
        "https://www.nationalrail.co.uk/journey-planner/?"
        f"type={'return' if is_return else 'single'}"
        f"&origin={origin}"
        f"&destination={destination}"
        f"&leavingType=departing"
        f"&leavingDate={depart_by.strftime('%d%m%y')}"
        f"&leavingHour={depart_by.strftime('%H')}"
        f"&leavingMin={rounded_min:02d}"
        f"&adults=1&extraTime=0#O"
    )



def _fetch_plan(origin: str, destination: str, depart_by: datetime, inward_time: datetime | None) -> dict:
    """Call RTJP. Retries when OJP sends back HTML instead of SOAP (happens
    sometimes, probably rate limit / load balancer). Retrying usually works.
    """
    import time as _time
    client = create_client()
    kwargs = {
        "origin":          {"stationCRS": origin},
        "destination":     {"stationCRS": destination},
        "realtimeEnquiry": "STANDARD",
        "outwardTime":     {"departBy": depart_by},
        "via":             [],
        "notVia":          [],
        "directTrains":    False,
        "fareRequestDetails": {
            "passengers": {"adult": 1, "child": 0},
            "fareClass":  "STANDARD",
        },
    }
    if inward_time:
        kwargs["inwardTime"] = {"departBy": inward_time}

    last_exc = None
    for attempt in range(3):
        try:
            return serialize_object(client.service.RealtimeJourneyPlan(**kwargs))
        except Exception as exc:
            msg = str(exc).lower()
            # only retry the HTML-instead-of-SOAP case, not real SOAP errors
            if "root element found is html" in msg or "does not contain a valid" in msg:
                last_exc = exc
                _time.sleep(0.8 * (attempt + 1))
                continue
            raise
    raise last_exc if last_exc else RuntimeError("RTJP failed after retries")


def _min_fare_pence(journey: dict) -> int | None:
    fares = journey.get("fare") or []
    if isinstance(fares, dict):
        fares = [fares]
    prices = []
    for f in fares:
        if isinstance(f, dict) and f.get("totalPrice") is not None:
            try:
                prices.append(int(f["totalPrice"]))
            except (TypeError, ValueError):
                pass
    return min(prices) if prices else None


def _format_time(value) -> str:
    if value is None:
        return "-"
    if hasattr(value, "strftime"):
        return value.strftime("%H:%M")
    s = str(value)
    return s[11:16] if "T" in s else s[:5]


def _station_name(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for k in ("_value_1", "stationCRS", "stationCrs"):
            if value.get(k):
                return str(value[k])
    return "?"


def _extract_journeys(plan: dict) -> list[dict]:
    raw = plan.get("outwardJourney") or []
    if isinstance(raw, dict):
        raw = [raw]
    return [j for j in raw if isinstance(j, dict)]


def _format_journey(journey: dict, depart_by: datetime, is_return: bool) -> dict:
    tt    = journey.get("timetable") or {}
    sched = tt.get("scheduled") or {}
    rt    = tt.get("realtime") or {}
    dep   = rt.get("departure") or sched.get("departure")
    arr   = rt.get("arrival")   or sched.get("arrival")
    pence = _min_fare_pence(journey)
    origin      = _station_name(journey.get("origin"))
    destination = _station_name(journey.get("destination"))
    # use main-terminal codes for the booking link
    booking_origin = _CANONICAL_FOR_BOOKING.get(origin, origin)
    booking_dest   = _CANONICAL_FOR_BOOKING.get(destination, destination)
    if dep is not None and hasattr(dep, "hour"):
        link_time = depart_by.replace(hour=dep.hour, minute=dep.minute, second=0, microsecond=0)
    else:
        link_time = depart_by
    return {
        "origin":      origin,
        "destination": destination,
        "departure":   _format_time(dep),
        "arrival":     _format_time(arr),
        "price":       f"£{pence / 100:.2f}" if pence else "-",
        "link":        _booking_link(booking_origin, booking_dest, link_time, is_return),
        "pence":       pence or float("inf"),
        "dep_dt":      link_time,
    }



def _search_all_pairs(
    origin_codes: list[str],
    dest_codes: list[str],
    depart_by: datetime,
    inward_time: datetime | None,
) -> tuple[list[dict], list[str]]:
    """Search every origin×destination CRS pair, collect all journeys."""
    is_return    = inward_time is not None
    all_journeys = []
    errors = []
    for origin, destination in product(origin_codes, dest_codes):
        try:
            plan      = _fetch_plan(origin, destination, depart_by, inward_time)
            journeys  = _extract_journeys(plan)
            formatted = [_format_journey(j, depart_by, is_return) for j in journeys]
            all_journeys.extend(formatted)
        except Exception as e:
            errors.append(f"{origin}→{destination}: {e}")
    return all_journeys, errors



def handle_ticket_message(user_input: str, state: dict[str, Any], *, debug: bool = False) -> dict[str, Any]:
    dbg: dict[str, Any] = {}

    # Step 1 - extract fields
    extracted = _extract_ticket_fields(user_input, state)
    _update_state(state, extracted)

    # Step 2 - ask follow-up if slots missing
    if not _is_complete(state):
        question = extracted.get("next_question") or "Please tell me your origin, destination, and departure time."
        resp = {"kind": "ticket_search", "done": False, "message": question}
        if debug:
            resp["ticket_debug"] = {"step": "collecting_fields", "state": dict(state)}
        return resp

    # Step 3 - resolve city names to CRS codes
    try:
        origin_codes = resolve_station(str(state["origin"]))
    except StationAmbiguous as e:
        state["origin"] = None
        opts = " or ".join(e.candidates)
        resp = {"kind": "ticket_search", "done": False,
                "message": f"Did you mean {opts}? Please clarify your departure station."}
        if debug:
            resp["ticket_debug"] = {**dbg, "step": "station_ambiguous", "field": "origin", "candidates": e.candidates}
        return resp

    try:
        dest_codes = resolve_station(str(state["destination"]))
    except StationAmbiguous as e:
        state["destination"] = None
        opts = " or ".join(e.candidates)
        resp = {"kind": "ticket_search", "done": False,
                "message": f"Did you mean {opts}? Please clarify your destination station."}
        if debug:
            resp["ticket_debug"] = {**dbg, "step": "station_ambiguous", "field": "destination", "candidates": e.candidates}
        return resp
    # London destination: pick the terminal that actually serves this origin
    # so the booking link points at a real train.
    if dest_codes and all(c in _LONDON_TERMINALS for c in dest_codes):
        primary_origin = origin_codes[0] if origin_codes else ""
        london_terminal = _ORIGIN_TO_LONDON_TERMINAL.get(primary_origin, "EUS")
        dest_codes = [london_terminal]

    # Same thing the other way: London as origin, pick one terminal.
    if origin_codes and all(c in _LONDON_TERMINALS for c in origin_codes):
        primary_dest = dest_codes[0] if dest_codes else ""
        london_terminal = _ORIGIN_TO_LONDON_TERMINAL.get(primary_dest, "EUS")
        origin_codes = [london_terminal]

    dbg["origin_input"]   = state["origin"]
    dbg["origin_crs"]     = origin_codes
    dbg["dest_input"]     = state["destination"]
    dbg["dest_crs"]       = dest_codes
    dbg["departure_time"] = str(state["departure_time"])
    dbg["return_time"]    = str(state.get("return_time"))

    if not origin_codes:
        bad = state["origin"]
        state["origin"] = None
        resp = {"kind": "ticket_search", "done": False,
                "message": f"I couldn't find a station for '{bad}'. Could you be more specific?"}
        if debug:
            resp["ticket_debug"] = {**dbg, "step": "crs_lookup_failed", "failed_for": "origin"}
        return resp

    if not dest_codes:
        bad = state["destination"]
        state["destination"] = None
        resp = {"kind": "ticket_search", "done": False,
                "message": f"I couldn't find a station for '{bad}'. Could you be more specific?"}
        if debug:
            resp["ticket_debug"] = {**dbg, "step": "crs_lookup_failed", "failed_for": "destination"}
        return resp

    # Step 4 - parse times
    depart_by, assumed = _parse_time(str(state["departure_time"]))
    if depart_by is None:
        state["departure_time"] = None
        resp = {"kind": "ticket_search", "done": False, "message": "Please provide a departure time."}
        if debug:
            resp["ticket_debug"] = {**dbg, "step": "time_parse_failed"}
        return resp

    inward_time, _ = _parse_time(state.get("return_time"))
    dbg["depart_by"]    = depart_by.isoformat()
    dbg["inward_time"]  = inward_time.isoformat() if inward_time else None
    dbg["pairs_searched"] = [f"{o}→{d}" for o in origin_codes for d in dest_codes]

    # Step 5 - search all station pair combinations
    all_journeys, search_errors = _search_all_pairs(origin_codes, dest_codes, depart_by, inward_time)
    dbg["journeys_found"] = len(all_journeys)
    if search_errors:
        dbg["search_errors"] = search_errors

    if not all_journeys:
        dbg["step"] = "api_fallback"
        outbound = _booking_link(origin_codes[0], dest_codes[0], depart_by, is_return=inward_time is not None)
        msg = (
            f"I couldn't retrieve live fares right now, but you can check prices and book here: "
            f"[National Rail journey planner]({outbound})"
        )
        if inward_time:
            inbound = _booking_link(dest_codes[0], origin_codes[0], inward_time, is_return=True)
            msg += f"\n\nReturn journey: [National Rail journey planner]({inbound})"
        state.update(default_ticket_state())
        resp = {"kind": "ticket_search", "done": True, "message": msg, "journeys": []}
        if debug:
            resp["ticket_debug"] = dbg
        return resp

    # Step 6 - prefer journeys within ±3 h of requested departure, then sort by price
    window = timedelta(hours=3)
    in_window = [
        j for j in all_journeys
        if abs((j["dep_dt"] - depart_by).total_seconds()) <= window.total_seconds()
    ]
    pool = in_window if in_window else all_journeys
    top5 = sorted(pool, key=lambda j: j["pence"])[:5]

    # Within each (origin, dest) group point every booking link at the cheapest
    # journey's time. NR orders by departure time, so otherwise a later/pricier
    # row would link to a page where the cheaper option is hidden below.
    is_return = inward_time is not None
    groups: dict[tuple[str, str], list[dict]] = {}
    for j in top5:
        groups.setdefault((j["origin"], j["destination"]), []).append(j)
    for (orig, dest), group in groups.items():
        cheapest = min(group, key=lambda j: j["pence"])
        shared_link = _booking_link(orig, dest, cheapest["dep_dt"], is_return)
        for j in group:
            j["link"] = shared_link

    lines = [
        f"{i}. {j['origin']} → {j['destination']} | dep {j['departure']} arr {j['arrival']} | from {j['price']} - [Book ticket]({j['link']})"
        for i, j in enumerate(top5, 1)
    ]

    note   = "Note: departure time was assumed.\n" if assumed else ""
    plural = f"Searched {len(origin_codes)}×{len(dest_codes)} station combinations.\n" if len(origin_codes) + len(dest_codes) > 2 else ""
    prefix = note + plural

    state.update(default_ticket_state())

    resp = {
        "kind":     "ticket_search",
        "done":     True,
        "message":  prefix + "\n".join(lines),
        "journeys": [{k: v for k, v in j.items() if k not in ("pence", "dep_dt")} for j in top5],
    }
    if debug:
        dbg["step"] = "success"
        resp["ticket_debug"] = dbg
    return resp
