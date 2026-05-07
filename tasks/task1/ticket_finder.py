from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from itertools import product
import pandas as pd
from rapidfuzz import process
from zeep.helpers import serialize_object

from rtjp.client import create_client
from tasks.task3.llm import chat_json


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

# Major cities map to many stations in the CSV, pin the most useful terminals first
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


def resolve_station(user_input: str) -> list[str]:
    """Fuzzy match user input to city or station name. Returns CRS codes."""
    query = user_input.strip().lower()

    # Check priority overrides first
    if query in _CITY_PRIORITY:
        return _CITY_PRIORITY[query][:MAX_STATIONS]

    if query in STATION_LOOKUP:
        return STATION_LOOKUP[query][:MAX_STATIONS]
    match, score, _ = process.extractOne(query, STATION_LOOKUP.keys())
    if score >= 75:
        return STATION_LOOKUP[match][:MAX_STATIONS]
    return []



TICKET_KEYS = ["origin", "destination", "departure_time", "return_time", "next_question"]


def default_ticket_state() -> dict[str, Any]:
    return {
        "origin":         None,
        "destination":    None,
        "departure_time": None,
        "return_time":    None,
    }



def _now_uk() -> datetime:
    return datetime.now(timezone.utc)


def _now_str() -> str:
    return _now_uk().strftime("%A %d %B %Y, %H:%M UTC")



def _extract_ticket_fields(user_input: str, state: dict[str, Any]) -> dict[str, Any]:
    prompt = (
        f"Current date and time: {_now_str()}\n\n"
        "Extract ticket search fields from the user message.\n"
        "origin and destination should be city or station names as the user said them.\n"
        "departure_time and return_time must be ISO 8601 format (e.g. 2025-05-10T09:00:00). "
        "Use the current date above to resolve relative times like 'tomorrow', 'in 2 hours', 'next Friday'.\n"
        "If any of origin, destination, or departure_time is missing, "
        "put one short follow-up question in next_question.\n\n"
        f"Current state: {state}\n"
        f'User: "{user_input}"'
    )
    return chat_json(
        [
            {"role": "system", "content": "You extract structured ticket booking fields from user messages."},
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
    """Returns (datetime, assumed) - assumed=True if we fell back to tomorrow 9am."""
    if not time_str:
        return None, False
    s = str(time_str).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
        if dt.hour == 0 and dt.minute == 0:
            dt = dt.replace(hour=9)
            return dt, True  # time was assumed
        return dt, False
    except ValueError:
        fallback = (_now_uk() + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
        return fallback, True



def _booking_link(origin: str, destination: str, depart_by: datetime, is_return: bool = False) -> str:
    """Generate a National Rail journey planner deep link."""
    return (
        "https://www.nationalrail.co.uk/journey-planner/?"
        f"type={'return' if is_return else 'single'}"
        f"&origin={origin}"
        f"&destination={destination}"
        f"&leavingType=departing"
        f"&leavingDate={depart_by.strftime('%d%m%y')}"
        f"&leavingHour={depart_by.strftime('%H')}"
        f"&leavingMin={depart_by.strftime('%M')}"
        f"&adults=1&children=0"
    )



def _fetch_plan(origin: str, destination: str, depart_by: datetime, inward_time: datetime | None) -> dict:
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
    return serialize_object(client.service.RealtimeJourneyPlan(**kwargs))


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
    # Use actual train departure time in booking link if available
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
        "link":        _booking_link(origin, destination, link_time, is_return),
        "pence":       pence or float("inf"),
    }



def _search_all_pairs(
    origin_codes: list[str],
    dest_codes: list[str],
    depart_by: datetime,
    inward_time: datetime | None,
) -> list[dict]:
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
    origin_codes = resolve_station(str(state["origin"]))
    dest_codes   = resolve_station(str(state["destination"]))
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
            f"I couldn't retrieve live fares right now, but here is the National Rail "
            f"journey planner link for your trip, prices and booking are available there:\n{outbound}"
        )
        if inward_time:
            inbound = _booking_link(dest_codes[0], origin_codes[0], inward_time, is_return=True)
            msg += f"\n\nReturn journey:\n{inbound}"
        state.update(default_ticket_state())
        resp = {"kind": "ticket_search", "done": True, "message": msg, "journeys": []}
        if debug:
            resp["ticket_debug"] = dbg
        return resp

    # Step 6 - sort by price, take top 5, format output
    top5 = sorted(all_journeys, key=lambda j: j["pence"])[:5]

    lines = [
        f"{i}. {j['origin']} → {j['destination']} | dep {j['departure']} arr {j['arrival']} | from {j['price']}\n"
        f"   Book: {j['link']}"
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
        "journeys": [{k: v for k, v in j.items() if k != "pence"} for j in top5],
    }
    if debug:
        dbg["step"] = "success"
        resp["ticket_debug"] = dbg
    return resp
