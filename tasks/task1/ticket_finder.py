from datetime import datetime, timedelta
from typing import Any
from itertools import product
from zoneinfo import ZoneInfo
from zeep.helpers import serialize_object

from rtjp.client import create_client
from llm.client import chat_json
from tasks.task1.stations import (
    StationAmbiguous,
    CANONICAL_FOR_BOOKING,
    resolve_station,
    london_terminal_for,
    is_london,
)



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
    now = _now_uk()
    upcoming = ", ".join(
        (now + timedelta(days=i)).strftime("%a %d %b")
        for i in range(1, 15)
    )
    return (
        f"Today is {now.strftime('%A %d %B %Y')}, current time {now.strftime('%H:%M %Z')}. "
        f"Upcoming dates: {upcoming}."
    )



def _extract_ticket_fields(user_input: str, state: dict[str, Any]) -> dict[str, Any]:
    prompt = (
        f"Current date and time: {_now_str()}\n\n"
        "Extract ticket search fields from the user message and reply with a single JSON object.\n"
        "Always copy origin and destination EXACTLY as the user wrote them, even if the word looks unusual or misspelled. Do NOT leave them null just because the name is unfamiliar.\n"
        "departure_time and return_time must be ISO 8601 (e.g. 2026-05-10T09:00:00). "
        "Use the current date above to resolve relative times like 'tomorrow', 'in 2 hours', 'next Friday'. "
        "Treat 'morning' as 09:00, 'afternoon' as 14:00, 'evening' as 18:00 when no specific time is given.\n"
        "FOLLOW-UP RULES (current state may already have values from a previous search):\n"
        "- If origin and/or destination are already in current state and the user does NOT mention new ones, keep them as-is (output null so _update_state leaves them untouched).\n"
        "- If the user says 'return', 'come back', 'going back', 'on the way back', or similar WITHOUT giving a new origin/destination, they want to add a return leg to the SAME journey  set return_time only, leave origin/destination null.\n"
        "- If the user gives a completely new origin or destination, override that field.\n"
        "If origin, destination, or departure_time is genuinely absent from the message AND not already in current state, "
        "put one short follow-up question in next_question; otherwise next_question must be null.\n\n"
        "Example follow-up: state has origin=Liverpool, destination=Southampton, departure_time=2026-05-25T18:00:00\n"
        "User: \"also want a return after tomorrow night\"\n"
        "Output: {\"origin\": null, \"destination\": null, \"departure_time\": null, \"return_time\": \"2026-05-25T23:00:00\", \"next_question\": null}\n\n"
        "Example new search: \"I want to go from Norwich to London tomorrow morning\"\n"
        "Output: {\"origin\": \"Norwich\", \"destination\": \"London\", \"departure_time\": \"2026-05-18T09:00:00\", \"return_time\": null, \"next_question\": null}\n\n"
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


def _is_valid_value(v) -> bool:
    """True if v is a real extracted value (not None / 'null' / empty)."""
    return bool(v) and str(v).lower() not in ("none", "null", "")


def _update_state(state: dict[str, Any], extracted: dict[str, Any]) -> None:
    for key in ("origin", "destination", "departure_time", "return_time"):
        if _is_valid_value(extracted.get(key)):
            state[key] = extracted[key]


def _is_complete(state: dict[str, Any]) -> bool:
    return all(state.get(k) for k in ("origin", "destination", "departure_time"))



def _parse_time(time_str: str | None) -> tuple[datetime | None, bool]:
    """Returns (datetime, assumed). assumed=True if we made up the time."""
    if not time_str:
        return None, False
    s = str(time_str).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_UK_TZ)
        if dt.hour == 0 and dt.minute == 0:
            # midnight is almost certainly "no time given", default to 9am
            dt = dt.replace(hour=9)
            return dt, True
        return dt, False
    except ValueError:
        # bad input, fall back to tomorrow 9am
        fallback = (_now_uk() + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
        return fallback, True



def _booking_link(origin: str, destination: str, depart_by: datetime) -> str:
    """Build a National Rail journey planner URL."""
    rounded_min = (depart_by.minute // 15) * 15
    return (
        "https://www.nationalrail.co.uk/journey-planner/?"
        f"type=single"
        f"&origin={origin}"
        f"&destination={destination}"
        f"&leavingType=departing"
        f"&leavingDate={depart_by.strftime('%d%m%y')}"
        f"&leavingHour={depart_by.strftime('%H')}"
        f"&leavingMin={rounded_min:02d}"
        f"&adults=1&extraTime=0#O"
    )



def _fetch_plan(origin: str, destination: str, depart_by: datetime, inward_time: datetime | None) -> dict:
    """Call RTJP with retry on transient HTML-instead-of-SOAP errors."""
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


def _extract_journeys(plan: dict, key: str = "outwardJourney") -> list[dict]:
    raw = plan.get(key) or []
    if isinstance(raw, dict):
        raw = [raw]
    return [j for j in raw if isinstance(j, dict)]


def _format_journey(journey: dict, depart_by: datetime) -> dict:
    tt    = journey.get("timetable") or {}
    sched = tt.get("scheduled") or {}
    rt    = tt.get("realtime") or {}
    dep   = rt.get("departure") or sched.get("departure")
    arr   = rt.get("arrival")   or sched.get("arrival")
    pence = _min_fare_pence(journey)
    origin      = _station_name(journey.get("origin"))
    destination = _station_name(journey.get("destination"))
    # use main-terminal codes for the booking link
    booking_origin = CANONICAL_FOR_BOOKING.get(origin, origin)
    booking_dest   = CANONICAL_FOR_BOOKING.get(destination, destination)
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
        "link":        _booking_link(booking_origin, booking_dest, link_time),
        "pence":       pence or float("inf"),  # internal: used for sorting; stripped from final response
        "dep_dt":      link_time,              # internal: used for time-window filter; stripped from final response
    }



def _pick_top5(journeys: list[dict], depart_by: datetime) -> list[dict]:
    """Filter to ±3 h window, sort by price, take top 5."""
    window = timedelta(hours=3)
    in_window = [j for j in journeys if abs((j["dep_dt"] - depart_by).total_seconds()) <= window.total_seconds()]
    return sorted(in_window or journeys, key=lambda j: j["pence"])[:5]


def _search_all_pairs(
    origin_codes: list[str],
    dest_codes: list[str],
    depart_by: datetime,
    inward_time: datetime | None,
) -> tuple[list[dict], list[dict], list[str]]:
    """Search every origin×destination CRS pair, collect outward and (if return) inward journeys."""
    is_return   = inward_time is not None
    all_outward = []
    all_inward  = []
    errors = []
    for origin, destination in product(origin_codes, dest_codes):
        try:
            plan = _fetch_plan(origin, destination, depart_by, inward_time)
            out  = _extract_journeys(plan, "outwardJourney")
            all_outward.extend(_format_journey(j, depart_by) for j in out)
            if is_return:
                inn = _extract_journeys(plan, "inwardJourney")
                all_inward.extend(_format_journey(j, inward_time) for j in inn)
        except Exception as e:
            errors.append(f"{origin}→{destination}: {e}")
    return all_outward, all_inward, errors



def _collect_slots(user_input: str, state: dict[str, Any], debug: bool) -> dict[str, Any] | None:
    """Extract fields from the user message. Returns a follow-up response if slots are still
    missing, or None when all required slots are filled and search can proceed."""

    was_complete = _is_complete(state)
    extracted = _extract_ticket_fields(user_input, state)
    _update_state(state, extracted)

    # Still missing required fields — ask a follow-up question
    if not _is_complete(state):
        question = extracted.get("next_question") or "Please tell me your origin, destination, and departure time."
        resp: dict[str, Any] = {"kind": "ticket_search", "done": False, "message": question}
        if debug:
            resp["ticket_debug"] = {"step": "collecting_fields", "state": dict(state)}
        return resp

    # State was already complete and user didn't add anything new — avoid re-searching
    user_added_new_field = any(
        _is_valid_value(extracted.get(k))
        for k in ("origin", "destination", "departure_time", "return_time")
    )
    if was_complete and not user_added_new_field:
        resp = {
            "kind": "ticket_search",
            "done": False,
            "message": (
                "Those are the best fares available for that journey. "
                "Would you like to try a different departure time, date, or route?"
            ),
        }
        if debug:
            resp["ticket_debug"] = {"step": "comment_on_results", "state": dict(state)}
        return resp

    return None  # all slots filled and new data present — proceed to search


def _run_search(state: dict[str, Any], debug: bool) -> dict[str, Any]:
    """Resolve stations, call RTJP, and return formatted results. Assumes all slots are filled."""
    dbg: dict[str, Any] = {}

    # Resolve origin/destination names to CRS codes
    try:
        origin_codes = resolve_station(str(state["origin"]))
    except StationAmbiguous as e:
        state["origin"] = None
        opts = " or ".join(e.candidates)
        resp: dict[str, Any] = {"kind": "ticket_search", "done": False,
                "message": f"Did you mean {opts}? Please clarify your departure station."}
        if debug:
            resp["ticket_debug"] = {"step": "station_ambiguous", "field": "origin", "candidates": e.candidates}
        return resp

    try:
        dest_codes = resolve_station(str(state["destination"]))
    except StationAmbiguous as e:
        state["destination"] = None
        opts = " or ".join(e.candidates)
        resp = {"kind": "ticket_search", "done": False,
                "message": f"Did you mean {opts}? Please clarify your destination station."}
        if debug:
            resp["ticket_debug"] = {"step": "station_ambiguous", "field": "destination", "candidates": e.candidates}
        return resp

    # London terminal disambiguation: pin to the one terminal that serves this route
    if is_london(dest_codes):
        dest_codes = [london_terminal_for(origin_codes[0] if origin_codes else "")]
    if is_london(origin_codes):
        origin_codes = [london_terminal_for(dest_codes[0] if dest_codes else "")]

    dbg.update({
        "origin_input":   state["origin"],
        "origin_crs":     origin_codes,
        "dest_input":     state["destination"],
        "dest_crs":       dest_codes,
        "departure_time": str(state["departure_time"]),
        "return_time":    str(state.get("return_time")),
    })

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

    # Parse times
    depart_by, assumed = _parse_time(str(state["departure_time"]))
    if depart_by is None:
        state["departure_time"] = None
        resp = {"kind": "ticket_search", "done": False, "message": "Please provide a departure time."}
        if debug:
            resp["ticket_debug"] = {**dbg, "step": "time_parse_failed"}
        return resp

    if depart_by < _now_uk():
        state["departure_time"] = None
        resp = {"kind": "ticket_search", "done": False,
                "message": f"The departure time I understood ({depart_by.strftime('%A %d %B at %H:%M')}) is in the past. When would you like to travel?"}
        if debug:
            resp["ticket_debug"] = {**dbg, "step": "departure_in_past", "parsed_as": depart_by.isoformat()}
        return resp

    inward_time, _ = _parse_time(state.get("return_time"))
    dbg["depart_by"]      = depart_by.isoformat()
    dbg["inward_time"]    = inward_time.isoformat() if inward_time else None
    dbg["pairs_searched"] = [f"{o}→{d}" for o in origin_codes for d in dest_codes]

    # Call RTJP for every origin×destination pair
    all_outward, all_inward, search_errors = _search_all_pairs(origin_codes, dest_codes, depart_by, inward_time)
    dbg["journeys_found"] = len(all_outward)
    if search_errors:
        dbg["search_errors"] = search_errors

    is_return = inward_time is not None

    if not all_outward:
        dbg["step"] = "api_fallback"
        outbound = _booking_link(origin_codes[0], dest_codes[0], depart_by)
        msg = (
            f"I couldn't retrieve live fares right now, but you can check prices and book here: "
            f"[National Rail journey planner]({outbound})"
        )
        if inward_time:
            inbound = _booking_link(dest_codes[0], origin_codes[0], inward_time)
            msg += f"\n\nReturn journey (single ticket): [National Rail journey planner]({inbound})"
        state["return_time"] = None
        resp = {"kind": "ticket_search", "done": True, "message": msg, "journeys": []}
        if debug:
            resp["ticket_debug"] = dbg
        return resp

    # Pick top-5 outward journeys (±3 h window, sorted by price)
    top5_out = _pick_top5(all_outward, depart_by)

    lines = ["**Outward journeys (single ticket):**"] + [
        f"{i}. {j['origin']} → {j['destination']} | dep {j['departure']} arr {j['arrival']} | from {j['price']} - [Book ticket]({j['link']})"
        for i, j in enumerate(top5_out, 1)
    ]

    # Pick top-5 return journeys when the user asked for a return
    top5_in: list[dict] = []
    if is_return and all_inward and inward_time:
        top5_in = _pick_top5(all_inward, inward_time)
        lines += ["", "**Return journeys (single ticket):**"] + [
            f"{i}. {j['origin']} → {j['destination']} | dep {j['departure']} arr {j['arrival']} | from {j['price']} - [Book ticket]({j['link']})"
            for i, j in enumerate(top5_in, 1)
        ]

    note   = "Note: departure time was assumed.\n" if assumed else ""
    plural = f"Searched {len(origin_codes)}×{len(dest_codes)} station combinations.\n" if len(origin_codes) + len(dest_codes) > 2 else ""

    state["return_time"] = None

    all_top5 = top5_out + top5_in
    resp = {
        "kind":     "ticket_search",
        "done":     True,
        "message":  note + plural + "\n".join(lines),
        "journeys": [{k: v for k, v in j.items() if k not in ("pence", "dep_dt")} for j in all_top5],
    }
    if debug:
        dbg["step"] = "success"
        resp["ticket_debug"] = dbg
    return resp


def handle_ticket_message(user_input: str, state: dict[str, Any], *, debug: bool = False) -> dict[str, Any]:
    early = _collect_slots(user_input, state, debug)
    if early is not None:
        return early
    return _run_search(state, debug)
