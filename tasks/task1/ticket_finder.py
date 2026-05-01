import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from zeep.helpers import serialize_object

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rtjp.client import create_client

# TODO: Add response ID of last response to consequent messages?
# TODO: Are we supposed to travel return tickets? Would need time for departure from destination, can just swap origin/destination
# YES, we need to handle return tickets

# TODO: Use OpenAI spec? Anthropic? Does it matter?
# Yes, we will use OpenAI spec for whole project

# TODO: How/where to host LLM for demo?
# Yes, we will host on our local computer
# We will use gemma-4-e2b-it-gguf model for now

# Collect both date and time
# TODO: How to handle cases such as when user says 'Next Saturday' ?

# TODO: Add validations for arrival and departure time so departure is always after arrival


# LM Studio / vLLM / other OpenAI-compatible local servers:
#   http://localhost:1234/v1/chat/completions
BASE_URL = "http://localhost:1234/v1/"
MODEL = "google/gemma-3-4b"

# ---- JSON Schema (structured output / response_format) ----
TICKET_EXTRACTION_SCHEMA = {
    "name": "ticket_extraction",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "origin": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "description": "Departure city or station, or null if unknown.",
            },
            "destination": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "description": "Arrival city or station, or null if unknown.",
            },
            "departure_time": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "description": "When the user wants to leave (natural language ok), or null.",
            },
            "return_time": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "description": "Return leg time if mentioned, else null.",
            },
            "next_question": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "description": "One short follow-up if required fields are missing, else null.",
            },
        },
        "required": [
            "origin",
            "destination",
            "departure_time",
            "return_time",
            "next_question",
        ],
        "additionalProperties": False,
    },
}

# ---- Conversation State ----
state = {
    "origin": None,
    "destination": None,
    "departure_time": None,
    "return_time": None,
    "is_return_journey": None
}


def _chat_completions_url():
    base = BASE_URL.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


# TODO: Add current time of conversation to either prompt, or current state
#
def build_messages(state, user_input):
    system = """You are a travel assistant for train tickets.

Extract origin, destination, departure_time, and optional return_time from the user message.
Prefer 3-letter National Rail CRS codes for origin and destination when known (e.g. PAD, MAN).
Merge with the current state: keep known values unless the user changes them.
If origin, destination, or departure_time is still missing, set next_question to one clear follow-up; otherwise set next_question to null."""

    user = (
        f"Current state:\n"
        f"  origin: {state['origin']}\n"
        f"  destination: {state['destination']}\n"
        f"  departure_date: {state['departure_date']}\n"
        f"  departure_time: {state['departure_time']}\n"
        f"  return_date: {state['return_date']}\n\n"
        f"  return_time: {state['return_time']}\n\n"
        f'User message: "{user_input}"'
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def call_llm(messages):
    """POST OpenAI-compatible chat/completions with structured JSON output."""
    url = _chat_completions_url()
    headers = {"Content-Type": "application/json"}

    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": TICKET_EXTRACTION_SCHEMA,
        },
    }

    response = requests.post(url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        print("Unexpected response shape:", data)
        raise RuntimeError("LLM response missing choices[0].message.content") from e

    if content is None or (isinstance(content, str) and not content.strip()):
        print("Empty message content:", data)
        return None

    if isinstance(content, str):
        return json.loads(content)

    if isinstance(content, dict):
        return content

    return json.loads(str(content))


def update_state(state, llm_output):
    for key in state.keys():
        if key in llm_output and llm_output.get(key) is not None and llm_output.get(key) != "None":
            state[key] = llm_output[key]
    return state


def is_complete(state):
    return (
        state["origin"] is not None
        and state["destination"] is not None
        and state["departure_time"] is not None
    )


def departure_str_to_datetime(departure_time_str):
    if not departure_time_str:
        return None
    s = str(departure_time_str).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        pass
    dt = datetime.now(timezone.utc) + timedelta(days=1)
    return dt.replace(hour=9, minute=0, second=0, microsecond=0)


def realtime_journey_plan_fares(
    origin_crs,
    destination_crs,
    depart_by,
    adults=1,
    children=0,
    fare_class="STANDARD",
    realtime_enquiry="STANDARD",
    direct_trains=False,
    inward_time=None,
    client=None,
):
    origin_crs = origin_crs.strip().upper()
    destination_crs = destination_crs.strip().upper()

    if not isinstance(depart_by, datetime):
        raise TypeError("depart_by must be a datetime")

    if client is None:
        client = create_client()

    outward_time = {"departBy": depart_by}

    kwargs = {
        "origin": {"stationCRS": origin_crs},
        "destination": {"stationCRS": destination_crs},
        "realtimeEnquiry": realtime_enquiry,
        "outwardTime": outward_time,
        "via": [],
        "notVia": [],
        "directTrains": direct_trains,
        "fareRequestDetails": {
            "passengers": {"adult": adults, "child": children},
            "fareClass": fare_class,
        },
    }

    if inward_time is not None:
        if not isinstance(inward_time, datetime):
            raise TypeError("inward_time must be a datetime or None")
        kwargs["inwardTime"] = {"departBy": inward_time}

    response = client.service.RealtimeJourneyPlan(**kwargs)
    return serialize_object(response)


def _fare_list(journey):
    fares = journey.get("fare")
    if fares is None:
        return []
    if isinstance(fares, dict):
        return [fares]
    if isinstance(fares, list):
        return [f for f in fares if isinstance(f, dict)]
    return []


def _min_fare_pence(journey):
    prices = []
    for f in _fare_list(journey):
        tp = f.get("totalPrice")
        if tp is not None:
            try:
                prices.append(int(tp))
            except (TypeError, ValueError):
                pass
    return min(prices) if prices else None


def _cheapest_fare(journey):
    best = None
    best_p = None
    for f in _fare_list(journey):
        tp = f.get("totalPrice")
        if tp is None:
            continue
        try:
            p = int(tp)
        except (TypeError, ValueError):
            continue
        if best is None or p < best_p:
            best = f
            best_p = p
    return best


def _scalar_station(value):
    if value is None:
        return "?"
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for k in ("_value_1", "stationCRS", "stationCrs"):
            if k in value and value[k] is not None:
                return str(value[k])
        return str(next(iter(value.values()), "?"))
    return str(value)


def _journey_end_times(journey):
    tt = journey.get("timetable") or {}
    sched = (tt.get("scheduled") or {}) if isinstance(tt, dict) else {}
    rt = (tt.get("realtime") or {}) if isinstance(tt, dict) else {}
    dep = rt.get("departure") or sched.get("departure")
    arr = rt.get("arrival") or sched.get("arrival")
    return dep, arr


def _format_time(value):
    if value is None:
        return "—"
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d %H:%M")
    s = str(value)
    if "T" in s:
        s = s.replace("T", " ")
    return s[:16]


def _pence_to_gbp(pence):
    if pence is None:
        return "—"
    return f"£{int(pence) / 100:.2f}"


def outward_journeys_from_plan(plan):
    if not isinstance(plan, dict):
        return []
    raw = plan.get("outwardJourney")
    if raw is None:
        return []
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, list):
        return [j for j in raw if isinstance(j, dict)]
    return []


def print_journeys_sorted_by_fare(plan):
    if not isinstance(plan, dict):
        print("No journey data.")
        return
    status = plan.get("response")
    if status is not None and str(status) != "Ok":
        details = plan.get("responseDetails") or ""
        print(f"Journey plan response: {status} {details}".strip())
        return
    journeys = outward_journeys_from_plan(plan)
    if not journeys:
        print("No outward journeys in this response.")
        return
    ranked = []
    for j in journeys:
        pence = _min_fare_pence(j)
        ranked.append((float("inf") if pence is None else pence, j))
    ranked.sort(key=lambda x: x[0])
    print("\nOutward journeys (lowest fare first):\n")
    for i, (pence, j) in enumerate(ranked, 1):
        origin = _scalar_station(j.get("origin"))
        dest = _scalar_station(j.get("destination"))
        dep, arr = _journey_end_times(j)
        legs = j.get("leg")
        if isinstance(legs, list):
            n_legs = len(legs)
        elif isinstance(legs, dict):
            n_legs = 1
        else:
            n_legs = 0
        fare = _cheapest_fare(j)
        price_str = _pence_to_gbp(pence if pence != float("inf") else None)
        line = (
            f"{i}. {origin} → {dest} | "
            f"dep {_format_time(dep)} | arr {_format_time(arr)} | "
            f"from {price_str}"
        )
        if n_legs:
            line += f" | {n_legs} leg(s)"
        if fare:
            desc = fare.get("description") or ""
            fclass = fare.get("fareClass") or ""
            cat = fare.get("fareCategory") or ""
            bits = [b for b in (fclass, cat, desc[:60] if desc else "") if b]
            if bits:
                line += " | " + " · ".join(bits)
        print(line)


def run_chatbot():
    print("Train ticket assistant started (type 'quit' to exit)\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "quit":
            break

        messages = build_messages(state, user_input)
        try:
            llm_output = call_llm(messages)
        except requests.HTTPError as e:
            print(f"HTTP error from LLM: {e}")
            if e.response is not None:
                try:
                    print(e.response.text)
                except Exception:
                    pass
            print(
                "Tip: If your server does not support response_format json_schema, "
                "try a newer LM Studio / vLLM build or point BASE_URL at your local API."
            )
            continue
        except (json.JSONDecodeError, RuntimeError) as e:
            print(f"Failed to parse LLM output: {e}")
            continue

        if llm_output is None:
            print("Bot: Sorry, I didn't understand that. Could you rephrase?")
            continue

        print(llm_output)
        update_state(state, llm_output)

        if is_complete(state):
            print("\nAll information collected:")
            print(json.dumps(state, indent=2))
            origin = str(state["origin"]).strip().upper()
            destination = str(state["destination"]).strip().upper()
            if len(origin) != 3 or len(destination) != 3:
                print(
                    "\nSOAP journey planning expects 3-letter CRS codes for origin and destination."
                )
            else:
                depart_by = departure_str_to_datetime(state["departure_time"])
                try:
                    plan = realtime_journey_plan_fares(
                        origin,
                        destination,
                        depart_by,
                    )
                    print_journeys_sorted_by_fare(plan)
                except Exception as e:
                    print(f"\nCould not fetch journey plan: {e}")
            break

        next_q = llm_output.get("next_question")

        if next_q:
            print(f"Bot: {next_q}")
        else:
            print("Bot: Could you provide more details?")


if __name__ == "__main__":
    run_chatbot()
