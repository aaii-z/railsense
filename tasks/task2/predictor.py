import math
import pickle
import csv
import re
from difflib import get_close_matches
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo
import traceback

import numpy as np

from tasks.task2.preprocessing import ROUTE, ROUTE_WEY2WAT, ROUTE_WAT2WEY, parse_time_to_minutes
from llm.client import chat_json

_UK_TZ = ZoneInfo("Europe/London")


TARGET_COLS = [
    "station",
    "destination",
    "current_delay_minutes",
    "planned_arrival_time",
    "journey_date",
    "next_question",
]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODEL_PATH = _REPO_ROOT / "models" / "task2" / "random_forest.pkl"
_ENCODER_PATH = _REPO_ROOT / "models" / "task2" / "station_encoder.pkl"

_MODEL = None
_ENCODER = None
_ROUTE_NAME_LOOKUP = None
_ROUTE_ALIAS_OVERRIDES = {
    "waterloo": "WAT",
    "waterloo london": "WAT",
    "london waterloo": "WAT",
    "london waterloo east": "WAT",
    "southampton": "SOU",
}


def default_delay_state() -> dict[str, Any]:
    return {
        "station": None,
        "destination": None,
        "current_delay_minutes": None,
        "planned_arrival_time": None,
        "journey_date": None,
    }


def _load_assets():
    global _MODEL, _ENCODER
    if _MODEL is None:
        if not _MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Model file not found at {_MODEL_PATH}. Train Task 2 model first."
            )
        with _MODEL_PATH.open("rb") as f:
            _MODEL = pickle.load(f)
    if _ENCODER is None:
        if not _ENCODER_PATH.is_file():
            raise FileNotFoundError(
                f"Encoder file not found at {_ENCODER_PATH}. Train Task 2 model first."
            )
        with _ENCODER_PATH.open("rb") as f:
            _ENCODER = pickle.load(f)
    return _MODEL, _ENCODER


def _peak_hour(mins: int) -> int:
    return int((420 <= mins <= 570) or (1020 <= mins <= 1170))


def _date_parts(journey_date: str | None) -> tuple[int, int]:
    if journey_date:
        dt = datetime.fromisoformat(journey_date)
    else:
        dt = datetime.now(_UK_TZ)
    return dt.weekday(), dt.isocalendar().week


def _normalise_station_code(value: str) -> str:
    raw = str(value).strip()
    if not raw:
        raise ValueError("Station cannot be empty.")

    code = raw.upper()
    if code not in ROUTE:
        mapped = _map_station_name_to_code(raw)
        if mapped:
            return mapped
        raise ValueError(
            f"{raw!r} is not part of the trained route: {', '.join(ROUTE)}."
        )
    return code


def _normalise_station_text(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    text = re.sub(r"\bstn\b|\bstation\b", "", text).strip()
    return re.sub(r"\s+", " ", text)


def _load_route_name_lookup() -> dict[str, list[str]]:
    global _ROUTE_NAME_LOOKUP
    if _ROUTE_NAME_LOOKUP is not None:
        return _ROUTE_NAME_LOOKUP

    lookup: dict[str, list[str]] = {}
    csv_path = _REPO_ROOT / "data" / "station_cities.csv"
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = str(row.get("crsCode", "")).strip().upper()
            if code not in ROUTE:
                continue
            for candidate in (row.get("stationName", ""), row.get("city", "")):
                key = _normalise_station_text(str(candidate))
                if key:
                    lookup.setdefault(key, [])
                    if code not in lookup[key]:
                        lookup[key].append(code)

    _ROUTE_NAME_LOOKUP = lookup
    return lookup


def _map_station_name_to_code(value: str) -> str | None:
    lookup = _load_route_name_lookup()
    key = _normalise_station_text(value)
    if not key:
        return None
    if key in _ROUTE_ALIAS_OVERRIDES:
        return _ROUTE_ALIAS_OVERRIDES[key]
    if key in lookup:
        return _choose_preferred_code(lookup[key])

    match = get_close_matches(key, lookup.keys(), n=1, cutoff=0.86)
    if match:
        return _choose_preferred_code(lookup[match[0]])
    return None


def _choose_preferred_code(codes: list[str]) -> str:
    if len(codes) == 1:
        return codes[0]
    route_order = {code: idx for idx, code in enumerate(ROUTE)}
    return sorted(codes, key=lambda code: route_order.get(code, 999))[0]


def _stops_remaining(station_code: str, destination_code: str) -> int:
    """Number of route stops between current station and destination."""
    for route in (ROUTE_WEY2WAT, ROUTE_WAT2WEY):
        if station_code in route and destination_code in route:
            si = route.index(station_code)
            di = route.index(destination_code)
            if di > si:
                return di - si
    return 0


def predict_delay_minutes(
    *,
    station: str,
    destination: str,
    current_delay_minutes: float,
    planned_arrival_time: str,
    departure_delay_minutes: float = 0.0,
    has_delay_reason: int = 0,
    journey_date: str | None = None,
) -> float:
    model, encoder = _load_assets()

    station_code = _normalise_station_code(station)
    destination_code = _normalise_station_code(destination)
    planned_arr_mins = parse_time_to_minutes(planned_arrival_time)
    if planned_arr_mins is None or (isinstance(planned_arr_mins, float) and math.isnan(planned_arr_mins)):
        raise ValueError("planned_arrival_time must be in HH:MM format.")

    day_of_week, week_of_year = _date_parts(journey_date)
    stops = _stops_remaining(station_code, destination_code)

    features = np.array(
        [[
            float(encoder.transform([station_code])[0]),
            float(encoder.transform([destination_code])[0]),
            float(current_delay_minutes),
            float(departure_delay_minutes),
            float(planned_arr_mins),
            float(day_of_week),
            float(week_of_year),
            float(_peak_hour(int(planned_arr_mins))),
            float(stops),
            float(has_delay_reason),
        ]]
    )
    return float(model.predict(features)[0])


def _extract_delay_fields(
    user_input: str, state: dict[str, Any], *, debug: bool = False
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = (
        "Extract delay prediction fields from the user input.\n"
        "Accept station/destination as either station names or 3-letter CRS codes.\n"
        "current_delay_minutes must be numeric when present.\n"
        "planned_arrival_time should be HH:MM when present.\n"
        "journey_date should be YYYY-MM-DD when present.\n"
        "If required fields are missing, return one concise follow-up question in next_question.\n\n"
        f"Current state: {state}\n"
        f'User: "{user_input}"'
    )
    result = chat_json(
        [
            {"role": "system", "content": "You extract structured data for rail delay prediction."},
            {"role": "user", "content": prompt},
        ],
        expected_keys=TARGET_COLS,
        return_debug=debug,
    )
    if debug:
        extracted, llm_debug = result
        return extracted, llm_debug
    return result, {}


def _update_state(state: dict[str, Any], extracted: dict[str, Any]) -> None:
    for key in ("station", "destination", "current_delay_minutes", "planned_arrival_time", "journey_date"):
        value = extracted.get(key)
        if value is not None and value != "None":
            state[key] = value


def _is_complete(state: dict[str, Any]) -> bool:
    return all(
        state.get(key) is not None
        for key in ("station", "destination", "current_delay_minutes", "planned_arrival_time")
    )


def handle_delay_message(
    user_input: str, state: dict[str, Any], *, debug: bool = False
) -> dict[str, Any]:
    try:
        extracted, llm_debug = _extract_delay_fields(user_input, state, debug=debug)
    except Exception as exc:
        response = {
            "kind": "delay_prediction",
            "done": False,
            "message": f"Delay field extraction failed: {exc}",
        }
        if debug:
            response["delay_debug"] = {
                "error_stage": "delay_extraction",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        return response
    _update_state(state, extracted)

    if not _is_complete(state):
        question = extracted.get("next_question") or (
            "Please share current station, destination, current delay minutes, and planned arrival time (HH:MM)."
        )
        response = {
            "kind": "delay_prediction",
            "done": False,
            "message": question,
        }
        if debug:
            response["delay_debug"] = {"delay_extracted": extracted, **llm_debug}
        return response

    try:
        prediction = predict_delay_minutes(
            station=str(state["station"]),
            destination=str(state["destination"]),
            current_delay_minutes=float(state["current_delay_minutes"]),
            planned_arrival_time=str(state["planned_arrival_time"]),
            journey_date=state.get("journey_date"),
        )
    except Exception as exc:
        response = {
            "kind": "delay_prediction",
            "done": False,
            "message": f"Delay prediction failed: {exc}",
        }
        if debug:
            response["delay_debug"] = {
                "delay_extracted": extracted,
                **llm_debug,
                "error_stage": "delay_prediction",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        return response
    response = {
        "kind": "delay_prediction",
        "done": True,
        "message": (
            f"Predicted delay at {str(state['destination']).upper()}: "
            f"**{prediction:.1f} minutes**."
        ),
        "prediction": {
            "station": str(state["station"]).upper(),
            "destination": str(state["destination"]).upper(),
            "current_delay_minutes": float(state["current_delay_minutes"]),
            "planned_arrival_time": str(state["planned_arrival_time"]),
            "journey_date": state.get("journey_date"),
            "predicted_delay_minutes": round(prediction, 2),
        },
    }
    state.update(default_delay_state())
    if debug:
        response["delay_debug"] = {"delay_extracted": extracted, **llm_debug}
    return response
