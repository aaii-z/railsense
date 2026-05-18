from typing import Any
import logging
import os
import re
from datetime import datetime, timezone

import requests
from zeep.exceptions import Error as ZeepError

from llm.client import chat_text
from tasks.task1.ticket_finder import default_ticket_state, handle_ticket_message
from tasks.task2.predictor import default_delay_state, handle_delay_message
from tasks.task3.llm import answer_contingency_query
from db.conversations import save_turn, ensure_table

log = logging.getLogger(__name__)

try:
    ensure_table()
except Exception:
    log.warning("ensure_table() failed at import; conversation history may not persist", exc_info=True)

_TICKET_RE = re.compile(
    r"\b(ticket|tickets|fare|fares|book(?:ing)?|travel(?:ling)?|journey|trip|"
    r"depart(?:ure)?|go(?:ing)?\s+(?:to|from)|from\s+\w[\w\s]+to\s+\w|"
    r"train\s+(?:to|from)|return\s+(?:ticket|journey|trip)|come\s+back|"
    r"cheapest|single|day\s+return)\b",
    re.IGNORECASE,
)
_DELAY_RE = re.compile(
    r"\b(delay(?:ed)?|running\s+late|arrive\s+late|how\s+late|"
    r"behind\s+schedule|\d+\s+minutes?\s+(late|delay)|disruption|"
    r"predict\s+arrival|arrival\s+time)\b",
    re.IGNORECASE,
)


GREETING = (
    "Hello! Welcome to RailSense. I can help you with:\n"
    "- Finding the cheapest train ticket for your journey\n"
    "- Predicting how late a delayed train will arrive\n\n"
    "How can I help you today?"
)

STAFF_GREETING = (
    "Hello! Welcome to the RailSense Staff Portal.\n\n"
    "I can help you with station contingency plans and operational disruption guidance, "
    "including emergency procedures, passenger management, and escalation protocols.\n\n"
    "Which station or situation can I help with?"
)


def init_dialogue_state() -> dict[str, Any]:
    return {
        "active_task": None,
        "ticket_state": default_ticket_state(),
        "delay_state": default_delay_state(),
        "history": [],
    }


def _detect_intent(user_input: str, history: list, active_task: str | None = None) -> str:
    """Pick intent: keyword regex first, then LLM if unclear."""
    delay_match  = bool(_DELAY_RE.search(user_input))
    ticket_match = bool(_TICKET_RE.search(user_input))

    # no active task: trust the regex
    if active_task is None:
        if delay_match:
            return "delay_prediction"
        if ticket_match:
            return "ticket_search"
    else:
        # mid-task: regex can only confirm the current task, not switch it.
        # Words like "single" / "return" are too ambiguous mid-conversation,
        # let the LLM decide switching with the active-task hint.
        if active_task == "delay_prediction" and delay_match and not ticket_match:
            return "delay_prediction"
        if active_task == "ticket_search" and ticket_match and not delay_match:
            return "ticket_search"

    recent = history[-4:] if history else []
    ongoing = (
        f"\nThe user is currently mid-conversation on '{active_task}'. "
        "Only switch to a different intent if they explicitly want to change topic.\n"
    ) if active_task else ""
    prompt = f"""You are a railway assistant chatbot with three capabilities:
1. ticket_search, help a user find and book the cheapest train ticket (journey planning, fares, travel dates)
2. delay_prediction, predict how late a delayed train will arrive at its destination
3. contingency, support operational staff with step-by-step guidance on handling disruptions, emergencies, or station contingency plans

Classify this message as exactly one of: ticket_search, delay_prediction, contingency.
{ongoing}
Recent conversation: {recent}
User message: "{user_input}"

Reply with exactly one word only: ticket_search, delay_prediction, or contingency."""

    try:
        intent = chat_text([{"role": "user", "content": prompt}]).strip().lower()
        if intent in {"delay_prediction", "ticket_search", "contingency"}:
            return intent
    except Exception:
        log.warning("intent classification LLM call failed", exc_info=True)

    # LLM failed or gave a weird reply, stay on the active task rather than
    # silently bouncing to 'contingency' (which is out-of-scope for non-staff).
    return active_task or "contingency"


def _build_debug_block(
    intent: str,
    state: dict[str, Any],
    response: dict[str, Any],
    user_input: str,
    started_at: str,
    ticket_state_snapshot: dict[str, Any],
) -> dict[str, Any]:
    if intent == "ticket_search":
        task_state = {k: v for k, v in ticket_state_snapshot.items() if v is not None}
    elif intent == "delay_prediction":
        task_state = {k: v for k, v in state["delay_state"].items() if v is not None}
    else:  # contingency
        task_state = {}
    block = {
        "intent": intent,
        "active_task": state.get("active_task"),
        "extracted_fields": task_state,
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "user_input": user_input,
    }
    if "ticket_debug" in response:
        block["ticket"] = response.pop("ticket_debug")
    if "delay_debug" in response:
        block["delay"] = response.pop("delay_debug")
    if "contingency_debug" in response:
        block["contingency"] = response.pop("contingency_debug")
    return block


_OUT_OF_SCOPE_MSG = (
    "I'm here to help with train travel. I can search for tickets and fares "
    "or predict how late a delayed train will arrive. How can I help you with that?"
)


def handle_message(
    user_input: str,
    state: dict[str, Any],
    is_staff: bool = False,
    session_id: str | None = None,
) -> dict[str, Any]:
    intent = _detect_intent(user_input, state["history"], state.get("active_task"))

    if is_staff and intent != "contingency":
        out_of_scope = (
            "As a staff member I can only assist with station contingency plans and "
            "operational disruption guidance. Please ask about a station's emergency "
            "procedures, disruption handling, or escalation protocols."
        )
        state["history"].append({"role": "user", "content": user_input})
        state["history"].append({"role": "assistant", "content": out_of_scope})
        if len(state["history"]) > 20:
            state["history"] = state["history"][-20:]
        return {"kind": "out_of_scope", "done": True, "message": out_of_scope}

    if not is_staff and intent not in {"ticket_search", "delay_prediction"}:
        state["history"].append({"role": "user", "content": user_input})
        state["history"].append({"role": "assistant", "content": _OUT_OF_SCOPE_MSG})
        if len(state["history"]) > 20:
            state["history"] = state["history"][-20:]
        return {"kind": "out_of_scope", "done": True, "message": _OUT_OF_SCOPE_MSG}

    debug_mode = os.environ.get("RAILSENSE_DEBUG", "").lower() in {"1", "true", "yes"}
    started_at = datetime.now(timezone.utc).isoformat()
    _ticket_state_snapshot: dict = {}

    try:
        if intent == "ticket_search":
            state["active_task"] = "ticket_search"
            _ticket_state_snapshot = dict(state["ticket_state"])
            response = handle_ticket_message(user_input, state["ticket_state"], debug=debug_mode)

        elif intent == "delay_prediction":
            state["active_task"] = "delay_prediction"
            response = handle_delay_message(user_input, state["delay_state"], debug=debug_mode)

        else:
            state["active_task"] = None
            contingency_result = answer_contingency_query(
                user_input, history=state["history"], return_debug=debug_mode
            )
            if debug_mode:
                answer, contingency_debug = contingency_result
            else:
                answer, contingency_debug = contingency_result, {}
            response = {
                "kind": "contingency",
                "done": True,
                "message": answer,
                "contingency_debug": contingency_debug,
            }

    except requests.RequestException as exc:
        response = {"kind": intent, "done": False,
                    "message": f"I couldn't reach the model or API: {exc}"}

    except (ValueError, FileNotFoundError, KeyError) as exc:
        response = {"kind": intent, "done": False,
                    "message": f"I couldn't process that request: {exc}"}

    except ZeepError as exc:
        response = {"kind": intent, "done": False,
                    "message": f"Rail service error: {exc}"}

    except Exception as exc:
        response = {"kind": intent, "done": False,
                    "message": f"Something went wrong: {exc}"}

    if response.get("done"):
        state["active_task"] = None

    if debug_mode:
        response["debug"] = _build_debug_block(
            intent, state, response, user_input, started_at, _ticket_state_snapshot
        )

    state["history"].append({"role": "user", "content": user_input})
    state["history"].append({"role": "assistant", "content": response["message"]})
    if len(state["history"]) > 20:
        state["history"] = state["history"][-20:]

    if session_id:
        assistant_extras = {
            k: response.get(k)
            for k in ("journeys", "prediction", "debug")
            if response.get(k)
        }
        try:
            save_turn(session_id, "user", user_input, intent)
            save_turn(session_id, "assistant", response["message"], intent, extras=assistant_extras or None)
        except Exception:
            pass

    return response
