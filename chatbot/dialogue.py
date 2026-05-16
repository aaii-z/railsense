from typing import Any
import os
import re
from datetime import datetime, timezone

import requests
from zeep.exceptions import Error as ZeepError

from llm.client import chat_text
from tasks.task1.ticket_finder import default_ticket_state, handle_ticket_message
from tasks.task2.predictor import default_delay_state, handle_delay_message
from tasks.task3.llm import answer_general_query

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


def init_dialogue_state() -> dict[str, Any]:
    return {
        "active_task": None,
        "ticket_state": default_ticket_state(),
        "delay_state": default_delay_state(),
        "history": [],
    }


def _detect_intent(user_input: str, history: list, active_task: str | None = None) -> str:
    """Classify intent: keyword fast-path first, then LLM for ambiguous cases."""
    # Fast path always runs — strong keyword signals switch intent even mid-task
    if _DELAY_RE.search(user_input):
        return "delay_prediction"
    if _TICKET_RE.search(user_input):
        return "ticket_search"

    recent = history[-4:] if history else []
    ongoing = (
        f"\nThe user is currently mid-conversation on '{active_task}'. "
        "Only switch to a different intent if they explicitly want to change topic.\n"
    ) if active_task else ""
    prompt = f"""You are a railway assistant chatbot with three capabilities:
1. ticket_search, help a user find and book the cheapest train ticket (journey planning, fares, travel dates)
2. delay_prediction, predict how late a delayed train will arrive at its destination
3. general, answer staff/passenger questions about contingency plans, policies, or station information

Classify this message as exactly one of: ticket_search, delay_prediction, general.
{ongoing}
Recent conversation: {recent}
User message: "{user_input}"

Reply with exactly one word only: ticket_search, delay_prediction, or general."""

    try:
        intent = chat_text([{"role": "user", "content": prompt}]).strip().lower()
        if intent in {"delay_prediction", "ticket_search", "general"}:
            return intent
        return "general"
    except Exception:
        return "general"


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
    else:
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
    return block


def handle_message(user_input: str, state: dict[str, Any]) -> dict[str, Any]:
    intent = _detect_intent(user_input, state["history"], state.get("active_task"))
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
            general_result = answer_general_query(
                user_input, history=state["history"], return_debug=debug_mode
            )
            answer = general_result[0] if debug_mode else general_result
            response = {"kind": "general", "done": True, "message": answer}

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
    return response
