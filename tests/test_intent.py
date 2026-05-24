from unittest.mock import patch

from chatbot.dialogue import _detect_intent


def test_delay_keyword():
    assert _detect_intent("my train is delayed by 10 minutes", []) == "delay_prediction"

def test_ticket_keyword():
    assert _detect_intent("I want to book a ticket from London to Bristol", []) == "ticket_search"

def test_mid_task_delay_stays():
    assert _detect_intent("it's running 5 minutes late", [], active_task="delay_prediction") == "delay_prediction"

def test_mid_task_ticket_stays():
    assert _detect_intent("cheapest single ticket please", [], active_task="ticket_search") == "ticket_search"

def test_ambiguous_uses_llm():
    with patch("chatbot.dialogue.chat_text", return_value="contingency") as mock_llm:
        result = _detect_intent("what should I do at the station?", [])
    mock_llm.assert_called_once()
    assert result == "contingency"
