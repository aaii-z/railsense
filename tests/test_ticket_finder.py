from datetime import datetime
from zoneinfo import ZoneInfo

from tasks.task1.ticket_finder import _parse_time, _booking_link, _is_complete

_UK = ZoneInfo("Europe/London")


# --- _parse_time ---

def test_parse_normal_time():
    dt, assumed = _parse_time("2026-06-01T14:30:00")
    assert dt.hour == 14 and dt.minute == 30
    assert assumed is False

def test_parse_midnight_defaults_to_9am():
    dt, assumed = _parse_time("2026-06-01T00:00:00")
    assert dt.hour == 9
    assert assumed is True

def test_parse_none_returns_none():
    dt, assumed = _parse_time(None)
    assert dt is None and assumed is False

def test_parse_invalid_falls_back():
    dt, assumed = _parse_time("not-a-date")
    assert dt is not None
    assert assumed is True

def test_parse_naive_gets_uk_tz():
    dt, _ = _parse_time("2026-06-01T10:00:00")
    assert dt.tzinfo is not None


# --- _booking_link ---

def test_booking_link_single():
    dt = datetime(2026, 6, 1, 9, 17, tzinfo=_UK)
    url = _booking_link("SOU", "WAT", dt)
    assert "type=single" in url
    assert "origin=SOU" in url
    assert "destination=WAT" in url
    assert "leavingDate=010626" in url

def test_booking_link_rounds_minutes():
    dt = datetime(2026, 6, 1, 9, 23, tzinfo=_UK)
    url = _booking_link("SOU", "WAT", dt)
    assert "leavingMin=15" in url  # 23 rounds down to 15

def test_booking_link_return():
    dt = datetime(2026, 6, 1, 9, 0, tzinfo=_UK)
    url = _booking_link("WAT", "SOU", dt)
    assert "type=single" in url   # return leg is also booked as a single ticket


# --- _is_complete ---

def test_complete_when_all_fields():
    state = {"origin": "Norwich", "destination": "London", "departure_time": "2026-06-01T09:00"}
    assert _is_complete(state) is True

def test_incomplete_missing_departure():
    state = {"origin": "Norwich", "destination": "London", "departure_time": None}
    assert _is_complete(state) is False

def test_incomplete_missing_origin():
    state = {"origin": None, "destination": "London", "departure_time": "2026-06-01T09:00"}
    assert _is_complete(state) is False
