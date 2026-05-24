import pytest
from tasks.task1.stations import resolve_station, is_london, london_terminal_for, StationAmbiguous


def test_exact_city_match():
    codes = resolve_station("london")
    assert "WAT" in codes or "VIC" in codes

def test_exact_station_name():
    codes = resolve_station("waterloo")
    assert "WAT" in codes

def test_typo_resolves():
    codes = resolve_station("londn")
    assert len(codes) > 0

def test_unknown_returns_empty():
    assert resolve_station("zzzznotastation") == []

def test_ambiguous_raises():
    with pytest.raises(StationAmbiguous) as exc:
        resolve_station("sout")  # matches Southampton, Southall, etc.
    assert len(exc.value.candidates) > 0

def test_is_london_true():
    assert is_london(["WAT"]) is True
    assert is_london(["VIC", "WAT"]) is True

def test_is_london_false():
    assert is_london(["SOU"]) is False
    assert is_london([]) is False

def test_london_terminal_for_known():
    assert london_terminal_for("SOU") == "WAT"
    assert london_terminal_for("LIV") == "EUS"
    assert london_terminal_for("EDB") == "KGX"

def test_london_terminal_for_unknown():
    assert london_terminal_for("XYZ") == "EUS"
