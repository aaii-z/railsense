import numpy as np
import pandas as pd
import pytest

from tasks.task2.preprocessing import parse_time_to_minutes, fix_midnight_wraparound, clean_data


# --- parse_time_to_minutes ---

def test_parse_hhmm():
    assert parse_time_to_minutes("09:30") == 570

def test_parse_hhmmss():
    assert parse_time_to_minutes("09:30:00") == 570

def test_parse_midnight():
    assert parse_time_to_minutes("00:00") == 0

def test_parse_nan_returns_nan():
    assert np.isnan(parse_time_to_minutes(np.nan))

def test_parse_empty_returns_nan():
    assert np.isnan(parse_time_to_minutes(""))

def test_parse_invalid_returns_nan():
    assert np.isnan(parse_time_to_minutes("bad"))


# --- fix_midnight_wraparound ---

def test_no_wrap():
    assert fix_midnight_wraparound(5.0) == 5.0

def test_negative_wrap():
    assert fix_midnight_wraparound(-300.0) == -300.0 + 1440

def test_positive_wrap():
    assert fix_midnight_wraparound(1100.0) == 1100.0 - 1440

def test_nan_passthrough():
    assert np.isnan(fix_midnight_wraparound(np.nan))


# --- clean_data ---

def _make_df(rows):
    return pd.DataFrame(rows, columns=["rid", "location", "actual_arrival_time", "planned_arrival_time"])

def test_removes_cancelled_trains():
    df = _make_df([
        ("R1", "A", None, "09:00"),
        ("R1", "B", None, "09:10"),
        ("R2", "A", "09:05", "09:00"),
    ])
    result = clean_data(df)
    assert "R1" not in result["rid"].values
    assert "R2" in result["rid"].values

def test_keeps_actual_over_duplicate():
    df = _make_df([
        ("R1", "A", "09:05", "09:00"),
        ("R1", "A", None,    "09:00"),
    ])
    result = clean_data(df)
    assert len(result) == 1
    assert result.iloc[0]["actual_arrival_time"] == "09:05"
