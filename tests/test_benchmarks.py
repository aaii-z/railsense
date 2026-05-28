"""
pytest-benchmark tests for RailSense core components.

Run:
    pytest tests/test_benchmarks.py --benchmark-only
    pytest tests/test_benchmarks.py --benchmark-only --benchmark-histogram
    pytest tests/test_benchmarks.py --benchmark-save=baseline
    pytest tests/test_benchmarks.py --benchmark-compare=baseline
"""

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── helpers ───────────────────────────────────────────────────────────────────

def _model_available() -> bool:
    return (_ROOT / "models" / "task2" / "random_forest.pkl").is_file()


def _db_available() -> bool:
    """True if PostgreSQL + pgvector is reachable."""
    try:
        from db import get_conn, put_conn
        conn = get_conn()
        put_conn(conn)
        return True
    except Exception:
        return False


# ── Station resolver benchmarks ───────────────────────────────────────────────

class TestStationResolver:
    """Benchmarks for tasks/task1/stations.py  resolve_station()."""

    def test_exact_city_match(self, benchmark):
        """Fast path: city is in _CITY_PRIORITY dict  O(1) lookup."""
        from tasks.task1.stations import resolve_station
        result = benchmark(resolve_station, "manchester")
        assert result  # must return at least one CRS code

    def test_exact_station_name(self, benchmark):
        """Fast path: exact station name match in STATION_LOOKUP dict."""
        from tasks.task1.stations import resolve_station
        result = benchmark(resolve_station, "london waterloo")
        assert result

    def test_fuzzy_typo(self, benchmark):
        """Slow path: rapidfuzz scan across the full lookup table."""
        from tasks.task1.stations import resolve_station, StationAmbiguous
        def _call():
            try:
                return resolve_station("manchestr picadily")
            except StationAmbiguous:
                return []   # ambiguous is still a valid (non-crash) outcome
        result = benchmark(_call)
        # result can be [] or a list  just confirm no exception leaks
        assert result is not None

    def test_unknown_station(self, benchmark):
        """Worst case: no match found  rapidfuzz exhausts all candidates."""
        from tasks.task1.stations import resolve_station
        result = benchmark(resolve_station, "zzz_not_a_real_station_xyz")
        assert result == []

    def test_london_terminal_for(self, benchmark):
        """O(1) dict lookup  should be nanoseconds."""
        from tasks.task1.stations import london_terminal_for
        result = benchmark(london_terminal_for, "MAN")
        assert result == "EUS"


# ── ML delay prediction benchmarks ───────────────────────────────────────────

@pytest.mark.skipif(not _model_available(), reason="random_forest.pkl not trained yet")
class TestDelayPredictor:
    """Benchmarks for tasks/task2/predictor.py  predict_delay_minutes()."""

    def test_single_prediction(self, benchmark):
        """Single ML inference call  the main production hot path."""
        from tasks.task2.predictor import predict_delay_minutes
        result = benchmark(
            predict_delay_minutes,
            station="SOU",
            destination="WAT",
            current_delay_minutes=15,
            planned_arrival_time="14:30",
        )
        assert isinstance(result, float)

    def test_peak_hour_prediction(self, benchmark):
        """Inference during peak hours (07:00–09:30)."""
        from tasks.task2.predictor import predict_delay_minutes
        result = benchmark(
            predict_delay_minutes,
            station="BMH",
            destination="WAT",
            current_delay_minutes=8,
            planned_arrival_time="08:15",
        )
        assert isinstance(result, float)

    def test_prediction_with_delay_reason(self, benchmark):
        """Inference with has_delay_reason=1 flag set."""
        from tasks.task2.predictor import predict_delay_minutes
        result = benchmark(
            predict_delay_minutes,
            station="WOK",
            destination="WAT",
            current_delay_minutes=22,
            planned_arrival_time="17:45",
            has_delay_reason=1,
        )
        assert isinstance(result, float)

    def test_batch_100_predictions(self, benchmark):
        """100 sequential predictions  simulates a burst of user requests."""
        from tasks.task2.predictor import predict_delay_minutes

        stations = [
            ("SOU", "WAT", 10, "08:30"),
            ("BMH", "WAT", 5,  "17:00"),
            ("WOK", "WAT", 20, "09:15"),
            ("WIN", "WAT", 3,  "12:00"),
            ("BSK", "WAT", 15, "18:30"),
        ]

        def _run_batch():
            for i in range(100):
                s, d, delay, t = stations[i % len(stations)]
                predict_delay_minutes(
                    station=s,
                    destination=d,
                    current_delay_minutes=delay,
                    planned_arrival_time=t,
                )

        benchmark(_run_batch)


# ── Route helpers benchmarks ──────────────────────────────────────────────────

class TestRouteHelpers:
    """Benchmarks for internal route utilities in predictor.py."""

    def test_stops_remaining(self, benchmark):
        from tasks.task2.predictor import _stops_remaining
        result = benchmark(_stops_remaining, "SOU", "WAT")
        assert result > 0

    def test_normalise_station_text(self, benchmark):
        from tasks.task2.predictor import _normalise_station_text
        result = benchmark(_normalise_station_text, "London Waterloo Station")
        assert result == "london waterloo"

    def test_peak_hour_flag(self, benchmark):
        from tasks.task2.predictor import _peak_hour
        # morning peak: 07:00 = 420 mins
        result = benchmark(_peak_hour, 450)
        assert result == 1


# ── Task 3: RAG / contingency benchmarks ─────────────────────────────────────

class TestTask3RAG:
    """Benchmarks for tasks/task3/  embedding generation and retrieval."""

    def test_embed_short_query(self, benchmark):
        """Encode a short staff question  the hot path on every RAG call."""
        from tasks.task3.retriever import _get_embedder
        embedder = _get_embedder()   # warm up model before timing starts
        result = benchmark(embedder.encode, "What is the evacuation procedure?")
        assert result.shape[0] == 384  # all-MiniLM-L6-v2 output dim

    def test_embed_long_query(self, benchmark):
        """Encode a longer, multi-sentence staff query."""
        from tasks.task3.retriever import _get_embedder
        embedder = _get_embedder()
        query = (
            "There is a major disruption at Waterloo station. "
            "A passenger has collapsed on the platform and the train "
            "is delayed by 45 minutes. What steps should I take to "
            "manage the crowd and coordinate with the emergency services?"
        )
        result = benchmark(embedder.encode, query)
        assert result.shape[0] == 384

    def test_embed_batch_10(self, benchmark):
        """Encode 10 queries at once  tests batch throughput."""
        from tasks.task3.retriever import _get_embedder
        embedder = _get_embedder()
        queries = [
            "Evacuation procedure for platform 3",
            "How to handle a points failure",
            "Passenger assistance during delays",
            "Emergency contact for signal failure",
            "Crowd management during disruption",
            "What to do when a train is cancelled",
            "Escalation protocol for major incident",
            "Handling press enquiries during disruption",
            "Contingency for power failure at station",
            "Steps when a driver reports a fault",
        ]
        result = benchmark(embedder.encode, queries)
        assert result.shape == (10, 384)

    def test_format_context(self, benchmark):
        """Format retrieved chunks into a prompt context string  pure Python."""
        from tasks.task3.retriever import format_context
        chunks = [
            {"station": "Waterloo", "section": "Evacuation",
             "chunk_text": "In the event of evacuation, staff should direct passengers to exits A and B."},
            {"station": "Waterloo", "section": "Emergency Contacts",
             "chunk_text": "Call the control centre on 0800 XXX XXXX. Escalate to duty manager."},
            {"station": "Guildford", "section": "Crowd Management",
             "chunk_text": "Deploy barrier staff to platform 2 when passenger numbers exceed 200."},
        ]
        result = benchmark(format_context, chunks)
        assert "Waterloo" in result

    def test_format_sources(self, benchmark):
        """Deduplicate and format source citations  pure Python."""
        from tasks.task3.retriever import format_sources
        chunks = [
            {"station": "Waterloo", "region": "London", "doc_date": "2024-01-15",
             "section": "Evacuation", "chunk_text": "...", "score": 0.91},
            {"station": "Waterloo", "region": "London", "doc_date": "2024-01-15",
             "section": "Contacts",  "chunk_text": "...", "score": 0.88},
            {"station": "Guildford", "region": "South", "doc_date": "2023-11-01",
             "section": "Crowd",     "chunk_text": "...", "score": 0.75},
        ]
        result = benchmark(format_sources, chunks)
        # Waterloo appears twice in chunks but only once in sources (deduped)
        assert result.count("Waterloo") == 1
        assert result.count("Guildford") == 1

    @pytest.mark.skipif(not _db_available(), reason="PostgreSQL/pgvector not reachable")
    def test_vector_db_retrieval(self, benchmark):
        """Full pgvector similarity search  requires live DB with ingested docs."""
        from tasks.task3.retriever import retrieve
        results = benchmark(retrieve, "evacuation procedure", top_k=5)
        assert isinstance(results, list)

    @pytest.mark.skipif(not _db_available(), reason="PostgreSQL/pgvector not reachable")
    def test_vector_db_retrieval_with_station_filter(self, benchmark):
        """pgvector search with a station filter (WHERE station ILIKE ...)."""
        from tasks.task3.retriever import retrieve
        results = benchmark(retrieve, "crowd management", top_k=5, station="Waterloo")
        assert isinstance(results, list)


# ── Task 1: Ticket pipeline benchmarks ───────────────────────────────────────

class TestTask1TicketPipeline:
    """Benchmarks for the pure-computation parts of tasks/task1/ticket_finder.py."""

    def test_booking_link_builder(self, benchmark):
        """Build a National Rail booking URL  pure string formatting."""
        from datetime import datetime
        from zoneinfo import ZoneInfo
        from tasks.task1.ticket_finder import _booking_link
        dt = datetime(2026, 6, 15, 9, 0, tzinfo=ZoneInfo("Europe/London"))
        result = benchmark(_booking_link, "MAN", "EUS", dt)
        assert result.startswith("https://www.nationalrail.co.uk")
        assert "MAN" in result and "EUS" in result

    def test_parse_time_iso(self, benchmark):
        """Parse a clean ISO 8601 datetime string from the LLM."""
        from tasks.task1.ticket_finder import _parse_time
        result, assumed = benchmark(_parse_time, "2026-06-15T09:30:00")
        assert result is not None
        assert assumed is False

    def test_parse_time_relative_midnight(self, benchmark):
        """Midnight is treated as 'no time given' and defaulted to 09:00."""
        from tasks.task1.ticket_finder import _parse_time
        result, assumed = benchmark(_parse_time, "2026-06-15T00:00:00")
        assert assumed is True
        assert result.hour == 9

    def test_parse_time_invalid(self, benchmark):
        """Garbage input falls back gracefully to tomorrow 09:00."""
        from tasks.task1.ticket_finder import _parse_time
        result, assumed = benchmark(_parse_time, "not-a-date")
        assert result is not None
        assert assumed is True

    def test_min_fare_pence(self, benchmark):
        """Extract cheapest fare from a journey dict  pure dict traversal."""
        from tasks.task1.ticket_finder import _min_fare_pence
        journey = {
            "fare": [
                {"totalPrice": 2450},
                {"totalPrice": 1890},
                {"totalPrice": 3100},
            ]
        }
        result = benchmark(_min_fare_pence, journey)
        assert result == 1890

    def test_pick_top5_sorting(self, benchmark):
        """Sort and filter a list of journeys by price within a ±3h window."""
        from datetime import datetime
        from zoneinfo import ZoneInfo
        from tasks.task1.ticket_finder import _pick_top5, _booking_link

        depart_by = datetime(2026, 6, 15, 9, 0, tzinfo=ZoneInfo("Europe/London"))

        # Build 10 fake journeys covering a 6-hour window
        journeys = []
        for i in range(10):
            dep_dt = datetime(2026, 6, 15, 7 + i, 0, tzinfo=ZoneInfo("Europe/London"))
            journeys.append({
                "origin":      "MAN",
                "destination": "EUS",
                "departure":   f"0{7+i}:00",
                "arrival":     f"{10+i}:00",
                "price":       f"£{20 + i * 3:.2f}",
                "pence":       2000 + i * 300,
                "dep_dt":      dep_dt,
                "link":        _booking_link("MAN", "EUS", dep_dt),
            })

        result = benchmark(_pick_top5, journeys, depart_by)
        assert len(result) <= 5
        # results within window should be sorted cheapest first
        prices = [j["pence"] for j in result]
        assert prices == sorted(prices)
