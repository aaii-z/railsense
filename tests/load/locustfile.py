"""
Locust load test for RailSense core functions.

Run (headless, 20 users, 60 seconds):
    locust -f tests/load/locustfile.py --headless -u 20 -r 2 -t 60s --no-web

Run (with browser dashboard):
    locust -f tests/load/locustfile.py
"""

import sys
import time
import random
from pathlib import Path

from locust import User, task, between, events

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── fixtures ──────────────────────────────────────────────────────────────────

_STATION_QUERIES = [
    # exact city / common names
    "Manchester", "London", "Birmingham", "Edinburgh", "Bristol",
    # with typos (fuzzy path)
    "Manchestr", "Birminghm", "Edinbrgh", "Bristl",
    # full station names
    "London Waterloo", "Manchester Piccadilly", "Birmingham New Street",
    # CRS codes (should pass through if handled)
    "WAT", "MAN", "BHM",
]

_DELAY_SCENARIOS = [
    dict(station="SOU", destination="WAT",
         current_delay_minutes=15, planned_arrival_time="14:30"),
    dict(station="BMH", destination="WAT",
         current_delay_minutes=8,  planned_arrival_time="08:15"),
    dict(station="WOK", destination="WAT",
         current_delay_minutes=22, planned_arrival_time="17:45"),
    dict(station="WIN", destination="WAT",
         current_delay_minutes=3,  planned_arrival_time="12:00"),
    dict(station="BSK", destination="WAT",
         current_delay_minutes=30, planned_arrival_time="18:30"),
    dict(station="CLJ", destination="WAT",
         current_delay_minutes=5,  planned_arrival_time="09:10"),
    dict(station="POO", destination="WAT",
         current_delay_minutes=12, planned_arrival_time="07:45"),
]

_MODEL_AVAILABLE = (_ROOT / "models" / "task2" / "random_forest.pkl").is_file()

def _db_available() -> bool:
    try:
        from db import get_conn, put_conn
        conn = get_conn()
        put_conn(conn)
        return True
    except Exception:
        return False

_DB_AVAILABLE = _db_available()

_CONTINGENCY_QUERIES = [
    "What is the evacuation procedure for platform 3?",
    "A passenger has collapsed, what should I do?",
    "How do I handle a points failure at the station?",
    "Who do I call when there is a signal failure?",
    "Crowd management steps during major disruption",
    "What are the steps when a driver reports a fault?",
    "Emergency procedure for a fire alarm at the station",
]


# ── helper: fire a timed event so Locust tracks it ───────────────────────────

def _timed(env, name: str, func, *args, **kwargs):
    """Call func(*args, **kwargs) and report timing + pass/fail to Locust."""
    start = time.perf_counter()
    exc = None
    try:
        return func(*args, **kwargs)
    except Exception as e:
        exc = e
        raise
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        env.events.request.fire(
            request_type="func",
            name=name,
            response_time=elapsed_ms,
            response_length=0,
            exception=exc,
        )


# ── User class ────────────────────────────────────────────────────────────────

class RailSenseUser(User):
    """Simulates a single concurrent RailSense user."""

    wait_time = between(0.5, 2.0)

    def on_start(self):
        from tasks.task1.stations import resolve_station
        resolve_station("london")

        if _MODEL_AVAILABLE:
            from tasks.task2.predictor import _load_assets
            _load_assets()

        from tasks.task3.retriever import _get_embedder
        _get_embedder()

    # ── Task 1: station name resolution ───────────────────────────────────────

    @task(4)
    def task_resolve_station_exact(self):
        """Exact city name lookup  fast path."""
        from tasks.task1.stations import resolve_station
        query = random.choice(["london", "manchester", "birmingham", "edinburgh"])
        _timed(self.environment, "resolve_station/exact", resolve_station, query)

    @task(2)
    def task_resolve_station_fuzzy(self):
        """Fuzzy/misspelled station lookup  slow path (rapidfuzz scan)."""
        from tasks.task1.stations import resolve_station, StationAmbiguous
        query = random.choice(["manchestr", "londn watarloo", "birminghm"])

        def _safe_resolve(q):
            try:
                return resolve_station(q)
            except StationAmbiguous as e:
                return e.candidates  # ambiguous is fine

        _timed(self.environment, "resolve_station/fuzzy", _safe_resolve, query)

    # ── Task 2: ML delay prediction ───────────────────────────────────────────

    @task(3)
    def task_predict_delay(self):
        """Single ML inference call  the core delay prediction hot path."""
        if not _MODEL_AVAILABLE:
            return  # skip silently if model not yet trained

        from tasks.task2.predictor import predict_delay_minutes
        scenario = random.choice(_DELAY_SCENARIOS)
        _timed(
            self.environment,
            "predict_delay_minutes",
            predict_delay_minutes,
            **scenario,
        )

    # ── Task 3: RAG embedding (runs without DB or LLM) ────────────────────────

    @task(3)
    def task_embed_query(self):
        """Encode a staff contingency query  most expensive pure-compute step in Task 3."""
        from tasks.task3.retriever import _get_embedder
        query = random.choice(_CONTINGENCY_QUERIES)
        _timed(
            self.environment,
            "task3/embed_query",
            _get_embedder().encode,
            query,
        )

    @task(1)
    def task_vector_db_retrieve(self):
        """pgvector similarity search  skipped when DB is not reachable."""
        if not _DB_AVAILABLE:
            return
        from tasks.task3.retriever import retrieve
        query = random.choice(_CONTINGENCY_QUERIES)
        _timed(
            self.environment,
            "task3/vector_db_retrieve",
            retrieve,
            query,
            top_k=5,
        )

    # ── Task 1: ticket pipeline pure-compute steps ────────────────────────────

    @task(2)
    def task_booking_link(self):
        """Build a National Rail booking URL  pure string formatting."""
        from datetime import datetime
        from zoneinfo import ZoneInfo
        from tasks.task1.ticket_finder import _booking_link
        hour = random.randint(6, 22)
        minute = random.choice([0, 15, 30, 45])
        dt = datetime(2026, 6, 15, hour, minute, tzinfo=ZoneInfo("Europe/London"))
        _timed(
            self.environment,
            "task1/booking_link",
            _booking_link,
            "MAN", "EUS", dt,
        )

    @task(2)
    def task_parse_time(self):
        """Parse an ISO 8601 datetime string coming from the LLM."""
        from tasks.task1.ticket_finder import _parse_time
        times = [
            "2026-06-15T09:30:00",
            "2026-06-15T00:00:00",   # midnight → defaults to 09:00
            "2026-06-16T17:45:00",
            "not-a-date",            # bad input → graceful fallback
        ]
        _timed(
            self.environment,
            "task1/parse_time",
            _parse_time,
            random.choice(times),
        )

    # ── Combined: realistic station + prediction (Task 1+2 flow) ─────────────

    @task(1)
    def task_station_then_predict(self):
        """
        Realistic user journey:
        1. Resolve a station name (what the chatbot does on input)
        2. Run delay prediction with that station
        """
        if not _MODEL_AVAILABLE:
            return

        from tasks.task1.stations import resolve_station, StationAmbiguous
        from tasks.task2.predictor import predict_delay_minutes

        # Step 1  station resolution
        try:
            codes = _timed(
                self.environment,
                "resolve_station/exact",
                resolve_station,
                "southampton",
            )
        except StationAmbiguous as e:
            codes = ["SOU"]

        if not codes:
            codes = ["SOU"]

        # Step 2  prediction using resolved code
        station_code = codes[0] if codes else "SOU"
        _timed(
            self.environment,
            "predict_delay_minutes",
            predict_delay_minutes,
            station=station_code,
            destination="WAT",
            current_delay_minutes=random.randint(1, 30),
            planned_arrival_time=f"{random.randint(7, 22):02d}:{random.choice(['00','15','30','45'])}",
        )
