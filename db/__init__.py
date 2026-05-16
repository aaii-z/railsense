import os
import threading

import psycopg2
import psycopg2.pool
from pgvector.psycopg2 import register_vector

DB_URL = os.environ.get("DATABASE_URL", "postgresql://railsense:railsense@localhost:5432/railsense")

_pool: psycopg2.pool.ThreadedConnectionPool | None = None
_pool_lock = threading.Lock()


def _get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = psycopg2.pool.ThreadedConnectionPool(1, 10, DB_URL)
    return _pool


def get_conn():
    conn = _get_pool().getconn()
    register_vector(conn)
    return conn


def put_conn(conn) -> None:
    _get_pool().putconn(conn)
