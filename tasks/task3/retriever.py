import os
import threading
import psycopg2
import psycopg2.pool
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

DB_URL = os.environ.get("DATABASE_URL", "postgresql://railsense:railsense@localhost:5432/railsense")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

_pool: psycopg2.pool.ThreadedConnectionPool | None = None
_pool_lock = threading.Lock()


def _get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = psycopg2.pool.ThreadedConnectionPool(1, 10, DB_URL)
    return _pool


def retrieve(query: str, top_k: int = 5, station: str = None) -> list[dict]:
    """
    Search pgvector for chunks relevant to the query.
    Optionally filter by station name for more precise results.
    """
    embedding = embedder.encode(query).tolist()

    pool = _get_pool()
    conn = pool.getconn()
    try:
        register_vector(conn)
        cursor = conn.cursor()
        try:
            if station:
                cursor.execute(
                    """
                    SELECT station, region, section, chunk_text,
                           1 - (embedding <=> %s::vector) AS score
                    FROM documents
                    WHERE station ILIKE %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (embedding, f"%{station}%", embedding, top_k)
                )
            else:
                cursor.execute(
                    """
                    SELECT station, region, section, chunk_text,
                           1 - (embedding <=> %s::vector) AS score
                    FROM documents
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (embedding, embedding, top_k)
                )
            rows = cursor.fetchall()
        finally:
            cursor.close()
    finally:
        pool.putconn(conn)

    return [
        {
            "station":    row[0],
            "region":     row[1],
            "section":    row[2],
            "chunk_text": row[3],
            "score":      round(float(row[4]), 4),
        }
        for row in rows
    ]


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a context string for Gemma."""
    parts = []
    for c in chunks:
        header = f"[{c['station']}, {c['section']}]"
        parts.append(f"{header}\n{c['chunk_text']}")
    return "\n\n".join(parts)
