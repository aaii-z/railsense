from sentence_transformers import SentenceTransformer

from db import get_conn, put_conn

_embedder: SentenceTransformer | None = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def retrieve(query: str, top_k: int = 5, station: str = None) -> list[dict]:
    embedding = _get_embedder().encode(query).tolist()

    conn = get_conn()
    try:
        cursor = conn.cursor()
        try:
            if station:
                cursor.execute(
                    """
                    SELECT station, region, doc_date, section, chunk_text,
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
                    SELECT station, region, doc_date, section, chunk_text,
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
        put_conn(conn)

    return [
        {
            "station":    row[0],
            "region":     row[1],
            "doc_date":   row[2],
            "section":    row[3],
            "chunk_text": row[4],
            "score":      round(float(row[5]), 4),
        }
        for row in rows
    ]


def format_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        header = f"[{c['station']}, {c['section']}]"
        parts.append(f"{header}\n{c['chunk_text']}")
    return "\n\n".join(parts)


def format_sources(chunks: list[dict]) -> str:
    seen = set()
    sources = []
    for c in chunks:
        key = (c["station"], c["region"], c["doc_date"])
        if key not in seen:
            seen.add(key)
            date = f", {c['doc_date']}" if c["doc_date"] else ""
            sources.append(f"- {c['station']} Station Disruption Plan ({c['region']}{date})")
    return "\n".join(sources)
