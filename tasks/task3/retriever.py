from sentence_transformers import SentenceTransformer

from db import db_cursor

_embedder: SentenceTransformer | None = None

_MODEL = "multi-qa-MiniLM-L6-cos-v1"


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(_MODEL)
    return _embedder


MIN_SCORE = 0.35


def retrieve(
    query: str,
    top_k: int = 5,
    station: str | None = None,
    min_score: float = MIN_SCORE,
) -> list[dict]:
    """Return the top-k most relevant chunks, dropping any below min_score."""
    embedding = _get_embedder().encode(query).tolist()

    # Optionally narrow results to a specific station
    station_filter = "AND station ILIKE %s" if station else ""
    station_param  = [f"%{station}%"] if station else []

    with db_cursor() as cur:
        cur.execute(
            f"""
            SELECT station, region, doc_date, section, chunk_text, score
            FROM (
                SELECT DISTINCT ON (chunk_text)
                       station, region, doc_date, section, chunk_text,
                       1 - (embedding <=> %s::vector) AS score
                FROM documents
                WHERE section != 'Table'
                {station_filter}
                ORDER BY chunk_text, embedding <=> %s::vector
            ) sub
            ORDER BY score DESC
            LIMIT %s
            """,
            (embedding, *station_param, embedding, top_k),
        )
        rows = cur.fetchall()

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
        if float(row[5]) >= min_score
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
