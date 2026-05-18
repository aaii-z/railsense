import time

from llm.client import chat_text
from tasks.task3.retriever import retrieve, format_context, format_sources


def _extract_station(user_input: str) -> tuple[str | None, float]:
    """Returns (station name or None, time taken)."""
    prompt = (
        "Extract the railway station name from the following message. "
        "Reply with only the station name (e.g. 'Guildford', 'Surbiton'). "
        "If no station is mentioned, reply with exactly: none\n\n"
        f"Message: {user_input}"
    )
    t0 = time.perf_counter()
    try:
        result = chat_text([{"role": "user", "content": prompt}]).strip()
        station = None if (result.lower() == "none" or not result) else result
    except Exception:
        station = None
    return station, round(time.perf_counter() - t0, 2)


def answer_contingency_query(
    user_input: str,
    history: list[dict] | None = None,
    station: str | None = None,
    return_debug: bool = False,
) -> str | tuple[str, dict]:
    """RAG: pull relevant chunks from pgvector and answer the contingency
    question, grounded in the disruption plan docs."""
    timings: dict[str, float] = {}

    station_extraction_secs = 0.0
    if not station:
        station, station_extraction_secs = _extract_station(user_input)
    timings["station_extraction_llm"] = station_extraction_secs

    t0 = time.perf_counter()
    chunks = retrieve(user_input, top_k=5, station=station)
    timings["vector_db_retrieval"] = round(time.perf_counter() - t0, 2)

    context = format_context(chunks)

    system_prompt = (
        "You are an expert railway operations assistant for South Western Railway, "
        "supporting operational staff such as station managers and duty managers. "
        "Using only the provided disruption plan context, give clear step-by-step "
        "guidance on how to handle the reported contingency. "
        "If the context does not cover the situation, say so and advise the staff "
        "member to escalate to the relevant control centre. "
        "Be concise, authoritative, and action-oriented."
    )

    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history[-6:])
    messages.append({
        "role": "user",
        "content": f"Context from disruption plans:\n{context}\n\nQuestion: {user_input}"
    })

    t0 = time.perf_counter()
    answer = chat_text(messages).strip()
    timings["answer_llm"] = round(time.perf_counter() - t0, 2)

    timings["total"] = round(sum(timings.values()), 2)

    if chunks:
        answer += f"\n\n---\n**Sources:**\n{format_sources(chunks)}"

    if return_debug:
        return answer, {
            "chunks_retrieved": len(chunks),
            "top_chunk_score": chunks[0]["score"] if chunks else None,
            "stations_found": list({c["station"] for c in chunks}),
            "timings_secs": timings,
        }
    return answer
