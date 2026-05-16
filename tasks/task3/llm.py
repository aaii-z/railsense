from llm.client import chat_text
from tasks.task3.retriever import retrieve, format_context


def answer_contingency_query(
    user_input: str,
    history: list[dict] | None = None,
    station: str = None,
    return_debug: bool = False,
) -> str | tuple[str, dict]:
    """
    Retrieve relevant chunks from pgvector and answer a contingency query
    for operational staff, grounded in the disruption plan documents.
    """
    chunks  = retrieve(user_input, top_k=5, station=station)
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

    answer = chat_text(messages).strip()

    if return_debug:
        return answer, {
            "chunks_retrieved": len(chunks),
            "top_chunk_score":  chunks[0]["score"] if chunks else None,
            "stations_found":   list({c["station"] for c in chunks}),
        }
    return answer
