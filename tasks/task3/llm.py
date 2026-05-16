from llm.client import chat_text
from tasks.task3.retriever import retrieve, format_context


def answer_general_query(
    user_input: str,
    history: list[dict] | None = None,
    station: str = None,
    return_debug: bool = False,
) -> str | tuple[str, dict]:
    """
    Retrieve relevant chunks from pgvector and ask Gemma to answer
    grounded in those documents.
    """
    chunks  = retrieve(user_input, top_k=5, station=station)
    context = format_context(chunks)

    system_prompt = (
        "You are a helpful railway assistant for South Western Railway. "
        "Answer the passenger's question using only the provided context. "
        "If the context doesn't contain enough information, say so honestly. "
        "Be concise and clear."
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
