import json
import re
import os
import ollama
from tasks.task3.retriever import retrieve, format_context


MODEL = os.environ.get("LLM_MODEL", "gemma3")


def chat_json(messages: list[dict], expected_keys: list[str], return_debug: bool = False):
    """Call Gemma and parse JSON response. Used by ticket and delay handlers."""
    last_content = ""
    for attempt in range(2):
        response = ollama.chat(model=MODEL, messages=messages)
        last_content = response["message"]["content"]
        clean = re.sub(r"```(?:json)?|```", "", last_content).strip()
        try:
            parsed = json.loads(clean)
            if return_debug:
                return parsed, {"raw_response": last_content}
            return parsed
        except json.JSONDecodeError:
            if attempt == 0:
                messages = messages + [
                    {"role": "assistant", "content": last_content},
                    {"role": "user", "content": "Reply with valid JSON only, no other text."},
                ]

    parsed = {key: None for key in expected_keys}
    if return_debug:
        return parsed, {"raw_response": last_content, "parse_error": "JSON decode failed after retry"}
    return parsed


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

    response = ollama.chat(model=MODEL, messages=messages)
    answer   = response["message"]["content"].strip()

    if return_debug:
        return answer, {
            "chunks_retrieved": len(chunks),
            "top_chunk_score":  chunks[0]["score"] if chunks else None,
            "stations_found":   list({c["station"] for c in chunks}),
        }
    return answer
