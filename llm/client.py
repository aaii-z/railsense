import json
import os
import re

import ollama

MODEL = os.environ.get("LLM_MODEL", "gemma3")


def chat_json(messages: list[dict], expected_keys: list[str], return_debug: bool = False):
    """Call the LLM and parse a JSON response. Retries once with an explicit reminder."""
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
