import json
import os
import re

from openai import OpenAI

MODEL = os.environ.get("LLM_MODEL", "gemma3")

_LOCAL_BASE_URL = "http://localhost:11434/v1"


def _get_client() -> OpenAI:
    api_key = os.environ.get("LLM_API_KEY", "")
    if api_key:
        base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
        return OpenAI(api_key=api_key, base_url=base_url)
    base_url = os.environ.get("LLM_BASE_URL", _LOCAL_BASE_URL)
    return OpenAI(api_key="ollama", base_url=base_url)


def chat_text(messages: list[dict]) -> str:
    """Send a chat request and return the assistant reply as plain text."""
    client = _get_client()
    resp = client.chat.completions.create(model=MODEL, messages=messages)
    return resp.choices[0].message.content


def chat_json(messages: list[dict], expected_keys: list[str], return_debug: bool = False):
    """Call the LLM and parse a JSON response. Retries once with an explicit reminder."""
    last_content = ""
    for attempt in range(2):
        last_content = chat_text(messages)
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
