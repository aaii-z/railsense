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


def chat_text(messages: list[dict], *, temperature: float = 0.2) -> str:
    """Plain text chat call."""
    client = _get_client()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def _chat_json_raw(messages: list[dict]) -> str:
    """Chat call asking for JSON mode, falls back if the provider doesn't support it."""
    client = _get_client()
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
        )
    except Exception:
        # some older Ollama / providers reject response_format, retry without it
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0,
        )
    return resp.choices[0].message.content or ""


def chat_json(messages: list[dict], expected_keys: list[str], return_debug: bool = False):
    """Call the LLM and parse JSON out of the reply. Retries 3 times."""
    last_content = ""
    attempts: list[str] = []
    for attempt in range(3):
        last_content = _chat_json_raw(messages)
        attempts.append(last_content)
        clean = re.sub(r"```(?:json)?|```", "", last_content).strip()
        # small models sometimes wrap JSON in text, grab the first {...} block
        if not clean.startswith("{"):
            match = re.search(r"\{.*\}", clean, re.DOTALL)
            if match:
                clean = match.group(0)
        try:
            parsed = json.loads(clean)
            if return_debug:
                return parsed, {"raw_response": last_content, "attempts": attempts}
            return parsed
        except json.JSONDecodeError:
            messages = messages + [
                {"role": "assistant", "content": last_content},
                {"role": "user", "content": "That was not valid JSON. Reply with a single JSON object only, no prose, no code fences."},
            ]

    parsed = {key: None for key in expected_keys}
    if return_debug:
        return parsed, {"raw_response": last_content, "attempts": attempts, "parse_error": "JSON decode failed after 3 attempts"}
    return parsed
