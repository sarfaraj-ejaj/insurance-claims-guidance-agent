"""LLM client for the fuzzy steps (T08 incident/location type, T09 damage amount).

ask_json(prompt, schema) -> dict. Answers are cached under data/llm_cache keyed by a hash of
model + system + prompt + schema, so reruns are identical and free (NFR-2). Without credentials, or on any
refusal / truncation / API failure, it raises LLMUnavailable and the caller writes null plus a flag -
a missing answer is never guessed. The key is read from the environment only and is never stored.

Needs `pip install anthropic` only for real calls; tests inject a fake client.
"""
import hashlib
import json
import os
from pathlib import Path

MODEL = "claude-opus-5-5"  # one place to change; claude-sonnet-5-5 is the cheaper option
FALLBACK_BETA = "server-side-fallback-2026-07-01"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "llm_cache"


class LLMUnavailable(Exception):
    """No usable answer (no credentials, refusal, truncation, API or parse error)."""


def _default_client():
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        raise LLMUnavailable("ANTHROPIC_API_KEY is not set")
    try:
        import anthropic
    except ImportError as e:
        raise LLMUnavailable("the anthropic package is not installed (pip install anthropic)") from e
    return anthropic.Anthropic()


def ask_json(prompt, schema, system=None, *, client=None, cache_dir=CACHE_DIR, model=MODEL):
    """Return the model's JSON answer for `prompt`, shaped by `schema` (cached)."""
    key = hashlib.sha256(json.dumps({"model": model, "system": system, "prompt": prompt, "schema": schema},
                                    sort_keys=True).encode("utf-8")).hexdigest()
    path = Path(cache_dir) / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["response"]

    client = client or _default_client()
    try:
        resp = client.beta.messages.create(
            model=model, max_tokens=16000, system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": "medium", "format": {"type": "json_schema", "schema": schema}},
            betas=[FALLBACK_BETA], fallbacks="default")
    except Exception as e:  # SDK/network errors: surface as 'unavailable', keep the cause in the message
        raise LLMUnavailable(f"{type(e).__name__}: {e}") from e
    if resp.stop_reason == "refusal":
        category = getattr(getattr(resp, "stop_details", None), "category", None)
        raise LLMUnavailable(f"refused ({category})")
    if resp.stop_reason == "max_tokens":
        raise LLMUnavailable("answer truncated at max_tokens")
    text = next((b.text for b in resp.content if b.type == "text"), None)
    try:
        data = json.loads(text)
    except (TypeError, ValueError) as e:
        raise LLMUnavailable("response was not valid JSON") from e
    if not isinstance(data, dict):
        raise LLMUnavailable("response was not a JSON object")

    path.parent.mkdir(parents=True, exist_ok=True)  # only the answer is stored: no key, no claim text
    path.write_text(json.dumps({"model": model, "response": data}, sort_keys=True, ensure_ascii=False) + "\n",
                    encoding="utf-8", newline="\n")
    return data
