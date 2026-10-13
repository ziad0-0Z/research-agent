"""Find which models are usable right now: speed, availability, tool calling.

Usage (from the project root):
    python scripts/check_models.py gemini
    python scripts/check_models.py groq

Each candidate gets one tiny chat request and one tiny tool-calling request,
with no retries and a short timeout, so busy or missing models fail fast.
"""
from __future__ import annotations

import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PROVIDERS = {
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
}

GEMINI_CANDIDATES = [
    "gemini-flash-lite-latest",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-flash-latest",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.8-flash",
]

TOOL = {
    "type": "function",
    "function": {
        "name": "get_code",
        "description": "Return the reference code for a topic.",
        "parameters": {
            "type": "object",
            "properties": {"topic": {"type": "string"}},
            "required": ["topic"],
        },
    },
}


def groq_candidates(client: OpenAI) -> list[str]:
    ids = sorted(m.id for m in client.models.list())
    print("Groq models on your account:\n  " + "\n  ".join(ids) + "\n")
    skip = ("whisper", "tts", "guard", "playai", "orpheus", "distil", "compound")
    return [m for m in ids if not any(s in m.lower() for s in skip)]


def check(client: OpenAI, model: str) -> None:
    start = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with the single word: pong"}],
        )
        chat_s = time.perf_counter() - start
        text = (resp.choices[0].message.content or "").strip()[:20]
    except Exception as exc:  # noqa: BLE001 - we only report
        print(f"  ❌ {model:32s} {type(exc).__name__}: {str(exc)[:90]}")
        return

    start = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Use the get_code tool for the topic 'llm'."}],
            tools=[TOOL],
        )
        tool_s = time.perf_counter() - start
        tools_work = bool(resp.choices[0].message.tool_calls)
        tool_ok = "tools ✅" if tools_work else "tools ❌ (no call)"
    except Exception as exc:  # noqa: BLE001
        tool_s = time.perf_counter() - start
        tools_work = False
        tool_ok = f"tools ❌ {type(exc).__name__}"
    # ✅ only when both chat and tool calling work; ⚠️ when chat works but tools don't
    mark = "✅" if tools_work else "⚠️"
    print(f"  {mark} {model:32s} chat {chat_s:5.1f}s ({text!r}) | {tool_ok} {tool_s:5.1f}s")


def main() -> int:
    provider = (sys.argv[1] if len(sys.argv) > 1 else "gemini").lower()
    if provider not in PROVIDERS:
        print(f"Unknown provider {provider!r}; use gemini or groq")
        return 1
    base_url, key_name = PROVIDERS[provider]
    key = os.getenv(key_name, "").strip()
    if not key:
        print(f"{key_name} is empty in .env")
        return 1
    client = OpenAI(base_url=base_url, api_key=key, max_retries=0, timeout=25)
    models = GEMINI_CANDIDATES if provider == "gemini" else groq_candidates(client)
    print(f"Testing {len(models)} {provider} models (no retries, 25s timeout)...\n")
    for model in models:
        check(client, model)
    print("\n✅ = chat and tools work · ⚠️ = chat only (not usable for agents) · ❌ = unavailable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
