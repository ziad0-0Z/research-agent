"""How long does one model call take? Finds the fastest working setting.

    python scripts/latency_check.py                       # your GEMINI_MODEL, reasoning off/none/low
    python scripts/latency_check.py --models gemini-3.5-flash-lite gemini-3.1-flash-lite
    python scripts/latency_check.py --groq                # also time the Groq fallback

Each setting is timed on a tiny prompt and on a page-sized prompt (~6k chars, like the
Claim Extractor gets). Calls run one after another with no retries, so the numbers are
the real model latency. Uses about 2 x settings x repeats requests from your daily quota.
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openai import AsyncOpenAI  # noqa: E402

from research_agent.config import ConfigError, load_settings  # noqa: E402

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GROQ_URL = "https://api.groq.com/openai/v1"
TINY = 'Return JSON {"answer": "<one word>"}: what colour is the sky on a clear day?'


def page_prompt() -> str:
    page = next(iter(sorted((ROOT / "data" / "corpus").glob("p03*"))), None)
    text = page.read_text(encoding="utf-8")[:6000] if page else ("Self-hosting costs vary. " * 250)
    return ("From the page below, return JSON {\"claims\": [{\"claim\": str, \"quote\": str}]} "
            "with up to 3 claims about LLM costs. Quotes must be copied exactly.\n\n" + text)


async def one_call(client: AsyncOpenAI, model: str, prompt: str, effort: str) -> tuple[float, str]:
    kwargs = {"model": model, "messages": [{"role": "user", "content": prompt}],
              "response_format": {"type": "json_object"}, "temperature": 0.1}
    if effort != "off":
        kwargs["reasoning_effort"] = effort
    t = time.perf_counter()
    try:
        r = await client.chat.completions.create(**kwargs)
        usage = r.usage
        tokens = f"{usage.prompt_tokens} in / {usage.completion_tokens} out" if usage else ""
        return time.perf_counter() - t, tokens
    except Exception as exc:  # noqa: BLE001 - report every failure, keep going
        return -1.0, f"{type(exc).__name__}: {str(exc)[:120]}"


async def time_setting(client, label, model, effort, repeats, prompts) -> None:
    for name, prompt in prompts:
        times, note = [], ""
        for _ in range(repeats):
            secs, info = await one_call(client, model, prompt, effort)
            if secs < 0:
                note = info
                break
            times.append(secs)
            note = info
        if times:
            print(f"  {label:42s} {name:6s} median {statistics.median(times):6.1f}s  "
                  f"(runs: {', '.join(f'{x:.1f}' for x in times)})  {note}")
        else:
            print(f"  {label:42s} {name:6s} ❌ {note}")


async def main(args) -> int:
    try:
        s = load_settings()
    except ConfigError as exc:
        print(f"[config] {exc}")
        return 1
    prompts = [("tiny", TINY), ("page", page_prompt())]
    timeout = max(s.llm_timeout_seconds, 180)
    if s.gemini_api_key:
        gem = AsyncOpenAI(base_url=GEMINI_URL, api_key=s.gemini_api_key, max_retries=0, timeout=timeout)
        for model in args.models or [s.gemini_model]:
            for effort in args.efforts:
                await time_setting(gem, f"gemini {model} reasoning={effort}", model, effort, args.repeats, prompts)
    else:
        print("GEMINI_API_KEY is empty, skipping Gemini")
    if args.groq:
        if not s.groq_api_key:
            print("GROQ_API_KEY is empty, skipping Groq")
        else:
            groq = AsyncOpenAI(base_url=GROQ_URL, api_key=s.groq_api_key, max_retries=0, timeout=timeout)
            await time_setting(groq, f"groq {s.groq_model}", s.groq_model, "off", args.repeats, prompts)
    print("\nPick the fastest setting that works and put it in .env "
          "(GEMINI_MODEL / LLM_REASONING_EFFORT). Under ~10s per call is healthy.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", help="Gemini model names to compare (default: GEMINI_MODEL)")
    ap.add_argument("--efforts", nargs="*", default=["off", "none", "low"],
                    help="reasoning settings to compare: off (not sent), none, minimal, low, medium")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--groq", action="store_true")
    raise SystemExit(asyncio.run(main(ap.parse_args())))
