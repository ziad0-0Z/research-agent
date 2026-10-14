"""LLM layer: provider switch (Gemini / Groq), rate limiting, retries, record/replay.

Every agent gets its model from `get_model()`, so all LLM calls share one rate
limiter. Free tiers limit requests per minute; without this, parallel
specialists would trip 429 errors.

FALLBACK: when the main model is overloaded (503), rate-limited (429), times out or
cannot be reached, the same request goes to the next model in LLM_FALLBACKS (each with
its own rate limiter, since quotas are per model/provider). The main model is then skipped
for a cooldown period so a busy provider is not hammered. Every switch is logged.

LLM_MODE:
- live   : call the API (default)
- record : call the API and save each response under data/cache/llm/
- replay : never call the API; return saved responses. A request that was
           never recorded raises ReplayMissError (it never silently goes live).
"""
from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import pickle
import time
from collections import deque
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from openai import (
    APIConnectionError, APITimeoutError, AsyncOpenAI, InternalServerError, RateLimitError,
)
from openai.types.shared import Reasoning

from agents import ModelSettings, OpenAIChatCompletionsModel

from .config import LLMMode, Settings, load_settings


class ReplayMissError(RuntimeError):
    """Replay mode got a request that was never recorded."""


class AsyncRateLimiter:
    """Sliding-window requests-per-minute limit plus a concurrency cap.

    acquire() waits (never fails) until a slot is free, so callers degrade to
    slower instead of erroring when the free-tier budget is tight.
    """

    def __init__(self, requests_per_minute: int, max_concurrent: int, window_seconds: float = 60.0):
        if requests_per_minute <= 0 or max_concurrent <= 0:
            raise ValueError("limits must be positive")
        self.rpm = requests_per_minute
        self.window = window_seconds
        self._calls: deque[float] = deque()
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self.total_wait_seconds = 0.0
        self.total_calls = 0

    async def _wait_for_window(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] >= self.window:
                    self._calls.popleft()
                if len(self._calls) < self.rpm:
                    self._calls.append(now)
                    self.total_calls += 1
                    return
                sleep_for = self.window - (now - self._calls[0]) + 0.01
            self.total_wait_seconds += sleep_for
            await asyncio.sleep(sleep_for)

    async def __aenter__(self) -> "AsyncRateLimiter":
        await self._semaphore.acquire()
        try:
            await self._wait_for_window()
        except BaseException:
            self._semaphore.release()
            raise
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self._semaphore.release()


def request_key(model: str, system_instructions: str | None, input: Any,
                output_schema: Any, tools: list[Any]) -> str:
    """Stable fingerprint of one model request, used as the replay cache key."""
    schema_name = None
    if output_schema is not None:
        schema_name = getattr(output_schema, "name", lambda: None)()
    payload = {
        "model": model,
        "system": system_instructions or "",
        "input": input,
        "schema": schema_name,
        "tools": sorted(getattr(t, "name", str(t)) for t in tools),
    }
    raw = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


# Errors worth trying another model for. Bad requests (400) and auth errors are not:
# they would fail the same way, and hiding them would hide real bugs.
FALLBACK_ERRORS = (InternalServerError, RateLimitError, APITimeoutError, APIConnectionError)


@dataclasses.dataclass
class FallbackModel:
    label: str                      # e.g. "groq/qwen/qwen3.8-27b"
    model: Any                      # anything with the SDK Model.get_response signature
    limiter: AsyncRateLimiter
    drop_reasoning: bool = True     # reasoning settings differ per provider; fallbacks use defaults
    # Can it take tools AND a structured output (response_format) in the same request?
    # Groq can't ("json mode cannot be combined with tool/function calling", 400), so it is skipped
    # for tool-using agents with an output_type (the researchers) and still used for everything else.
    tools_with_schema: bool = True


# Providers that reject tools + response_format in one request (verified live, Oct 2026).
NO_TOOLS_WITH_SCHEMA = {"groq"}


class RateLimitedChatModel(OpenAIChatCompletionsModel):
    """Chat Completions model with the shared rate limiter, fallback chain and record/replay."""

    def __init__(self, model: str, openai_client: AsyncOpenAI, limiter: AsyncRateLimiter,
                 mode: LLMMode = "live", cache_dir: Path | None = None,
                 fallbacks: list[FallbackModel] | None = None, cooldown_seconds: float = 60.0,
                 label: str | None = None, tools_with_schema: bool = True):
        super().__init__(model=model, openai_client=openai_client)
        self.tools_with_schema = tools_with_schema
        self._limiter = limiter
        self.mode = mode
        self.cache_dir = cache_dir
        self.cache_hits = 0
        self.label = label or model
        self.fallbacks = list(fallbacks or [])
        self.cooldown_seconds = cooldown_seconds
        self._cooling_until: dict[str, float] = {}
        self.calls_by_model: dict[str, int] = {}
        self.fallback_events: list[dict] = []
        if mode != "live":
            if cache_dir is None:
                raise ValueError("record/replay mode needs a cache_dir")
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, key: str) -> Path:
        assert self.cache_dir is not None
        return self.cache_dir / f"{key}.pkl"

    async def get_response(self, system_instructions, input, model_settings, tools,  # type: ignore[override]
                           output_schema, handoffs, tracing, *args: Any, **kwargs: Any):
        key = None
        if self.mode != "live":
            key = request_key(self.model, system_instructions, input, output_schema, tools)
            path = self._cache_path(key)
            if path.exists():
                self.cache_hits += 1
                return pickle.loads(path.read_bytes())
            if self.mode == "replay":
                raise ReplayMissError(
                    f"No recorded response for request {key}. Run once with LLM_MODE=record first."
                )
        response = await self._call_with_fallback(
            system_instructions, input, model_settings, tools, output_schema, handoffs, tracing, *args, **kwargs)
        if self.mode == "record" and key is not None:
            self._cache_path(key).write_bytes(pickle.dumps(response))
        return response

    def _chain(self, tools: Any = None, output_schema: Any = None) -> list[tuple[str, Any, AsyncRateLimiter, bool]]:
        primary = (self.label, None, self._limiter, False, self.tools_with_schema)  # None = super().get_response
        chain = [primary] + [(f.label, f.model, f.limiter, f.drop_reasoning, f.tools_with_schema)
                             for f in self.fallbacks]
        if tools and output_schema is not None:  # keep only models that accept this kind of request
            chain = [c for c in chain if c[4]] or chain
        return [c[:4] for c in chain]

    async def _call_with_fallback(self, system_instructions, input, model_settings, tools, output_schema,
                                  handoffs, tracing, *args: Any, **kwargs: Any):
        chain = self._chain(tools, output_schema)
        now = time.monotonic()
        # Skip models that failed recently, unless that would skip everything.
        ready = [c for c in chain if self._cooling_until.get(c[0], 0) <= now] or chain
        last_exc: Exception | None = None
        for i, (label, model, limiter, drop_reasoning) in enumerate(ready):
            settings = model_settings
            if drop_reasoning and getattr(model_settings, "reasoning", None) is not None:
                settings = dataclasses.replace(model_settings, reasoning=None)
            try:
                async with limiter:
                    if model is None:
                        response = await super().get_response(
                            system_instructions, input, settings, tools, output_schema,
                            handoffs, tracing, *args, **kwargs)
                    else:
                        response = await model.get_response(
                            system_instructions, input, settings, tools, output_schema,
                            handoffs, tracing, *args, **kwargs)
            except FALLBACK_ERRORS as exc:
                last_exc = exc
                self._cooling_until[label] = time.monotonic() + self.cooldown_seconds
                if i + 1 < len(ready):
                    self.fallback_events.append({
                        "time": time.time(), "from": label, "to": ready[i + 1][0],
                        "error": f"{type(exc).__name__}: {str(exc)[:120]}",
                    })
                continue
            self.calls_by_model[label] = self.calls_by_model.get(label, 0) + 1
            return response
        assert last_exc is not None
        raise last_exc

    async def stream_response(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:  # type: ignore[override]
        if self.mode == "replay":
            raise ReplayMissError("Streaming is not supported in replay mode; use Runner.run.")
        async with self._limiter:
            async for event in super().stream_response(*args, **kwargs):
                yield event


_model: RateLimitedChatModel | None = None


def build_client(settings: Settings, provider: str | None = None, max_retries: int | None = None) -> AsyncOpenAI:
    """Client for the main provider, or for a fallback provider."""
    from .config import PROVIDER_BASE_URLS
    if provider is None:
        settings.require_llm_key()
    provider = provider or settings.llm_provider
    # The OpenAI client retries 429/5xx with exponential backoff on its own.
    return AsyncOpenAI(
        base_url=PROVIDER_BASE_URLS[provider],
        api_key=settings.api_key_for(provider) or "replay-mode-no-key",
        max_retries=settings.llm_max_retries if max_retries is None else max_retries,
        timeout=settings.llm_timeout_seconds,
    )


def _fallback_models(settings: Settings) -> list[FallbackModel]:
    out = []
    for provider, model in settings.llm_fallbacks:
        out.append(FallbackModel(
            label=f"{provider}/{model}",
            model=OpenAIChatCompletionsModel(model=model, openai_client=build_client(settings, provider)),
            limiter=AsyncRateLimiter(settings.llm_requests_per_minute, settings.llm_max_concurrent),
            tools_with_schema=provider not in NO_TOOLS_WITH_SCHEMA,
        ))
    return out


def get_model(settings: Settings | None = None) -> RateLimitedChatModel:
    """Return the shared, rate-limited model for the configured provider."""
    global _model
    if _model is None:
        settings = settings or load_settings()
        limiter = AsyncRateLimiter(settings.llm_requests_per_minute, settings.llm_max_concurrent)
        fallbacks = _fallback_models(settings) if settings.llm_mode != "replay" else []
        # With a fallback ready, give up on a busy main model sooner (1 retry) and switch.
        retries = min(settings.llm_max_retries, 1) if fallbacks else settings.llm_max_retries
        _model = RateLimitedChatModel(
            settings.model_name, build_client(settings, max_retries=retries), limiter,
            mode=settings.llm_mode, cache_dir=settings.llm_cache_dir,
            fallbacks=fallbacks, cooldown_seconds=settings.llm_fallback_cooldown_seconds,
            label=f"{settings.llm_provider}/{settings.model_name}",
            tools_with_schema=settings.llm_provider not in NO_TOOLS_WITH_SCHEMA,
        )
    return _model


def reset_model() -> None:
    """Drop the cached model (used after switching provider or mode, and in tests)."""
    global _model
    _model = None


def _reasoning(settings: Settings) -> Reasoning | None:
    effort = settings.llm_reasoning_effort
    return None if effort == "off" else Reasoning(effort=effort)  # type: ignore[arg-type]


def precise_settings(settings: Settings | None = None) -> ModelSettings:
    """Deterministic-leaning settings for structured steps (planning, judging)."""
    settings = settings or load_settings()
    return ModelSettings(temperature=0.1, reasoning=_reasoning(settings))


def writing_settings(settings: Settings | None = None) -> ModelSettings:
    """Slightly more fluent settings for writing the report."""
    settings = settings or load_settings()
    return ModelSettings(temperature=0.4, reasoning=_reasoning(settings))
