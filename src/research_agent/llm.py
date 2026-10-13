"""LLM layer: provider switch (Gemini / Groq), rate limiting, retries, record/replay.

Every agent gets its model from `get_model()`, so all LLM calls share one rate
limiter. Free tiers limit requests per minute; without this, parallel
specialists would trip 429 errors.

LLM_MODE:
- live   : call the API (default)
- record : call the API and save each response under data/cache/llm/
- replay : never call the API; return saved responses. A request that was
           never recorded raises ReplayMissError (it never silently goes live).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import pickle
import time
from collections import deque
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
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

    async def _wait_for_window(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] >= self.window:
                    self._calls.popleft()
                if len(self._calls) < self.rpm:
                    self._calls.append(now)
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


class RateLimitedChatModel(OpenAIChatCompletionsModel):
    """Chat Completions model with the shared rate limiter and record/replay."""

    def __init__(self, model: str, openai_client: AsyncOpenAI, limiter: AsyncRateLimiter,
                 mode: LLMMode = "live", cache_dir: Path | None = None):
        super().__init__(model=model, openai_client=openai_client)
        self._limiter = limiter
        self.mode = mode
        self.cache_dir = cache_dir
        self.cache_hits = 0
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
        async with self._limiter:
            response = await super().get_response(
                system_instructions, input, model_settings, tools, output_schema,
                handoffs, tracing, *args, **kwargs,
            )
        if self.mode == "record" and key is not None:
            self._cache_path(key).write_bytes(pickle.dumps(response))
        return response

    async def stream_response(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:  # type: ignore[override]
        if self.mode == "replay":
            raise ReplayMissError("Streaming is not supported in replay mode; use Runner.run.")
        async with self._limiter:
            async for event in super().stream_response(*args, **kwargs):
                yield event


_model: RateLimitedChatModel | None = None


def build_client(settings: Settings) -> AsyncOpenAI:
    settings.require_llm_key()
    # The OpenAI client retries 429/5xx with exponential backoff on its own.
    return AsyncOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key or "replay-mode-no-key",
        max_retries=settings.llm_max_retries,
        timeout=settings.llm_timeout_seconds,
    )


def get_model(settings: Settings | None = None) -> RateLimitedChatModel:
    """Return the shared, rate-limited model for the configured provider."""
    global _model
    if _model is None:
        settings = settings or load_settings()
        limiter = AsyncRateLimiter(settings.llm_requests_per_minute, settings.llm_max_concurrent)
        _model = RateLimitedChatModel(
            settings.model_name, build_client(settings), limiter,
            mode=settings.llm_mode, cache_dir=settings.llm_cache_dir,
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
