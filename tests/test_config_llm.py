import asyncio
import time

import pytest

from research_agent import config as cfg
from research_agent.llm import AsyncRateLimiter


def test_settings_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    s = cfg.load_settings()
    assert s.llm_base_url.startswith("https://generativelanguage.googleapis.com")
    assert s.max_critic_rounds == 3


def test_bad_provider_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(cfg.ConfigError):
        cfg.load_settings()


def test_missing_key_message(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "")
    s = cfg.load_settings()
    with pytest.raises(cfg.ConfigError, match="GROQ_API_KEY"):
        s.require_llm_key()


def test_bad_int(monkeypatch):
    monkeypatch.setenv("LLM_REQUESTS_PER_MINUTE", "fast")
    with pytest.raises(cfg.ConfigError):
        cfg.load_settings()


def test_rate_limiter_enforces_window():
    async def run():
        limiter = AsyncRateLimiter(requests_per_minute=3, max_concurrent=10, window_seconds=0.5)
        start = time.monotonic()
        for _ in range(4):
            async with limiter:
                pass
        return time.monotonic() - start

    assert asyncio.run(run()) >= 0.45  # 4th call had to wait for the window


def test_rate_limiter_caps_concurrency():
    async def run():
        limiter = AsyncRateLimiter(requests_per_minute=100, max_concurrent=2)
        active = peak = 0

        async def job():
            nonlocal active, peak
            async with limiter:
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(0.05)
                active -= 1

        await asyncio.gather(*(job() for _ in range(6)))
        return peak

    assert asyncio.run(run()) == 2
