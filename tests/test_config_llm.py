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


def test_worker_timeout_from_env(monkeypatch):
    from research_agent import config as cfg
    monkeypatch.setenv("WORKER_TIMEOUT_SECONDS", "600")
    assert cfg.load_settings().worker_timeout_seconds == 600
    monkeypatch.setenv("WORKER_TIMEOUT_SECONDS", "5")
    with pytest.raises(cfg.ConfigError):
        cfg.load_settings()


# ---------------------------------------------------------------- fallback chain

def _overloaded():
    import httpx
    from openai import InternalServerError
    resp = httpx.Response(503, request=httpx.Request("POST", "https://example.test"))
    return InternalServerError("This model is currently experiencing high demand.", response=resp, body=None)


class _FakeModel:
    def __init__(self, reply="ok", error=None):
        self.reply, self.error, self.calls, self.settings = reply, error, 0, []

    async def get_response(self, system_instructions, input, model_settings, *args, **kwargs):
        self.calls += 1
        self.settings.append(model_settings)
        if self.error:
            raise self.error
        return self.reply


def _chain_model(monkeypatch, primary: _FakeModel, *fallbacks: _FakeModel, cooldown=60.0, no_tools_schema=()):
    from agents import OpenAIChatCompletionsModel
    from openai import AsyncOpenAI
    from research_agent.llm import FallbackModel, RateLimitedChatModel

    async def primary_call(self, *args, **kwargs):
        return await primary.get_response(*args, **kwargs)
    monkeypatch.setattr(OpenAIChatCompletionsModel, "get_response", primary_call)
    return RateLimitedChatModel(
        "main", AsyncOpenAI(api_key="x"), AsyncRateLimiter(100, 5), label="gemini/main", cooldown_seconds=cooldown,
        fallbacks=[FallbackModel(f"fb{i}", fb, AsyncRateLimiter(100, 5), tools_with_schema=i not in no_tools_schema)
                   for i, fb in enumerate(fallbacks)])


def _ask(model):
    from agents import ModelSettings
    from openai.types.shared import Reasoning
    return asyncio.run(model.get_response("sys", "hi", ModelSettings(reasoning=Reasoning(effort="none")),
                                          [], None, [], None))


def test_overloaded_main_model_falls_back_and_cools_down(monkeypatch):
    primary, backup = _FakeModel(error=_overloaded()), _FakeModel(reply="from backup")
    model = _chain_model(monkeypatch, primary, backup)
    assert _ask(model) == "from backup"
    assert model.fallback_events[0]["from"] == "gemini/main" and model.fallback_events[0]["to"] == "fb0"
    assert backup.settings[0].reasoning is None  # provider-specific reasoning dropped for the fallback
    assert _ask(model) == "from backup"
    assert primary.calls == 1  # cooling down: the busy model was not called again
    assert model.calls_by_model == {"fb0": 2}


def test_fallback_errors_only_for_overload_not_bad_requests(monkeypatch):
    import httpx
    from openai import BadRequestError
    bad = BadRequestError("invalid argument", response=httpx.Response(
        400, request=httpx.Request("POST", "https://example.test")), body=None)
    model = _chain_model(monkeypatch, _FakeModel(error=bad), _FakeModel(reply="never"))
    with pytest.raises(BadRequestError):
        _ask(model)


def test_all_models_down_raises_last_error(monkeypatch):
    from openai import InternalServerError
    model = _chain_model(monkeypatch, _FakeModel(error=_overloaded()), _FakeModel(error=_overloaded()))
    with pytest.raises(InternalServerError):
        _ask(model)


def test_fallbacks_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("GROQ_API_KEY", "q")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    monkeypatch.delenv("LLM_FALLBACKS", raising=False)
    assert cfg.load_settings().llm_fallbacks == (("groq", "qwen/qwen3.8-27b"),)
    monkeypatch.setenv("LLM_FALLBACKS", "gemini:gemini-3.5-flash-lite, groq:qwen/qwen3.8-27b, gemini:gemini-3.1-flash-lite")
    assert cfg.load_settings().llm_fallbacks == (("gemini", "gemini-3.5-flash-lite"), ("groq", "qwen/qwen3.8-27b"))
    monkeypatch.setenv("GROQ_API_KEY", "")
    assert cfg.load_settings().llm_fallbacks == (("gemini", "gemini-3.5-flash-lite"),)  # no key → skipped
    monkeypatch.setenv("LLM_FALLBACKS", "none")
    assert cfg.load_settings().llm_fallbacks == ()
    monkeypatch.setenv("LLM_FALLBACKS", "openai-gpt")
    with pytest.raises(cfg.ConfigError):
        cfg.load_settings()


def test_tool_agents_with_output_type_skip_providers_that_reject_it(monkeypatch):
    """Live bug (Oct 2026): researchers fell back to Groq, which returns 400 for tools + response_format."""
    from agents import ModelSettings
    groq, gemini_b = _FakeModel(reply="from groq"), _FakeModel(reply="from gemini b")
    model = _chain_model(monkeypatch, _FakeModel(error=_overloaded()), groq, gemini_b, no_tools_schema=(0,))
    tools, schema = [object()], object()
    out = asyncio.run(model.get_response("sys", "hi", ModelSettings(), tools, schema, [], None))
    assert out == "from gemini b" and groq.calls == 0
    # Plain structured output (no tools) may still use Groq.
    model2 = _chain_model(monkeypatch, _FakeModel(error=_overloaded()), _FakeModel(reply="from groq"),
                          no_tools_schema=(0,))
    assert asyncio.run(model2.get_response("sys", "hi", ModelSettings(), [], schema, [], None)) == "from groq"
