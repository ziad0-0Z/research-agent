"""Record/replay: a recorded run can be replayed with no API calls."""
import asyncio

import pytest
from agents import ModelSettings, OpenAIChatCompletionsModel
from agents.items import ModelResponse
from agents.usage import Usage
from openai import AsyncOpenAI

from research_agent.llm import AsyncRateLimiter, RateLimitedChatModel, ReplayMissError, request_key


@pytest.fixture
def fake_api(monkeypatch):
    calls = {"n": 0}

    async def fake_get_response(self, *args, **kwargs):
        calls["n"] += 1
        return ModelResponse(output=[], usage=Usage(), response_id=f"resp-{calls['n']}")

    monkeypatch.setattr(OpenAIChatCompletionsModel, "get_response", fake_get_response)
    return calls


def _model(mode, cache_dir):
    client = AsyncOpenAI(api_key="test", base_url="http://localhost:1")
    return RateLimitedChatModel("m", client, AsyncRateLimiter(100, 5), mode=mode, cache_dir=cache_dir)


def _call(model, user_input="hello"):
    return asyncio.run(model.get_response(
        "system", user_input, ModelSettings(), [], None, [], None,
    ))


def test_record_then_replay(fake_api, tmp_path):
    recorded = _call(_model("record", tmp_path))
    assert fake_api["n"] == 1

    replay = _model("replay", tmp_path)
    again = _call(replay)
    assert fake_api["n"] == 1  # no new API call
    assert again.response_id == recorded.response_id
    assert replay.cache_hits == 1


def test_replay_miss_never_goes_live(fake_api, tmp_path):
    with pytest.raises(ReplayMissError):
        _call(_model("replay", tmp_path), "never recorded")
    assert fake_api["n"] == 0


def test_live_mode_does_not_cache(fake_api, tmp_path):
    _call(_model("live", tmp_path))
    _call(_model("live", tmp_path))
    assert fake_api["n"] == 2
    assert not list(tmp_path.glob("*.pkl"))


def test_request_key_changes_with_input():
    a = request_key("m", "sys", "question A", None, [])
    b = request_key("m", "sys", "question B", None, [])
    assert a != b and a == request_key("m", "sys", "question A", None, [])
