"""Whole-codebase review fixes (2026-10-14): each test fails without its fix."""
import asyncio

import pytest

from research_agent.context import AppContext
from research_agent.guardrails.url_safety import check_url
from research_agent.models import Evidence, Finding, ResearchReport, Source
from research_agent.store import Store
from research_agent.tools import web
from research_agent.workflows import pipeline

from test_workflows import make_fake_run_agent, settings  # noqa: F401  (fixture)


@pytest.mark.parametrize("url", [
    "http://127.1/", "http://0177.0.0.1/", "http://0x7f.0.0.1/", "http://2130706433/",  # loopback, old notations
    "http://100.64.0.1/",                                                               # carrier-grade NAT
    "http://[::ffff:169.254.169.254]/latest/meta-data",                                 # metadata via IPv6
])
def test_url_safety_blocks_non_public_addresses_in_any_notation(url):
    assert not check_url(url).allowed


@pytest.mark.parametrize("url", ["https://8.8.8.8/", "https://[2606:4700:4700::1111]/", "https://example.com/a",
                                 "https://123.example/"])
def test_url_safety_still_allows_public_addresses(url):
    assert check_url(url).allowed


def test_crashed_run_is_marked_failed_with_its_stage(settings):
    base = make_fake_run_agent()

    async def broken_planner(agent, prompt, ctx, max_turns):
        if agent.name == "Planner":
            raise RuntimeError("500 upstream")
        return await base(agent, prompt, ctx, max_turns)

    store = Store(":memory:")
    with pytest.raises(pipeline.PipelineError) as err:
        asyncio.run(pipeline.run_research("Should a 5-person startup use cloud LLM APIs or self-host?",
                                          settings=settings, store=store, run_agent=broken_planner))
    rows = store._query("SELECT status, state_json FROM runs")
    assert rows[0]["status"] == "failed" and '"stage": "planner"' in rows[0]["state_json"]
    assert err.value.stage == "planner"


def test_two_workers_scraping_one_page_register_it_once(settings, monkeypatch):
    real = web.fetch_page

    async def slow_fetch(ctx, url):
        page = await real(ctx, url)
        await asyncio.sleep(0.05)  # both workers are now past the first "already saved?" check
        return page

    monkeypatch.setattr(web, "fetch_page", slow_fetch)
    ctx = AppContext(settings=settings, store=Store(":memory:"), question="test question here")
    url = "https://applied-systems-institute.example/reports/llm-tco-small-teams"

    async def both():
        return await asyncio.gather(web.scrape_page(ctx, url), web.scrape_page(ctx, url))

    a, b = asyncio.run(both())
    assert a["source_id"] == b["source_id"] and len(ctx.store.list_sources(ctx.run_id)) == 1


def test_strength_rule_counts_pages_not_source_rows(settings):
    from research_agent.workflows.eval_loop import apply_strength_rules
    ctx = AppContext(settings=settings, store=Store(":memory:"), question="test question here")
    for sid in ("src1", "src2"):  # the same page registered twice (as the race used to do)
        ctx.store.add_source(ctx.run_id, Source(id=sid, url="https://blog.example/p", title="t", origin="offline",
                                                quality_score=0.5), text="x" * 300)
    for eid, sid in (("ev1", "src1"), ("ev2", "src2")):
        ctx.store.add_evidence(Evidence.build(evidence_id=eid, run_id=ctx.run_id, subquestion_id="sq1",
                                              claim="A long enough claim.", quote=f"quote number {eid} here",
                                              source_id=sid, verified=True))
    report = ResearchReport(title="Title here", executive_summary="x" * 40,
                            findings=[Finding(statement="One page said it twice.", evidence_ids=["ev1", "ev2"],
                                              strength="strong")],
                            contradictions=[], limitations=[], recommendation="y" * 25, impact="low")
    fixed, notes = apply_strength_rules(ctx, report, {})
    assert fixed.findings[0].strength == "moderate"
