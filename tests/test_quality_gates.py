"""Quality gates: reliability honesty, independent sources, memory relevance, UI details."""
from __future__ import annotations

import asyncio
import sqlite3
from datetime import timedelta

import pytest

from research_agent.context import CURRENT_SPECIALIST, AppContext
from research_agent.memory.matching import shares_topic, topic_terms
from research_agent.memory.store import MemoryItem
from research_agent.models import Evidence, Finding, ResearchPlan, ResearchReport, Source, SubQuestion, utcnow
from research_agent.store import Store
from research_agent.ui import components as c
from research_agent.workflows.parallel import WorkerTask, run_parallel
from research_agent.workflows.quality import (
    INSUFFICIENT_PREFIX, Coverage, enforce_recommendation_policy, independent_sources, normalize_quote, plan_gaps,
)
from research_agent.workflows.release import impact_gate
from research_agent.workflows.reliability import (
    build_reliability, degraded, error_count, explain_worker_error, fetch_degraded, search_notice,
)

from test_workflows import settings  # noqa: F401  (fixture)

ECOMMERCE_Q = ("Should a small e-commerce startup build its own recommendation system using collaborative filtering, "
               "or use a managed recommendation API? Compare implementation cost, recommendation quality, data "
               "privacy, scalability, vendor lock-in, and the time required to launch.")


def _ctx(settings, **kw) -> AppContext:  # noqa: F811
    return AppContext(settings=settings, store=Store(":memory:"), question="q", **kw)


def _source(ctx, sid, url, quality=0.8):
    ctx.store.add_source(ctx.run_id, Source(id=sid, url=url, title=sid, origin="offline", quality_score=quality),
                         text="x" * 300)


def _evidence(ctx, eid, sid, quote, sq="sq1", support="supported"):
    ev = Evidence.build(evidence_id=eid, run_id=ctx.run_id, subquestion_id=sq, claim="A claim long enough.",
                        quote=quote, source_id=sid, verified=True)
    ctx.store.add_evidence(ev)
    if support != "unchecked":
        ctx.store.set_support(ctx.run_id, eid, support)


def _plan(question=ECOMMERCE_Q, subs=None) -> ResearchPlan:
    subs = subs or [("sq1", "What does building a collaborative filtering recommender cost?", "market"),
                    ("sq2", "How do managed recommendation APIs handle customer data privacy?", "risk")]
    return ResearchPlan(main_question=question, decision_context="A small startup.", subquestions=[
        SubQuestion(id=i, question=q, type=t, priority=1, success_criteria="Concrete numbers or facts",
                    search_queries=[q[:40]]) for i, q, t in subs])


def _report(recommendation="Build it in-house now; revisit at scale.", ids=("ev1",)) -> ResearchReport:
    return ResearchReport(title="A report title", executive_summary="Summary of the decision, long enough here.",
                          findings=[Finding(statement="A finding statement.", evidence_ids=list(ids),
                                            strength="moderate")],
                          contradictions=[], limitations=[], recommendation=recommendation, impact="medium")


# --------------------------------------------------------------------------- reliability

def test_failed_fetches_are_degradation_and_errors(settings):  # noqa: F811
    ctx = _ctx(settings)
    for i in range(17):
        ctx.log_tool("scrape_page", False, f"https://site{i}.example: HTTPStatusError")
    rel = build_reliability(ctx, [])
    assert rel["fetch"] == {"calls": 17, "failed": 17, "reasons": rel["fetch"]["reasons"]}
    assert fetch_degraded(rel) and degraded(rel) and error_count(rel) == 17
    assert "HTTPStatusError" in rel["fetch"]["reasons"][0]["reason"]


def test_a_few_failed_fetches_are_not_degradation(settings):  # noqa: F811
    ctx = _ctx(settings)
    for ok in (True, True, True, True, False):
        ctx.log_tool("scrape_page", ok, "x")
    assert not fetch_degraded(build_reliability(ctx, []))


def test_tool_log_knows_which_specialist_called(settings):  # noqa: F811
    ctx = _ctx(settings)

    async def as_market():
        CURRENT_SPECIALIST.set("market")
        ctx.log_tool("scrape_page", False, "boom")

    asyncio.run(as_market())
    assert ctx.tool_log[-1].specialist == "market"


def test_worker_errors_get_plain_reasons():
    assert explain_worker_error("MaxTurnsExceeded: Max turns (12) exceeded") == "hit the turn limit before finishing"
    assert explain_worker_error("timed out after 480s") == "ran past the worker timeout"
    assert explain_worker_error(None) == ""


def test_offline_only_search_is_explained(settings):  # noqa: F811
    off = settings.__class__(**{**settings.__dict__, "search_mode": "offline"})
    ctx = _ctx(off)
    ctx.search_log.append({"specialist": "market", "provider": "offline", "first_choice": "offline",
                           "fallback_used": False, "failed": False, "unavailable": []})
    rel = build_reliability(ctx, [])
    assert rel["search"]["offline_only"] and not rel["search"]["live"]
    assert "Live web search was off" in search_notice(rel)


def test_a_crashed_worker_is_retried_once_but_a_timeout_is_not(settings):  # noqa: F811
    ctx = _ctx(settings)
    calls = {"crash": 0, "slow": 0}

    async def crash():
        calls["crash"] += 1
        raise ValueError("schema mismatch")

    async def slow():
        calls["slow"] += 1
        await asyncio.sleep(5)

    out = asyncio.run(run_parallel(ctx, [WorkerTask("market", ["sq1"], crash), WorkerTask("risk", ["sq2"], slow)],
                                   timeout=0.2, retries=1))
    assert calls == {"crash": 2, "slow": 1}
    by = {w.specialist: w for w in out}
    assert by["market"].attempts == 2 and by["market"].status == "failed" and "ValueError" in by["market"].error
    assert by["risk"].attempts == 1 and by["risk"].status == "timeout"


# --------------------------------------------------------------------------- quality gates

def test_repeated_quote_and_same_site_count_once(settings):  # noqa: F811
    ctx = _ctx(settings)
    _source(ctx, "src1", "https://www.alpha.example/a")
    _source(ctx, "src2", "https://alpha.example/b")       # same site, other page
    _source(ctx, "src3", "https://beta.example/c")
    _evidence(ctx, "ev1", "src1", "Self-hosting keeps data inside the organisation.")
    _evidence(ctx, "ev2", "src2", "Another sentence from the same site entirely.")
    _evidence(ctx, "ev3", "src3", "Self-hosting keeps data inside the organization!")  # same sentence, US spelling
    assert normalize_quote("Labour, organisation") == normalize_quote("labor organization")
    assert independent_sources(ctx, ["ev1", "ev2"]) == {"alpha.example"}
    assert independent_sources(ctx, ["ev1", "ev3"]) == {"alpha.example"}  # ev3 repeats ev1


def test_single_site_report_goes_to_review(settings):  # noqa: F811
    ctx = _ctx(settings)
    reasons = impact_gate(ctx, _report(), {"score": 0.9, "independent_sources": 1})
    assert any("only 1 independent source" in r for r in reasons)
    assert not any("independent" in r for r in impact_gate(ctx, _report(), {"score": 0.9, "independent_sources": 2}))


def test_mostly_failed_fetches_go_to_review(settings):  # noqa: F811
    ctx = _ctx(settings)
    ctx.reliability = {"fetch": {"calls": 17, "failed": 17, "reasons": []}}
    assert any("17 of 17 page fetches failed" in r
               for r in impact_gate(ctx, _report(), {"score": 0.9, "independent_sources": 3}))


def test_plan_gaps_find_dimensions_without_a_subquestion():
    gaps = plan_gaps(_plan())
    assert "scalability" in gaps and "lock-in" in gaps and "time to launch" in gaps
    assert "cost" not in gaps and "privacy" not in gaps


def test_low_coverage_withholds_the_recommendation():
    cov = Coverage(covered=["sq3"], missing=["sq1", "sq2"], missing_questions=["cost?", "quality?"],
                   unplanned=["scalability"])
    report, withheld = enforce_recommendation_policy(_report(), cov, 0.5)
    assert withheld and report.recommendation.startswith(INSUFFICIENT_PREFIX)
    assert "cost?" in report.recommendation
    assert any("scalability" in x for x in report.limitations)


def test_enough_coverage_keeps_the_recommendation():
    cov = Coverage(covered=["sq1", "sq2"], missing=["sq3"], missing_questions=["x"])
    report, withheld = enforce_recommendation_policy(_report(), cov, 0.5)
    assert not withheld and report.recommendation.startswith("Build it")


# --------------------------------------------------------------------------- memory relevance

def _item(text: str) -> MemoryItem:
    now = utcnow()
    return MemoryItem(mem_id="m1", content_hash="h", claim=text, quote=text, claim_type="risk", specialist="risk",
                      subquestion_text="", source_url="https://x.example", source_host="x.example", source_title="t",
                      source_domain_type="official", source_published=None, source_quality=0.9, origin="offline",
                      page_hash="", support="supported", first_run_id="r", last_run_id="r", created_at=now,
                      last_verified_at=now, expires_at=now + timedelta(days=30))


def test_bank_regulation_memory_stays_out_of_an_ecommerce_question():
    bank = _item("Regulated financial institutions that use AI models to process customer financial data remain "
                 "fully responsible for that data and its privacy.")
    recsys = _item("Collaborative filtering recommenders need months of interaction data before they beat a "
                   "managed recommendation service.")
    assert "privacy" not in topic_terms(ECOMMERCE_Q) and "collaborative" in topic_terms(ECOMMERCE_Q)
    assert not shares_topic(ECOMMERCE_Q, bank)
    assert shares_topic(ECOMMERCE_Q, recsys)


# --------------------------------------------------------------------------- store + UI

def test_store_waits_for_a_busy_database_and_groups_writes(tmp_path):
    store = Store(tmp_path / "r.db")
    assert store._conn.execute("PRAGMA busy_timeout").fetchone()[0] >= 10_000
    with pytest.raises(sqlite3.OperationalError):
        store._exec_many([("CREATE TABLE t (x INTEGER)", ()), ("INSERT INTO missing VALUES (1)", ())])
    store._exec_many([("CREATE TABLE IF NOT EXISTS u (x INTEGER)", ()), ("INSERT INTO u VALUES (1)", ())])
    assert store._query("SELECT COUNT(*) n FROM u")[0]["n"] == 1


def test_clip_never_cuts_a_word_silently():
    assert c.clip("short", 20) == "short"
    out = c.clip("requires approximately three weeks of work to finish", 40)
    assert out.endswith("…") and "of w…" not in out and len(out) <= 41


def test_coverage_banner_lists_covered_and_missing():
    class Det:
        state = {"coverage": {"covered": ["sq3"], "missing": ["sq1", "sq2"], "missing_questions": [],
                              "unplanned_dimensions": ["scalability"]},
                 "plan": {"subquestions": [{"id": "sq1", "type": "market"}, {"id": "sq2", "type": "technical"},
                                           {"id": "sq3", "type": "risk"}]}}
        evidence = {}
    html = c.coverage_banner(Det())
    assert "1 of 3" in html and "Covered: sq3 (risk)" in html and "sq1 (market)" in html and "scalability" in html
    assert "rp-bad" in html
