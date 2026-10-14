"""Writer ⇄ Critic loop: deterministic checks, strength rules, review cleaning, stop conditions."""
import asyncio

import pytest

from research_agent import config as cfg
from research_agent.context import AppContext
from research_agent.models import (
    CitationCheck, Contradiction, CriticReview, EvidenceAnalysis, Finding, FindingCheck, ResearchPlan,
    ResearchReport, SubQuestion, WorkerFindings,
)
from research_agent.store import Store
from research_agent.tools import evidence, web
from research_agent.workflows import eval_loop
from research_agent.workflows.eval_loop import apply_strength_rules, code_checks, writer_critic_loop

P01 = "https://northwind-tech-review.example/2026/08/llm-api-pricing-small-teams"
P03 = "https://applied-systems-institute.example/reports/llm-tco-small-teams"
P09 = "https://builders-forum.example/t/self-hosted-llm-horror-stories"


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def ctx(monkeypatch, tmp_path):
    monkeypatch.setenv("SEARCH_MODE", "offline")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    c = AppContext(settings=cfg.load_settings(), store=Store(":memory:"), question="Cloud APIs or self-hosting?")
    for url in (P03, P01, P09):
        out = run(web.scrape_page(c, url))
        assert "source_id" in out, out
    evidence.save_evidence(c, source_id="src1", subquestion_id="sq1", claim="APIs had lower TCO below that volume.",
                           quote="Below that threshold, hosted APIs had the lower TCO in every scenario we modelled.")
    evidence.save_evidence(c, source_id="src2", subquestion_id="sq1", claim="Output tokens cost 3-5x input tokens.",
                           quote="Output tokens typically cost three to five times more than input tokens.")
    evidence.save_evidence(c, source_id="src3", subquestion_id="sq2", claim="A driver update caused a 6-hour outage.",
                           quote="A routine driver update broke the serving container on a Friday night and the bot "
                                 "was down for six hours.")
    return c


def _plan():
    sq = lambda i, t: SubQuestion(id=f"sq{i}", question="What does each option cost a small team?", type=t,
                                  priority=1, success_criteria="Concrete numbers", search_queries=["llm cost"])
    return ResearchPlan(main_question="Should a 5-person startup use APIs or self-host?",
                        decision_context="5-person startup", subquestions=[sq(1, "market"), sq(2, "risk")])


def _report(findings, contradictions=("[ev1] vs [ev3]: different scope",), limitations=("Synthetic corpus.",)):
    return ResearchReport(title="APIs vs self-hosting", executive_summary="APIs are the better start for a small team.",
                          findings=findings, contradictions=list(contradictions), limitations=list(limitations),
                          recommendation="Start with APIs and revisit at higher volume.", impact="medium")


def _f(statement, ids, strength="moderate"):
    return Finding(statement=statement, evidence_ids=ids, strength=strength)


def _worker(status="success"):
    return WorkerFindings(specialist="market", status=status, subquestion_ids=["sq1"], started_at=0, finished_at=1)


NO_ANALYSIS = EvidenceAnalysis(agreements=[], contradictions=[], gaps=[])
ONE_CONTRA = EvidenceAnalysis(agreements=[], gaps=[], contradictions=[
    Contradiction(topic="cost", evidence_ids_a=["ev1"], evidence_ids_b=["ev3"], explanation="scope")])


# ---------------------------------------------------------------- deterministic checks

def test_code_checks_clean_report_has_no_blocking_issue(ctx):
    report = _report([_f("APIs had lower TCO below several billion tokens.", ["ev1"]),
                      _f("Self-hosting failures caused long outages.", ["ev3"])])
    issues = code_checks(ctx, _plan(), ONE_CONTRA, [_worker()], report)
    assert [i for i in issues if i.severity == "blocking"] == []


def test_code_checks_flag_invented_citation(ctx):
    issues = code_checks(ctx, _plan(), NO_ANALYSIS, [_worker()], _report([_f("Something about cost here.", ["ev42"])]))
    assert any(i.kind == "invalid_citation" and i.severity == "blocking" and "ev42" in i.message for i in issues)


def test_code_checks_flag_ignored_contradiction(ctx):
    report = _report([_f("APIs had lower TCO below several billion tokens.", ["ev1"])], contradictions=())
    issues = code_checks(ctx, _plan(), ONE_CONTRA, [_worker()], report)
    blocking = [i for i in issues if i.severity == "blocking"]
    assert [i.kind for i in blocking] == ["contradiction_ignored"] and "ev1 vs ev3" in blocking[0].instruction


def test_code_checks_flag_hidden_research_failure(ctx):
    report = _report([_f("APIs had lower TCO below several billion tokens.", ["ev1"])], limitations=())
    issues = code_checks(ctx, _plan(), NO_ANALYSIS, [_worker("timeout")], report)
    assert any(i.kind == "gaps_hidden" and i.severity == "blocking" and "timeout" in i.message for i in issues)


def test_code_checks_flag_unused_subquestion_as_advisory(ctx):
    report = _report([_f("APIs had lower TCO below several billion tokens.", ["ev1"])])
    issues = code_checks(ctx, _plan(), NO_ANALYSIS, [_worker()], report)
    cov = [i for i in issues if i.kind == "coverage"]
    assert len(cov) == 1 and cov[0].severity == "advisory" and "sq2" in cov[0].message


# ---------------------------------------------------------------- strength rules (code decides)

def test_partial_support_is_never_strong(ctx):
    report = _report([_f("APIs and output tokens.", ["ev1", "ev2"], "strong")])
    fixed, fixes = apply_strength_rules(ctx, report, {1: "partial"})
    assert fixed.findings[0].strength == "moderate" and "partial" in fixes[0]


def test_low_quality_only_evidence_is_weak(ctx):
    report = _report([_f("Self-hosting causes outages.", ["ev3"], "moderate")])  # forum source
    fixed, fixes = apply_strength_rules(ctx, report, {1: "supported"})
    assert fixed.findings[0].strength == "weak" and "low-quality" in fixes[0]


def test_strong_needs_two_sources_or_one_high_quality(ctx):
    two = _report([_f("APIs are cheaper at low volume.", ["ev1", "ev2"], "strong")])
    assert apply_strength_rules(ctx, two, {1: "supported"})[0].findings[0].strength == "strong"


# ---------------------------------------------------------------- critic output is checked in code

def test_clean_review_drops_unknown_ids_and_caps_support():
    report = _report([_f("APIs had lower TCO below several billion tokens.", ["ev1"])])
    review = CriticReview(findings=[
        FindingCheck(finding=1, citations=[CitationCheck(evidence_id="ev1", support="partial"),
                                           CitationCheck(evidence_id="ev9", support="supported")],
                     support="supported", problem="", better_evidence_ids=["ev2", "ev77"]),
        FindingCheck(finding=5, citations=[], support="supported", problem="", better_evidence_ids=[]),
    ], contradictions_ignored=[], other_issues=[])
    out = eval_loop._clean_review(review, report, [1], {1: {"ev2"}})
    assert list(out) == [1]                                       # finding 5 was never asked about
    assert [c.evidence_id for c in out[1].citations] == ["ev1"]   # ev9 is not cited by finding 1
    assert out[1].better_evidence_ids == ["ev2"]                  # ev77 was never retrieved
    assert out[1].support == "partial"                            # can't beat its best citation


# ---------------------------------------------------------------- the loop: stop conditions

def _critic_marking(bad_word):
    async def run_critic(agent, prompt):
        import json
        items = json.loads(prompt.split("Findings to check (JSON):\n", 1)[1])
        out = []
        for it in items:
            lvl = "not_supported" if bad_word in it["statement"] else "supported"
            out.append(FindingCheck(finding=it["finding"], support=lvl, problem="overreach" if lvl != "supported" else "",
                                    citations=[CitationCheck(evidence_id=e["id"], support=lvl)
                                               for e in it["cited_evidence"]], better_evidence_ids=[]))
        return CriticReview(findings=out, contradictions_ignored=[], other_issues=[])
    return run_critic


def _loop(ctx, draft, run_writer, run_critic, max_rounds=3, events=None):
    return run(writer_critic_loop(
        ctx, _plan(), NO_ANALYSIS, [_worker()], "writer prompt", draft, run_writer=run_writer,
        run_critic=run_critic, writer_agent=lambda: None, critic_agent=lambda: None, max_rounds=max_rounds,
        on_event=(lambda s, d: events.append((s, d))) if events is not None else None))


def test_loop_stops_at_max_rounds_and_keeps_best_draft(ctx):
    n = {"v": 0}

    async def stubborn_writer(agent, prompt):  # changes wording every time, never fixes the problem
        n["v"] += 1
        return _report([_f(f"APIs had lower TCO (v{n['v']}).", ["ev1"]),
                        _f(f"Self-hosting is BAD in every case v{n['v']}.", ["ev3"])])

    draft = _report([_f("APIs had lower TCO (v0).", ["ev1"]), _f("Self-hosting is BAD in every case v0.", ["ev3"])])
    result = _loop(ctx, draft, stubborn_writer, _critic_marking("BAD"), max_rounds=3)
    assert [r.verdict for r in result.rounds] == ["REVISE", "REVISE", "FAIL"]
    assert result.verdict == "FAIL" and result.stop_reason == "max_rounds" and n["v"] == 2
    assert result.best_round == 3  # all tie on 1 blocking issue → the latest draft is kept


def test_loop_respects_hard_round_cap(ctx):
    async def writer(agent, prompt):
        return _report([_f(f"Self-hosting is BAD {len(prompt)}.", ["ev3"])])
    result = _loop(ctx, _report([_f("Self-hosting is BAD.", ["ev3"])]), writer, _critic_marking("BAD"), max_rounds=50)
    assert len(result.rounds) <= eval_loop.MAX_ROUNDS_CAP


def test_loop_survives_critic_error_and_writer_error(ctx):
    async def broken_critic(agent, prompt):
        raise RuntimeError("model returned invalid JSON")

    draft = _report([_f("APIs had lower TCO below several billion tokens.", ["ev1"])])
    result = _loop(ctx, draft, None, broken_critic)
    assert result.verdict == "PASS"  # no blocking issue from code; support is reported as unverified
    assert any("Critic review failed" in i.message for i in result.rounds[0].issues)

    async def broken_writer(agent, prompt):
        raise RuntimeError("503 overloaded")

    events = []
    bad = _report([_f("Something cited wrongly here.", ["ev42"])])
    result = _loop(ctx, bad, broken_writer, _critic_marking("BAD"), events=events)
    assert result.verdict == "FAIL" and result.stop_reason == "writer_error"
    assert result.report == bad and [s for s, _ in events] == ["critic_round", "revision_failed"]


def test_loop_stores_claim_support_on_evidence(ctx):
    async def writer(agent, prompt):
        return _report([_f("APIs had lower TCO below several billion tokens.", ["ev1"])])
    draft = _report([_f("APIs had lower TCO below several billion tokens.", ["ev1"]),
                     _f("Self-hosting is BAD always.", ["ev3"])])
    result = _loop(ctx, draft, writer, _critic_marking("BAD"))
    assert result.verdict == "PASS"
    assert ctx.store.get_evidence(ctx.run_id, "ev1").support == "supported"
