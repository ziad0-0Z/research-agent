"""Router, parallel executor, confidence and the full pipeline (offline, fake LLM)."""
import asyncio
import re
import time

import pytest

from research_agent import config as cfg
import json

from research_agent.models import (
    GuardrailVerdict, Assignment, CitationCheck, Contradiction, CriticReview, EvidenceAnalysis, Finding, FindingCheck,
    OrchestrationDecision, ResearcherOutput, ResearchPlan, ResearchReport, SubQuestion, SubquestionResult,
)
from research_agent.store import Store
from research_agent.tools import evidence, web
from research_agent.workflows import pipeline
from research_agent.workflows.parallel import WorkerTask, run_parallel, timeline
from research_agent.workflows.router import route_question, validate_assignments


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def settings(monkeypatch, tmp_path):
    monkeypatch.setenv("SEARCH_MODE", "offline")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-never-used")
    from research_agent import llm
    llm.reset_model()
    return cfg.load_settings()


def _sq(i, type_, q="What does it cost to self-host an LLM for a small team?"):
    return SubQuestion(id=f"sq{i}", question=q, type=type_, priority=1,
                       success_criteria="Concrete numbers from sources", search_queries=["self hosting llm cost"])


def _plan(*types):
    return ResearchPlan(main_question="Should a 5-person startup use cloud LLM APIs or self-host?",
                        decision_context="5-person startup, limited budget",
                        subquestions=[_sq(i + 1, t) for i, t in enumerate(types)])


# ---------------------------------------------------------------- router (code-orchestrated)

@pytest.mark.parametrize("q,domains", [
    ("Should our bank process customer financial data with a self-hosted LLM?", ["financial"]),
    ("Can we use a cloud API to summarise patient medical records?", ["medical"]),
    ("Should our law firm use an LLM to draft litigation documents?", ["legal"]),
])
def test_router_flags_sensitive_domains(q, domains):
    r = route_question(q)
    assert r.path == "sensitive" and r.requires_review and r.domains == domains and r.matched_terms


@pytest.mark.parametrize("q", [
    "Should a 5-person startup use cloud LLM APIs or self-host an open-source model?",
    "pgvector or a dedicated vector database for a RAG product?",
    "What are the security risks of self-hosting an LLM?",
    "How do embankment sensors work in civil engineering?",  # 'bank' inside a word must not match
])
def test_router_standard_questions(q):
    assert route_question(q).path == "standard"


def test_router_uses_guard_domain():
    r = route_question("Should we adopt this tool?", guard_domain="medical")
    assert r.requires_review and r.domains == ["medical"]


def test_validate_assignments_fixes_llm_mistakes():
    plan = _plan("technical", "market", "risk")
    decision = OrchestrationDecision(assignments=[
        Assignment(specialist="market", subquestion_ids=["sq1", "sq2", "sq9"]),
        Assignment(specialist="technical", subquestion_ids=["sq1"]),
    ], depth="standard", rationale="cost focus")
    out = validate_assignments(plan, decision)
    assert {k: [s.id for s in v] for k, v in out.by_specialist.items()} == {"market": ["sq1", "sq2"], "risk": ["sq3"]}
    assert any("sq9" in n for n in out.notes)                 # unknown id dropped
    assert any("assigned twice" in n for n in out.notes)      # duplicate kept once
    assert any("sq3 was unassigned" in n for n in out.notes)  # missing one assigned by type


# ---------------------------------------------------------------- parallel executor

def test_parallel_isolates_failures_and_timeouts(settings):
    from research_agent.context import AppContext
    ctx = AppContext(settings=settings, store=Store(":memory:"), question="test question here")
    run(web.scrape_page(ctx, "https://applied-systems-institute.example/reports/llm-tco-small-teams"))

    async def ok():
        await asyncio.sleep(0.2)
        evidence.save_evidence(ctx, source_id="src1", subquestion_id="sq1", claim="Labour dominates for small teams.",
                               quote="For small teams, labour dominates the comparison.")
        return ResearcherOutput(results=[SubquestionResult(subquestion_id="sq1", summary="Labour dominates [ev1].",
                                                           evidence_ids=["ev1"], open_issues="")])

    async def crash():
        await asyncio.sleep(0.1)
        raise RuntimeError("model returned invalid JSON")

    async def slow():
        await asyncio.sleep(5)

    tasks = [WorkerTask("market", ["sq1"], ok), WorkerTask("technical", ["sq2"], crash),
             WorkerTask("risk", ["sq3"], slow)]
    start = time.time()
    results = run(run_parallel(ctx, tasks, timeout=0.5))
    wall = time.time() - start
    status = {w.specialist: w.status for w in results}
    assert status == {"market": "success", "technical": "failed", "risk": "timeout"}
    assert wall < 1.5  # ran concurrently, not 0.2 + 0.1 + 0.5 in sequence per worker
    assert results[0].evidence_ids == ["ev1"] and "Labour" in results[0].summaries["sq1"]
    assert "invalid JSON" in results[1].error
    tl = timeline(results)
    assert tl["speedup"] > 1.0 and len(tl["bars"]) == 3


# ---------------------------------------------------------------- end-to-end pipeline (fake LLM, real tools)

def _sentences(text, n=2):
    out = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if 60 < len(s.strip()) < 300]
    return out[:n]


UNSUPPORTED_MARKERS = ("always cheaper",)


def fake_critic_review(prompt, calls=None):
    """Plays the Critic: statements with an unsupported marker are not_supported, the rest supported."""
    items = json.loads(prompt.split("Findings to check (JSON):\n", 1)[1])
    if calls is not None:
        calls.append([i["finding"] for i in items])
    checks = []
    for item in items:
        bad = any(m in item["statement"].lower() for m in UNSUPPORTED_MARKERS)
        level = "not_supported" if bad else "supported"
        checks.append(FindingCheck(
            finding=item["finding"],
            citations=[CitationCheck(evidence_id=e["id"], support=level) for e in item["cited_evidence"]],
            support=level, problem="the quote says the opposite" if bad else "",
            better_evidence_ids=[e["id"] for e in item["other_relevant_evidence"][:1]] if bad else []))
    return CriticReview(findings=checks, contradictions_ignored=[], other_issues=[])


def fake_guard(prompt):
    """Plays the Guard Agent: blocks off-scope chat, labels the domain."""
    q = prompt.split("<user_request>", 1)[1].lower()
    if "poem" in q:
        return GuardrailVerdict(verdict="block", category="off_scope", domain="other", reason="creative writing")
    domain = "financial" if "bank" in q else "technology"
    return GuardrailVerdict(verdict="allow", category="valid_research", domain=domain, reason="research question")


def make_fake_run_agent(break_worker=None, writer_fixes_on_revision=False, critic_calls=None, guard_error=False,
                        impact="medium", writer_calls=None):
    async def fake(agent, prompt, ctx, max_turns):
        name = agent.name
        if name == "Input Guard":
            if guard_error:
                raise RuntimeError("503 model overloaded")
            return fake_guard(prompt)
        if name == "Critic":
            return fake_critic_review(prompt, critic_calls)
        if name == "Planner":
            return _plan("market", "technical", "risk")
        if name == "Research Orchestrator":
            return OrchestrationDecision(assignments=[
                Assignment(specialist="market", subquestion_ids=["sq1"]),
                Assignment(specialist="technical", subquestion_ids=["sq2"]),
                Assignment(specialist="risk", subquestion_ids=["sq3"]),
            ], depth="quick", rationale="one specialist per subquestion")
        if name.endswith("Researcher"):
            kind = name.split()[0].lower()
            if kind == break_worker:
                raise RuntimeError("simulated model failure")
            sq_id = {"market": "sq1", "technical": "sq2", "risk": "sq3"}[kind]
            query = {"market": "api pricing cost small team", "technical": "open model quality benchmark",
                     "risk": "data retention privacy provider"}[kind]
            found = await web.search_web(ctx, query, 2)
            saved = []
            for r in found["results"][:2]:
                page = await web.scrape_page(ctx, r["url"])
                text = ctx.store.get_source_text(ctx.run_id, page["source_id"])
                for sentence in _sentences(text, 1):
                    out = evidence.save_evidence(ctx, source_id=page["source_id"], subquestion_id=sq_id,
                                                 claim=f"The source states: {sentence[:120]}", quote=sentence)
                    if out["saved"]:
                        saved.append(out["evidence_id"])
            return ResearcherOutput(results=[SubquestionResult(
                subquestion_id=sq_id, summary=f"Found {len(saved)} items.", evidence_ids=saved, open_issues="")])
        if name == "Evidence Analyst":
            ids = re.findall(r'"id": "(ev\d+)"', prompt)
            return EvidenceAnalysis(agreements=[f"Sources agree on costs [{ids[0]}]"],
                                    contradictions=[Contradiction(topic="cost", evidence_ids_a=[ids[0]],
                                                                  evidence_ids_b=[ids[1]],
                                                                  explanation="different volumes")],
                                    gaps=[])
        if name == "Writer":
            if writer_calls is not None:
                writer_calls.append(prompt)
            ids = re.findall(r'"id": "(ev\d+)"', prompt)
            fixed = writer_fixes_on_revision and "Revision instructions" in prompt
            bad = [] if fixed else [Finding(statement="Invented citation.", evidence_ids=["id:ev999"], strength="weak")]
            return ResearchReport(
                title="Cloud APIs vs self-hosting for a 5-person startup",
                executive_summary="Hosted APIs are the better starting point for a small team at modest volume.",
                findings=[Finding(statement=f"Finding backed by {eid}.", evidence_ids=[eid], strength="moderate")
                          for eid in ids[:4]] + bad,
                contradictions=["Sources differ on cost depending on volume."],
                limitations=["Offline synthetic corpus."],
                recommendation="Start with hosted APIs and revisit at higher volume.",
                impact=impact)
        raise AssertionError(f"unexpected agent {name}")
    return fake


def test_pipeline_end_to_end_offline(settings):
    events = []
    result = run(pipeline.run_research(
        "Should a 5-person startup use cloud LLM APIs or self-host an open-source model?",
        settings=settings, store=Store(":memory:"), run_agent=make_fake_run_agent(),
        on_event=lambda stage, data: events.append(stage)))

    assert result.route.path == "standard"
    assert [w.status for w in result.workers] == ["success", "success", "success"]
    # The fake Writer keeps inventing ev999: REVISE, then the same blocking issue again → FAIL (no progress).
    assert events == ["input_checked", "routed", "planned", "delegated", "research_started", "research_done",
                      "analysed", "coverage", "drafted", "critic_round", "revised", "critic_round", "critic_done", "checked",
                      "output_checked", "rejected"]
    # The Critic could not fix ev999, so the output guardrail refuses to release the report.
    assert result.status == "rejected_by_guardrail" and result.final_markdown is None
    assert any(i.check == "citations" and "ev999" in i.message for i in result.output_check.issues)
    assert len(result.analysis.contradictions) == 1
    assert result.invalid_citations == [{"evidence_id": "ev999",
                                         "reason": "evidence id does not exist in this run"}]
    assert result.critic.verdict == "FAIL" and result.critic.stop_reason == "no_progress"
    assert any("Critic FAIL" in r for r in result.ctx.review_reasons)
    assert 0 < result.confidence["score"] <= 1 and result.confidence["covered_subquestions"] == "3/3"
    assert "DRAFT" in result.draft_markdown and "## Evidence appendix" in result.draft_markdown
    assert result.timeline["bars"] and result.timeline["speedup"] > 0


def test_critic_loop_fixes_invented_citation(settings):
    events = []
    result = run(pipeline.run_research(
        "Should a 5-person startup use cloud LLM APIs or self-host an open-source model?",
        settings=settings, store=Store(":memory:"), run_agent=make_fake_run_agent(writer_fixes_on_revision=True),
        on_event=lambda stage, data: events.append((stage, data))))
    rounds = [d for st, d in events if st == "critic_round"]
    assert [r["verdict"] for r in rounds] == ["REVISE", "PASS"]
    assert any("ev999" in i for i in rounds[0]["issues"])
    assert result.invalid_citations == [] and "ev999" not in result.report.cited_evidence_ids()
    assert result.critic.verdict == "PASS" and "Critic verdict **PASS**" in result.draft_markdown


def test_planted_unsupported_claim_is_caught_and_fixed(settings):
    """A real quote cited for a claim it does not support → REVISE → PASS."""
    events = []
    result = run(pipeline.run_research(
        "Should a 5-person startup use cloud LLM APIs or self-host an open-source model?",
        settings=settings, store=Store(":memory:"), run_agent=make_fake_run_agent(writer_fixes_on_revision=True),
        on_event=lambda stage, data: events.append((stage, data)), demo_fault="unsupported_claim"))
    first = result.critic.rounds[0]
    planted_n = len(first.report.findings)  # the planted finding was appended last
    assert first.verdict == "REVISE"
    assert first.support[planted_n] == "not_supported"
    assert any(i.kind == "unsupported" and i.finding == planted_n and i.severity == "blocking" for i in first.issues)
    assert any("Candidates from retrieval" in x for x in first.critique.revision_instructions)
    assert result.critic.verdict == "PASS" and len(result.critic.rounds) == 2
    assert all("always cheaper" not in f.statement for f in result.report.findings)
    # The planted citation was a real, verified id: only the support check could catch it.
    assert result.invalid_citations == []


def test_critic_does_not_rejudge_unchanged_findings(settings):
    calls = []
    run(pipeline.run_research(
        "Should a 5-person startup use cloud LLM APIs or self-host an open-source model?",
        settings=settings, store=Store(":memory:"),
        run_agent=make_fake_run_agent(writer_fixes_on_revision=True, critic_calls=calls)))
    # Round 2: the Writer only removed the bad finding; the four good findings are reused, so no
    # second Critic call is needed at all.
    assert len(calls) == 1 and calls[0] == [1, 2, 3, 4]


def test_evidence_id_formatting_is_normalized_but_not_guessed():
    f = Finding(statement="Some statement here.", evidence_ids=["id:ev1", "[EV2]", "ev_3", "ev1", "v10"],
                strength="weak")
    assert f.evidence_ids == ["ev1", "ev2", "ev3", "v10"]


def test_pipeline_survives_a_failed_worker(settings):
    result = run(pipeline.run_research(
        "Should a 5-person startup use cloud LLM APIs or self-host an open-source model?",
        settings=settings, store=Store(":memory:"), run_agent=make_fake_run_agent(break_worker="technical")))
    status = {w.specialist: w.status for w in result.workers}
    assert status["technical"] == "failed" and status["market"] == "success"
    assert result.confidence["worker_factor"] == 0.85
    assert result.confidence["covered_subquestions"] == "2/3"


def test_pipeline_flags_sensitive_question(settings):
    result = run(pipeline.run_research(
        "Should our bank process customer financial data with a self-hosted LLM?",
        settings=settings, store=Store(":memory:"), run_agent=make_fake_run_agent()))
    assert result.route.requires_review and result.ctx.requires_review
    assert "financial" in result.ctx.review_reasons[0]
    assert result.input_check.domain == "financial"


def test_pipeline_rejects_empty_question(settings):
    with pytest.raises(pipeline.PipelineError, match="too short"):
        run(pipeline.run_research("hi", settings=settings, store=Store(":memory:"),
                                  run_agent=make_fake_run_agent()))


def test_worker_that_covers_everything_succeeds_despite_turn_limit(tmp_path, monkeypatch):
    import asyncio
    from research_agent import config as cfg
    from research_agent.context import AppContext
    from research_agent.store import Store
    from research_agent.tools import evidence, web
    from research_agent.workflows.parallel import WorkerTask, run_parallel
    monkeypatch.setenv("SEARCH_MODE", "offline")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    ctx = AppContext(settings=cfg.load_settings(), store=Store(":memory:"), question="Cloud APIs or self-hosting?")

    async def covers_then_hits_limit():
        await web.scrape_page(ctx, "https://applied-systems-institute.example/reports/llm-tco-small-teams")
        evidence.save_evidence(ctx, source_id="src1", subquestion_id="sq1", claim="APIs had lower TCO for small teams.",
                               quote="Below that threshold, hosted APIs had the lower TCO in every scenario we modelled.")
        raise RuntimeError("Max turns (10) exceeded")

    [w] = asyncio.run(run_parallel(ctx, [WorkerTask("market", ["sq1"], covers_then_hits_limit)], timeout=30))
    assert w.status == "success" and "Max turns" in w.error
