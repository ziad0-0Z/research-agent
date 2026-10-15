"""Phase 9: golden sets are sound, routing/claim scoring is honest, recorded metrics come from real runs only."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from research_agent import evaluation as ev
from research_agent.config import PROJECT_ROOT
from research_agent.models import ClaimSupportVerdict
from research_agent.store import Store
from research_agent.tools.corpus import load_corpus
from research_agent.tools.quotes import verify_quote
from research_agent.workflows import pipeline

from test_workflows import make_fake_run_agent, settings  # noqa: F401  (fixture)

Q = "Should a 5-person startup use cloud LLM APIs or self-host an open-source model?"


def test_golden_questions_are_well_formed():
    qs = ev.load_golden("questions")["questions"]
    assert len(qs) >= 20 and len({q["id"] for q in qs}) == len(qs)
    for q in qs:
        assert q["route"] in ("standard", "sensitive")
        assert (q["route"] == "sensitive") == bool(q["domains"])
        assert set(q["required"]) <= set(ev.SPECIALISTS) and set(q["forbidden"]) <= set(ev.SPECIALISTS)
        assert not set(q["required"]) & set(q["forbidden"])


def test_every_claim_pair_quote_is_real_corpus_text():
    """A claim-support set built on invented quotes would measure nothing: every quote must be in its page."""
    corpus = load_corpus(str(PROJECT_ROOT / "data" / "corpus"))
    pages = {p.file[:3]: p.text for p in corpus.pages}
    pairs = ev.load_golden("claim_pairs")["pairs"]
    assert {p["expected"] for p in pairs} == {"supported", "partial", "not_supported"}
    for p in pairs:
        ok, score = verify_quote(p["quote"], pages[p["source"]])
        assert ok, (p["id"], score)


def test_code_router_meets_target_on_golden_set_including_keyword_traps():
    res = ev.routing_offline(ev.load_golden("questions")["questions"])
    assert res["accuracy"] == 1.0, [r for r in res["rows"] if not r["ok"]]
    traps = [r for r in res["rows"] if r["note"].startswith("trap")]
    assert len(traps) >= 4 and all(r["path"] == "standard" for r in traps)


@pytest.mark.parametrize("q", [
    "Our app stores PHI for small clinics: can we use a hosted LLM?",
    "Should our stock trading platform summarise portfolios with an LLM?",
    "Can an LLM help with medical diagnosis in rural hospitals?",
    "Should our firm use an LLM to prepare court filings?",
])
def test_tightened_router_still_flags_real_sensitive_wording(q):
    from research_agent.workflows.router import route_question
    assert route_question(q).path == "sensitive"


def test_specialist_scoring_requires_and_forbids():
    exp = {"required": ["risk"], "forbidden": ["market"]}
    assert ev.score_specialists(exp, {"risk", "technical"}) == (True, "ok")
    ok, why = ev.score_specialists(exp, {"risk", "market"})
    assert not ok and "unexpected market" in why
    ok, why = ev.score_specialists(exp, {"technical"})
    assert not ok and "missing risk" in why


def test_live_routing_uses_the_real_path_and_never_counts_errors_as_passes(settings):  # noqa: F811
    fake = make_fake_run_agent()
    calls = []

    async def run_agent(agent, prompt, ctx, max_turns):
        calls.append(agent.name)
        if "explode" in prompt:
            raise RuntimeError("503 overloaded")
        return await fake(agent, prompt, ctx, max_turns)

    qs = [{"id": "a", "question": Q, "route": "standard", "domains": [], "required": ["market"], "forbidden": []},
          {"id": "b", "question": "Should our bank process customer financial data with a self-hosted LLM?",
           "route": "sensitive", "domains": ["financial"], "required": ["risk"], "forbidden": ["market"]},
          {"id": "c", "question": "Please explode the planner for this test question", "route": "standard",
           "domains": [], "required": [], "forbidden": []}]
    res = asyncio.run(ev.routing_live(qs, settings, run_agent=run_agent))
    rows = {r["id"]: r for r in res["rows"]}
    assert rows["a"]["ok"] and rows["a"]["specialists"] == ["market", "risk", "technical"]
    assert rows["b"]["route_ok"] and not rows["b"]["specialists_ok"]  # the fake always uses market
    assert "error" in rows["c"] and not rows["c"]["ok"]
    assert res["n"] == 2 and res["errors"] == 1 and res["accuracy"] == 0.5  # error excluded, not scored as a pass
    assert calls[:3] == ["Input Guard", "Planner", "Research Orchestrator"]


def test_claim_support_eval_scores_exact_label_and_gate():
    async def judge(agent, prompt):
        return ClaimSupportVerdict(support="partial" if "always" in prompt else "supported", reason="fake judge", overreach="")

    pairs = [{"id": "x", "claim": "Self-hosting is always cheaper.", "quote": "q", "expected": "not_supported"},
             {"id": "y", "claim": "APIs cost money.", "quote": "q", "expected": "supported"}]
    res = asyncio.run(ev.claim_support_live(pairs, run_agent=judge))
    assert res["accuracy"] == 0.5 and res["gate_accuracy"] == 0.5


def test_recorded_metrics_come_from_stored_runs_only(settings):  # noqa: F811
    store = Store(":memory:")
    assert ev.fault_outcomes(store)["output_guardrail_catch_rate"] is None
    assert ev.parallel_stats(store)["median_speedup"] is None
    assert all(s.how in ("missing", "tests") for s in ev.scenario_coverage(store))

    asyncio.run(pipeline.run_research(Q, settings=settings, store=store, run_id="run_ok",
                                      run_agent=make_fake_run_agent(writer_fixes_on_revision=True)))
    asyncio.run(pipeline.run_research(Q, settings=settings, store=store, run_id="run_fake", demo_fault="fake_citation",
                                      run_agent=make_fake_run_agent(writer_fixes_on_revision=True)))
    f = ev.fault_outcomes(store)
    assert f["faults"]["fake_citation"] == {"runs": 1, "caught": 1, "run_ids": ["run_fake"]}
    assert f["output_guardrail_catch_rate"] == 1.0
    cov = {s.number: s for s in ev.scenario_coverage(store)}
    assert cov[9].run_id == "run_fake" and cov[4].how == "recorded"
    rec = ev.routing_from_runs(store, ev.load_golden("questions")["questions"])
    assert rec["n"] == 1 and rec["rows"][0]["run_id"] in ("run_ok", "run_fake")


def test_run_eval_summary_on_an_empty_store_reports_not_measured(tmp_path, monkeypatch):
    sys.path.insert(0, str(PROJECT_ROOT / "eval"))
    import run_eval
    monkeypatch.setattr(ev, "RESULTS_DIR", tmp_path)
    store = Store(":memory:")

    class S:
        sensitive_domains = ("financial", "medical", "legal")
        traces_dir = tmp_path

    summary = run_eval.build_summary(store, S())
    m = summary["metrics"]
    assert m["citation_validity"]["value"] is None and m["citation_validity"]["verdict"] == "not measured"
    assert m["output_guardrail_catch_rate"]["verdict"] == "not measured"
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["metrics"]
