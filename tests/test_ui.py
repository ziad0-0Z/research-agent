"""Phase 8 UI: the read models return real data or None (never invented numbers), HTML is escaped, the app builds."""
from __future__ import annotations

import asyncio

import pytest

from research_agent.store import Store
from research_agent.ui import components as c
from research_agent.ui import data as d
from research_agent.workflows import pipeline

from test_workflows import make_fake_run_agent, settings  # noqa: F401  (fixture)

Q = "Should a 5-person startup use cloud LLM APIs or self-host an open-source model?"


def test_empty_store_gives_empty_states_not_numbers():
    store = Store(":memory:")
    m = d.overview_metrics(store)
    assert m["runs"] == 0 and m["completed"] == 0 and m["pending_reviews"] == 0
    g = d.grounding_metrics(store)
    assert g["citation_validity"] is None and g["grounding_rate"] is None  # unmeasured, not "100%"
    assert d.run_rows(store) == [] and d.evidence_rows(store) == [] and d.memory_rows(store) == []


def test_read_models_follow_a_real_pipeline_run(settings):  # noqa: F811
    store = Store(":memory:")
    result = asyncio.run(pipeline.run_research(Q, settings=settings, store=store, run_id="run_ui_test_1",
                                               run_agent=make_fake_run_agent(writer_fixes_on_revision=True)))
    assert result.run_id == "run_ui_test_1"  # the UI chooses the id so it can follow the run
    rows = d.run_rows(store)
    assert [r.run_id for r in rows] == ["run_ui_test_1"] and rows[0].evidence > 0
    det = d.run_detail(store, "run_ui_test_1")
    assert det.report and det.evidence and det.state["trace_ids"] is not None
    ev = next(iter(det.evidence.values()))
    page = d.quote_in_page(store, det.run_id, ev)
    assert page["still_matches"] and page["match"]
    md = d.report_markdown(det)
    assert det.report["title"] in md
    if det.status != "completed":
        assert "NOT RELEASED" in md
    html = c.report_body(det)
    assert "Evidence appendix" in html and 'href="#ev-' in html
    assert d.run_rows(store, status="completed" if det.status == "completed" else "paused_for_review")
    assert d.run_rows(store, query="no such words anywhere") == []


def test_untrusted_text_is_escaped():
    item = {"claim": "<script>alert(1)</script>", "quote": "<img src=x onerror=alert(1)>", "url": "https://a.example/x",
            "source_title": "<b>t</b>", "origin": "offline", "domain_type": "blog", "quality": 0.5, "verified": True,
            "support": "supported", "status": "active", "injection": True, "id": "ev1", "run_id": "r",
            "subquestion_id": "sq1", "retrieved_at": "2026-10-14T10:00:00+00:00", "memory": None}
    html = c.evidence_chain(item, {"has_text": True, "still_matches": True, "before": "<x>", "match": "<y>", "after": ""})
    assert "<script>" not in html and "<img" not in html and "&lt;script&gt;" in html
    assert "Injection detected" in html


def test_offline_corpus_sources_are_not_presented_as_live_links():
    assert "href" not in c.source_link("https://northwind.example/a", "T", "offline")
    assert 'href="https://arxiv.org/abs/1"' in c.source_link("https://arxiv.org/abs/1", "T", "tavily")


def test_rail_marks_progress_from_events():
    assert {s for *_, s in c.stage_states([], running=False)} == {"pending"}
    states = {k: s for k, _, _, s in c.stage_states(["input_checked", "routed", "planned"], running=True)}
    assert states["guard"] == states["plan"] == "done"
    assert states["memory"] == "skipped"          # no memory event before the planner: shown as "not run"
    assert states["delegate"] == "active" and states["outcome"] == "pending"


def test_blocked_run_shows_a_stopped_outcome():
    states = {k: s for k, _, _, s in c.stage_states(["input_checked", "blocked"], running=False)}
    assert states["outcome"] == "failed" and states["plan"] == "skipped"


def test_pytest_summary_parser():
    assert d.parse_pytest_summary("3 failed, 310 passed, 1 skipped in 5.2s") == {"failed": 3, "passed": 310, "skipped": 1}


def test_app_builds(settings):  # noqa: F811
    from research_agent.ui.app import build_app
    demo = build_app(settings)
    assert demo is not None
