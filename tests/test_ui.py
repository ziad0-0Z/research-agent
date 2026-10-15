"""UI: the read models return real data or None (never invented numbers), HTML is escaped, the app builds."""
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


def test_attention_list_finds_an_old_failed_run_behind_many_newer_ones():
    """Regression: the limit was applied before the status filter, so an old failed run never showed up."""
    import time as _t
    store = Store(":memory:")
    store.create_run("run_old_failed", "An old question that crashed in the writer stage")
    store.update_run("run_old_failed", status="failed", state={"error": {"stage": "writer"}})
    _t.sleep(0.01)
    for i in range(12):
        store.create_run(f"run_ok_{i}", f"Released question number {i}")
        store.update_run(f"run_ok_{i}", status="completed", state={})
    rows = d.run_rows(store, status="attention", limit=8)
    assert [r.run_id for r in rows] == ["run_old_failed"]
    assert d.overview_metrics(store)["needs_attention"] == 1


def test_planted_demo_fault_is_labelled_not_raised_as_an_alarm():
    store = Store(":memory:")
    store.create_run("run_demo", "Should a 5-person startup use cloud LLM APIs or self-host?")
    store.update_run("run_demo", status="rejected_by_guardrail", state={"demo_fault": "fake_citation"})
    store.create_run("run_real", "Should a 5-person startup use cloud LLM APIs or self-host?")
    store.update_run("run_real", status="rejected_by_guardrail", state={})
    assert [r.run_id for r in d.run_rows(store, status="attention")] == ["run_real"]
    m = d.overview_metrics(store)
    assert m["needs_attention"] == 1 and m["demo_faults_stopped"] == 1
    html = c.run_table_rows(d.run_rows(store))
    assert any("Demo: fake citation" in cell for row in html for cell in row)


def test_evidence_graph_is_built_from_stored_links_only():
    from research_agent.ui import evidence_graph as eg
    report = {"findings": [{"statement": "APIs are cheaper at low volume", "evidence_ids": ["ev1", "ev2"], "strength": "strong"},
                           {"statement": "<script>x</script>", "evidence_ids": ["ev3", "ev99"], "strength": "weak"}]}
    evidence = {"ev1": {"claim": "c1", "quote": "q1", "support": "supported", "source_id": "src1"},
                "ev2": {"claim": "c2", "quote": "q2", "support": "partial", "source_id": "src2"},
                "ev3": {"claim": "c3", "quote": "q3", "support": "not_supported", "source_id": "src1"},
                "ev4": {"claim": "c4", "quote": "q4", "support": "supported", "source_id": "src2"}}
    sources = {"src1": {"title": "Page one", "quality_score": 0.9}, "src2": {"title": "Page two", "quality_score": 0.5}}
    analysis = {"contradictions": [{"topic": "cost", "evidence_ids_a": ["ev1"], "evidence_ids_b": ["ev3", "ev404"]}]}
    g = eg.build_graph(report, evidence, sources, analysis)
    assert [e["id"] for e in g.evidence] == ["ev1", "ev2", "ev3"]  # ev99 does not exist; ev4 is not cited
    assert g.uncited == 1 and g.contradictions == [("ev1", "ev3", "cost")]  # unknown id ev404 dropped
    st = g.stats()
    assert st["multi_source_findings"] == 1 and st["sources"] == 2  # F2 cites one real page only
    svg = eg.render_svg(g)
    assert svg.startswith("<svg") and "<script>" not in svg and "&lt;script&gt;" in svg
    assert 'stroke-dasharray="6 4"' in svg and 'stroke-dasharray="2 4"' in svg  # partial / not supported by line style
    assert "No graph" in eg.graph_html(None, {}, {}, None)
