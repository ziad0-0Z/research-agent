"""Run the full research pipeline (Phases 2-3) from the command line.

    python scripts/run_research.py                       # primary demo question, offline corpus
    python scripts/run_research.py --question secondary   # pgvector vs vector DB
    python scripts/run_research.py --question sensitive   # bank question -> review path
    python scripts/run_research.py --question "Your own question?"
    python scripts/run_research.py --live-search          # Tavily -> DuckDuckGo -> offline

The draft report is saved to samples/<run_id>.md.
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError  # noqa: E402

from research_agent.config import ConfigError, load_settings  # noqa: E402
from research_agent.llm import get_model  # noqa: E402
from research_agent.store import Store  # noqa: E402
from research_agent.tracing import configure_tracing  # noqa: E402
from research_agent.workflows.pipeline import PipelineError, RunResult, run_research  # noqa: E402

QUESTIONS = {
    "primary": "Should a 5-person startup use cloud LLM APIs or self-host an open-source model?",
    "secondary": "Should we use PostgreSQL with pgvector or a dedicated vector database for a RAG product?",
    "sensitive": "Should our bank process customer financial data with a self-hosted LLM?",
    "risk": "What are the security risks of self-hosting an LLM?",
}
STATUS_ICON = {"success": "✅", "partial": "🟡", "failed": "❌", "timeout": "⏱️"}
T0 = time.time()
settings_rpm = 0


def on_event(stage: str, data: dict) -> None:
    t = f"[{time.time() - T0:6.1f}s]"
    if stage == "routed":
        extra = f" — {', '.join(data['domains'])} ({', '.join(data['matched'])})" if data["domains"] else ""
        print(f"{t} 🐍 Router: {data['path']}{extra}")
    elif stage == "planned":
        print(f"{t} 🤖 Planner: {len(data['subquestions'])} subquestions")
        for sq in data["subquestions"]:
            print(f"           {sq['id']} [{sq['type']}, p{sq['priority']}] {sq['question']}")
    elif stage == "delegated":
        print(f"{t} 🤖 Orchestrator: depth={data['depth']} — {data['rationale']}")
        for k, v in data["assignments"].items():
            print(f"           {k:9s} → {', '.join(v)}")
        for note in data["corrections"]:
            print(f"           🐍 corrected: {note}")
    elif stage == "research_started":
        print(f"{t} ⚡ {len(data['workers'])} specialists researching in parallel: {', '.join(data['workers'])} ...")
    elif stage == "research_done":
        for w in data["workers"]:
            dur = w["finished_at"] - w["started_at"]
            print(f"           {STATUS_ICON[w['status']]} {w['specialist']:9s} {w['status']:8s} "
                  f"{len(w['evidence_ids'])} evidence  {dur:5.1f}s" + (f"  ({w['error']})" if w["error"] else ""))
    elif stage == "analysed":
        print(f"{t} 🤖 Evidence Analyst: {data['agreements']} agreements, {data['contradictions']} contradictions, "
              f"gaps: {data['gaps'] or 'none'}")
    elif stage == "analysis_failed":
        print(f"{t} ⚠️  Evidence analysis failed, continuing without it: {data['error']}")
    elif stage == "drafted":
        print(f"{t} 🤖 Writer: {data['findings']} findings, impact={data['impact']}")
    elif stage == "citations_repaired":
        print(f"{t} 🔁 Writer repair (code-triggered): invalid citations {data['before']} → {data['after']}")
    elif stage == "citations_repair_failed":
        print(f"{t} ⚠️  Citation repair failed, keeping first draft: {data['error']}")
    elif stage == "checked":
        print(f"{t} 🐍 Checks: confidence {data['confidence']['score']} | invalid citations: "
              f"{len(data['invalid_citations'])}")


def print_timeline(result: RunResult) -> None:
    tl = result.timeline
    if not tl["bars"]:
        return
    width = 50
    total = max(b["end"] for b in tl["bars"]) or 1
    print("\nParallel timeline")
    for b in tl["bars"]:
        start, end = int(b["start"] / total * width), max(int(b["end"] / total * width), int(b["start"] / total * width) + 1)
        print(f"  {b['specialist']:9s} |{' ' * start}{'█' * (end - start)}{' ' * (width - end)}| {b['end'] - b['start']:5.1f}s")
    print(f"  wall time {tl['wall_seconds']}s vs {tl['sequential_seconds']}s if run one after another "
          f"→ {tl['speedup']}× faster")


def print_model_usage() -> None:
    """Which models answered, how often we switched, and time lost to the free-tier rate limit."""
    model = get_model()
    used = ", ".join(f"{k} {v}" for k, v in model.calls_by_model.items()) or "none"
    waited = model._limiter.total_wait_seconds + sum(f.limiter.total_wait_seconds for f in model.fallbacks)
    print(f"LLM calls by model: {used} | {waited:.0f}s queued for the {settings_rpm}/min rate limit "
          "(summed over parallel calls)")
    if model.fallback_events:
        print(f"🔁 Fallbacks: {len(model.fallback_events)} (first: {model.fallback_events[0]['from']} → "
              f"{model.fallback_events[0]['to']}, {model.fallback_events[0]['error'][:70]})")


def save(result: RunResult) -> Path:
    out_dir = ROOT / "samples"
    out_dir.mkdir(exist_ok=True)
    md = out_dir / f"{result.run_id}.md"
    md.write_text(result.draft_markdown, encoding="utf-8")
    summary = {
        "run_id": result.run_id, "question": result.question, "route": dataclasses.asdict(result.route),
        "plan": result.plan.model_dump(), "decision": result.decision.model_dump(),
        "assignment_corrections": result.assignment.notes,
        "workers": [w.model_dump() for w in result.workers], "timeline": result.timeline,
        "analysis": result.analysis.model_dump(), "report": result.report.model_dump(),
        "invalid_citations": result.invalid_citations, "confidence": result.confidence,
        "tool_calls": [{"tool": r.tool, "ok": r.ok, "summary": r.summary} for r in result.ctx.tool_log],
    }
    (out_dir / f"{result.run_id}.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return md


async def main(question: str, live_search: bool) -> int:
    try:
        settings = load_settings()
        settings.require_llm_key()
    except ConfigError as exc:
        print(f"[config] {exc}")
        return 1
    if not live_search:
        settings = dataclasses.replace(settings, search_mode="offline")
    configure_tracing(settings)
    global settings_rpm
    settings_rpm = settings.llm_requests_per_minute
    print(f"Question: {question}")
    fallbacks = " → ".join(f"{p}/{m}" for p, m in settings.llm_fallbacks) or "none"
    print(f"Model: {settings.llm_provider}/{settings.model_name} (fallback: {fallbacks}) | "
          f"search: {settings.search_mode} | rate limit: {settings.llm_requests_per_minute}/min\n")
    try:
        result = await run_research(question, settings=settings, store=Store(settings.db_path), on_event=on_event)
    except PipelineError as exc:
        print(f"\n❌ Pipeline stopped at stage '{exc.stage}': {exc}")
        print_model_usage()
        return 1
    except RateLimitError as exc:
        print(f"\n[rate limit] {exc.message}")
        return 1
    except APITimeoutError:
        print("\n[timeout] The model did not answer in time.")
        return 1
    except APIStatusError as exc:
        print(f"\n[api] {exc.status_code}: {exc.message}")
        return 1
    except APIConnectionError:
        print("\n[network] Could not reach the LLM provider.")
        return 1

    print_timeline(result)
    if result.analysis.contradictions:
        print("\nContradictions found")
        for c in result.analysis.contradictions:
            print(f"  ⚡ {c.topic}: {c.evidence_ids_a} vs {c.evidence_ids_b} — {c.explanation[:110]}")
    if result.invalid_citations:
        print("\n⚠️  Invalid citations (the Phase 5 output guardrail will reject these):")
        for ic in result.invalid_citations:
            print(f"  {ic['evidence_id']}: {ic['reason']}")
    c = result.confidence
    print(f"\nConfidence {c['score']} = coverage {c['coverage']} × quality {c['source_quality']} × "
          f"(1 − {c['contradiction_penalty']}) × workers {c['worker_factor']}")
    print(f"  {c['explanation']}")
    if result.route.requires_review:
        print(f"\n⏸️  Review required: {result.route.reason} (enforced from Phase 5)")

    r = result.report
    print(f"\n{'=' * 70}\n{r.title}\n{'=' * 70}\n{r.executive_summary}\n")
    for i, f in enumerate(r.findings, 1):
        print(f"{i}. {f.statement} {f.evidence_ids} ({f.strength})")
    print(f"\nRecommendation: {r.recommendation}")
    tools = result.ctx.tool_log
    by_tool: dict[str, list[int]] = {}
    for t in tools:
        by_tool.setdefault(t.tool, [0, 0])[0 if t.ok else 1] += 1
    print("\nTool calls (ok / rejected): " + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in by_tool.items()))
    print_model_usage()
    print(f"Total time {time.time() - T0:.0f}s")
    print(f"Draft report saved: {save(result).relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default="primary",
                        help="primary | secondary | sensitive | risk | or your own question in quotes")
    parser.add_argument("--live-search", action="store_true")
    args = parser.parse_args()
    q = QUESTIONS.get(args.question, args.question)
    raise SystemExit(asyncio.run(main(q, args.live_search)))
