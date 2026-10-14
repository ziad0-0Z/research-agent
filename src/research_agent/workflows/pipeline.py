"""The research pipeline (Phases 2-3).

    question
      → [CODE]  route_question          sensitive domain? → review required
      → [LLM]   Planner                 ResearchPlan                    (chain step 1)
      → [LLM]   Research Orchestrator   who researches what, how deep   (orchestrator-workers)
      → [CODE]  validate_assignments    every subquestion assigned once
      → [LLM×N] specialists in parallel tools: search → scrape → extract → save   (parallelization)
      → [LLM]   Evidence Analyst        agreements / contradictions     (chain step 2)
      → [LLM]   Writer                  ResearchReport                  (chain step 3)
      → [CODE]  citation check + confidence

Phase 4 adds the Writer ⇄ Critic loop, Phase 5 the guardrails and human-review gate.
Every stage runs inside one trace with named spans, and is logged to ctx.stage_log.
"""
from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from agents import Agent, Runner, custom_span, trace

from ..config import Settings, load_settings
from ..context import AppContext
from ..models import (
    EvidenceAnalysis, OrchestrationDecision, ResearcherOutput, ResearchPlan, ResearchReport, WorkerFindings,
)
from ..store import Store
from ..tools import analysis as analysis_tools
from ..tools.evidence import check_citation
from ..tools.review import render_markdown
from ..tools.web import ToolError
from .confidence import compute_confidence
from .parallel import WorkerTask, run_parallel, timeline
from .router import AssignmentPlan, RouteDecision, route_question, validate_assignments

RunAgent = Callable[[Agent, str, AppContext, int], Awaitable[Any]]
EventCallback = Callable[[str, dict], None]


async def default_run_agent(agent: Agent, prompt: str, ctx: AppContext, max_turns: int) -> Any:
    result = await Runner.run(agent, prompt, context=ctx, max_turns=max_turns)
    return result.final_output


class PipelineError(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(f"{stage}: {message}")
        self.stage = stage


@dataclass
class RunResult:
    run_id: str
    question: str
    route: RouteDecision
    plan: ResearchPlan
    decision: OrchestrationDecision
    assignment: AssignmentPlan
    workers: list[WorkerFindings]
    timeline: dict
    analysis: EvidenceAnalysis
    report: ResearchReport
    invalid_citations: list[dict]
    confidence: dict
    draft_markdown: str
    stage_log: list[dict] = field(default_factory=list)
    ctx: AppContext | None = None


def _emit(ctx: AppContext, on_event: EventCallback | None, stage: str, **data: Any) -> None:
    ctx.log_stage(stage, **data)
    if on_event:
        on_event(stage, data)


def _researcher_prompt(question: str, context: str, subquestions: list, depth: str) -> str:
    items = [{"id": sq.id, "question": sq.question, "success_criteria": sq.success_criteria,
              "search_queries": sq.search_queries} for sq in subquestions]
    return (f"Decision question: {question}\nContext: {context}\nDepth: {depth}\n\n"
            f"Your subquestions:\n{json.dumps(items, indent=1, ensure_ascii=False)}")


def _writer_prompt(ctx: AppContext, plan: ResearchPlan, analysis: EvidenceAnalysis,
                   workers: list[WorkerFindings]) -> str:
    evidence = []
    for ev in ctx.store.list_evidence(ctx.run_id):
        src = ctx.store.get_source(ctx.run_id, ev.source_id)
        evidence.append({
            "id": ev.id, "subquestion_id": ev.subquestion_id, "claim": ev.claim, "quote": ev.quote[:350],
            "source_title": src.title if src else "", "source_type": src.domain_type if src else "unknown",
            "source_quality": src.quality_score if src else 0.0,
            "published": src.published.isoformat() if src and src.published else None,
            "low_trust": bool(src and src.injection_detected),
        })
    worker_status = [{"specialist": w.specialist, "status": w.status, "subquestions": w.subquestion_ids,
                      "error": w.error} for w in workers]
    valid_ids = ", ".join(e["id"] for e in evidence)
    return (
        f"Decision question: {plan.main_question}\nContext: {plan.decision_context}\n\n"
        f"Plan:\n{plan.model_dump_json(indent=1)}\n\n"
        f"Valid evidence ids: {valid_ids}\n\n"
        f"Saved evidence ({len(evidence)} items):\n{json.dumps(evidence, indent=1, ensure_ascii=False)}\n\n"
        f"Evidence analysis:\n{analysis.model_dump_json(indent=1)}\n\n"
        f"Research worker status:\n{json.dumps(worker_status, indent=1)}"
    )


async def run_research(question: str, *, settings: Settings | None = None, store: Store | None = None,
                       run_agent: RunAgent | None = None, on_event: EventCallback | None = None) -> RunResult:
    question = (question or "").strip()
    if len(question) < 10:
        raise PipelineError("input", "question is too short")
    settings = settings or load_settings()
    store = store or Store(settings.db_path)
    run_agent = run_agent or default_run_agent
    ctx = AppContext(settings=settings, store=store, question=question)

    with trace("Research run", group_id=ctx.run_id,
               metadata={"run_id": ctx.run_id, "question": question[:200]}):
        # 1. CODE-ORCHESTRATED routing
        with custom_span("Router (code)"):
            route = route_question(question, sensitive_domains=settings.sensitive_domains)
            if route.requires_review:
                ctx.flag_review(route.reason)
        _emit(ctx, on_event, "routed", path=route.path, domains=route.domains, matched=route.matched_terms)

        # 2. Planner (chain step 1)
        try:
            plan: ResearchPlan = await run_agent(_planner(), question, ctx, 2)
        except Exception as exc:
            raise PipelineError("planner", f"{type(exc).__name__}: {exc}") from exc
        _emit(ctx, on_event, "planned", subquestions=[sq.model_dump() for sq in plan.subquestions])

        # 3. LLM-ORCHESTRATED delegation, validated by code
        try:
            decision: OrchestrationDecision = await run_agent(
                _orchestrator(), f"Research plan:\n{plan.model_dump_json(indent=1)}", ctx, 2)
        except Exception as exc:
            raise PipelineError("orchestrator", f"{type(exc).__name__}: {exc}") from exc
        with custom_span("Validate assignments (code)"):
            assignment = validate_assignments(plan, decision)
        _emit(ctx, on_event, "delegated", depth=decision.depth, rationale=decision.rationale,
              assignments={k: [sq.id for sq in v] for k, v in assignment.by_specialist.items()},
              corrections=assignment.notes)

        # 4. Specialists in parallel
        tasks = []
        for kind, sqs in assignment.by_specialist.items():
            prompt = _researcher_prompt(plan.main_question, plan.decision_context, sqs, decision.depth)
            agent = _researcher(kind, decision.depth)
            max_turns = 10 + 2 * len(sqs)  # batched process needs ~4 turns; the rest is slack

            def make_run(agent=agent, prompt=prompt, max_turns=max_turns) -> Awaitable[ResearcherOutput]:
                return run_agent(agent, prompt, ctx, max_turns)

            tasks.append(WorkerTask(kind, [sq.id for sq in sqs], make_run))
        _emit(ctx, on_event, "research_started", workers=[t.specialist for t in tasks])
        with custom_span("Parallel research", data={"workers": [t.specialist for t in tasks]}):
            workers = await run_parallel(ctx, tasks, timeout=settings.worker_timeout_seconds)
        tl = timeline(workers)
        _emit(ctx, on_event, "research_done", workers=[w.model_dump() for w in workers], timeline=tl)

        # 5. Evidence Analyst (chain step 2)
        n_evidence = len(ctx.store.list_evidence(ctx.run_id))
        if n_evidence == 0:
            raise PipelineError("research", "no evidence was saved by any worker")
        if n_evidence >= 2:
            try:
                await analysis_tools.compare_sources(
                    ctx, run_agent=lambda agent, prompt: run_agent(agent, prompt, ctx, 2))
            except (ToolError, Exception) as exc:  # analysis failure must not lose the research
                _emit(ctx, on_event, "analysis_failed", error=f"{type(exc).__name__}: {exc}")
        analysis = ctx.analysis or EvidenceAnalysis(agreements=[], contradictions=[], gaps=[])
        _emit(ctx, on_event, "analysed", agreements=len(analysis.agreements),
              contradictions=len(analysis.contradictions), gaps=analysis.gaps)

        # 6. Writer (chain step 3)
        writer_prompt = _writer_prompt(ctx, plan, analysis, workers)
        try:
            report: ResearchReport = await run_agent(_writer(), writer_prompt, ctx, 2)
        except Exception as exc:
            raise PipelineError("writer", f"{type(exc).__name__}: {exc}") from exc
        _emit(ctx, on_event, "drafted", findings=len(report.findings), impact=report.impact)

        # 7. CODE checks: every cited id must exist. One bounded repair round if not
        #    (Phase 4's Critic loop then checks support, not just existence).
        with custom_span("Citation check (code)"):
            invalid = _invalid_citations(ctx, report)
        for attempt in range(1, MAX_CITATION_REPAIRS + 1):
            if not invalid:
                break
            bad = ", ".join(i["evidence_id"] for i in invalid)
            repair_prompt = (writer_prompt + "\n\nRevision instructions:\n"
                             f"- These cited ids do not exist: {bad}. Replace each with the correct id from "
                             "\"Valid evidence ids\", or remove the citation (and the finding, if nothing supports it).")
            try:
                repaired: ResearchReport = await run_agent(_writer(), repair_prompt, ctx, 2)
            except Exception as exc:  # keep the first draft; its bad ids stay flagged
                _emit(ctx, on_event, "citations_repair_failed", error=f"{type(exc).__name__}: {exc}")
                break
            new_invalid = _invalid_citations(ctx, repaired)
            _emit(ctx, on_event, "citations_repaired", attempt=attempt, before=len(invalid), after=len(new_invalid))
            if len(new_invalid) < len(invalid):
                report, invalid = repaired, new_invalid
        ctx.final_report = report

        with custom_span("Confidence (code)"):
            confidence = compute_confidence(ctx, plan, report, workers)
        _emit(ctx, on_event, "checked", invalid_citations=invalid, confidence=confidence)

        draft = "> **DRAFT** — not yet reviewed by the Critic or the output guardrail.\n\n" + render_markdown(ctx)
        store.update_run(ctx.run_id, status="drafted", state={"finished_at": time.time()})

    return RunResult(
        run_id=ctx.run_id, question=question, route=route, plan=plan, decision=decision, assignment=assignment,
        workers=workers, timeline=tl, analysis=analysis, report=report, invalid_citations=invalid,
        confidence=confidence, draft_markdown=draft, stage_log=ctx.stage_log, ctx=ctx,
    )


MAX_CITATION_REPAIRS = 1


def _invalid_citations(ctx: AppContext, report: ResearchReport) -> list[dict]:
    out = []
    for eid in sorted(report.cited_evidence_ids()):
        det = check_citation(ctx, eid)
        if not det["valid"]:
            out.append({"evidence_id": eid, "reason": det["reason"]})
    return out


# Indirection so tests can run the pipeline without building real agents.
def _planner() -> Agent:
    from ..agents.planner import planner
    return planner()


def _orchestrator() -> Agent:
    from ..agents.orchestrator import orchestrator
    return orchestrator()


def _researcher(kind: str, depth: str) -> Agent:
    from ..agents.researchers import researcher
    return researcher(kind, depth)


def _writer() -> Agent:
    from ..agents.writer import writer
    return writer()
