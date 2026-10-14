"""The research pipeline (Phases 2-5).

    question
      → [CODE+LLM] input guardrail      regex pre-filter → Guard Agent → verdict in code   (blocks attacks)
      → [CODE]  route_question          sensitive domain? → review required
      → [CODE]  memory: related reports similar past questions (shown only, never sent to an agent)
      → [LLM]   Planner                 ResearchPlan; sees earlier turns when this is a follow-up (session)
      → [LLM]   Research Orchestrator   who researches what, how deep   (orchestrator-workers)
      → [CODE]  validate_assignments    every subquestion assigned once
      → [CODE]  memory reuse            remembered evidence → save_evidence (re-verified); memory-covered
                                        subquestions are researched at quick depth
      → [LLM×N] specialists in parallel tools: search → scrape → extract → save   (parallelization)
      → [LLM]   Evidence Analyst        agreements / contradictions     (chain step 2)
      → [LLM]   Writer                  ResearchReport                  (chain step 3)
      → [LOOP]  Writer ⇄ Critic         code checks + RAG + LLM review, verdict in code (evaluator-optimizer)
      → [CODE]  confidence
      → [CODE]  release: output guardrail → approval token → impact gate
                → paused_for_review (resume later) | generate_report → completed | rejected_by_guardrail
      → [CODE]  completed only: commit verified evidence + report to long-term memory (memory/commit.py)
Every stage runs inside one trace with named spans, and is logged to ctx.stage_log.
"""
from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from agents import Agent, Runner, custom_span, guardrail_span, trace

from ..config import Settings, load_settings
from ..context import AppContext
from ..guardrails.input import InputCheck, check_input
from ..guardrails.output import OutputCheck
from ..models import (
    EvidenceAnalysis, OrchestrationDecision, ResearcherOutput, ResearchPlan, ResearchReport, WorkerFindings,
)
from ..store import Store
from ..tools import analysis as analysis_tools
from ..tools.evidence import check_citation
from ..tools.review import render_markdown
from ..tools.web import ToolError
from .confidence import compute_confidence
from ..memory import MemoryStore, lower_depth, related_reports, reuse_for_plan
from ..memory.lookup import MemoryUse
from .eval_loop import LoopResult, writer_critic_loop
from .parallel import WorkerTask, run_parallel, timeline
from .release import Release, release_report
from .reliability import build_reliability, snapshot
from .router import AssignmentPlan, RouteDecision, route_question, validate_assignments
from .session import follow_up_prompt, open_session, previous_turns, record_turn

RunAgent = Callable[[Agent, str, AppContext, int], Awaitable[Any]]
EventCallback = Callable[[str, dict], None]


async def default_run_agent(agent: Agent, prompt: str, ctx: AppContext, max_turns: int) -> Any:
    # Session memory: only the Planner sees the conversation, and only for a follow-up (a first question
    # keeps exactly the request it had before sessions existed, so recorded replays still match).
    session = ctx.session if agent.name == "Planner" and ctx.follow_up else None
    result = await Runner.run(agent, prompt, context=ctx, max_turns=max_turns, session=session)
    return result.final_output


class PipelineError(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(f"{stage}: {message}")
        self.stage = stage


class InputBlockedError(PipelineError):
    """The input guardrail blocked the request. Nothing else ran."""

    def __init__(self, check: InputCheck, run_id: str):
        super().__init__("input_guardrail", f"blocked ({check.category}): {check.reason}")
        self.check = check
        self.run_id = run_id


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
    critic: LoopResult | None = None
    stage_log: list[dict] = field(default_factory=list)
    ctx: AppContext | None = None
    status: str = "draft"                      # completed | paused_for_review | rejected_by_guardrail
    input_check: InputCheck | None = None
    output_check: OutputCheck | None = None
    gate_reasons: list[str] = field(default_factory=list)
    final_markdown: str | None = None          # only set when status == "completed"
    memory: dict | None = None                 # what long-term memory contributed (lookup + commit)
    session_id: str | None = None


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


# Demo faults (test mode):
#   "unsupported_claim"  before the Critic: a real citation for a claim it does not support (scenario 6)
#   "fake_citation"      after the Critic: a finding citing an evidence id that does not exist (scenario 9)
#   "tampered_quote"     after the Critic: a cited quote altered in the store after verification (scenario 9)
DemoFault = str
#   "search_outage"      Tavily + DuckDuckGo simulated down (offline corpus answers); the risk specialist
#                        loses every provider → partial report, labelled in its text (scenario 10, Phase 7)
DEMO_FAULTS = ("unsupported_claim", "fake_citation", "tampered_quote", "search_outage")


async def run_research(question: str, *, settings: Settings | None = None, store: Store | None = None,
                       run_agent: RunAgent | None = None, on_event: EventCallback | None = None,
                       demo_fault: DemoFault | None = None, session_id: str | None = None,
                       clock: Callable[[], Any] | None = None, run_id: str | None = None) -> RunResult:
    question = (question or "").strip()
    if demo_fault and demo_fault not in DEMO_FAULTS:
        raise PipelineError("input", f"unknown demo fault {demo_fault!r}")
    settings = settings or load_settings()
    store = store or Store(settings.db_path)
    run_agent = run_agent or default_run_agent
    ctx = AppContext(settings=settings, store=store, question=question, demo_fault=demo_fault,
                     **({"run_id": run_id} if run_id else {}))  # the UI picks the id so it can follow the run
    if clock is not None:
        ctx.clock = clock
    turns: list[dict] = []
    if session_id:
        ctx.session, ctx.session_id = open_session(settings, session_id), session_id
        turns = await previous_turns(ctx.session)
        ctx.follow_up = bool(turns)
    previous_question = (turns[-1].get("main_question") or turns[-1].get("question")) if turns else None
    mstore = MemoryStore(store)

    try:
        rel_start = snapshot()
        with trace("Research run", group_id=ctx.run_id,
                   metadata={"run_id": ctx.run_id, "question": question[:200]}) as run_trace:
            ctx.trace_ids.append(run_trace.trace_id)  # links the run to data/traces/<trace_id>.jsonl (Trace page)
            # 0. Input guardrail: regex pre-filter (code) → Guard Agent (LLM) → decision (code)
            with guardrail_span("Input guardrail") as span:
                check = await check_input(question, lambda prompt: run_agent(_guard(), prompt, ctx, 1),
                                          previous_question=previous_question)
                span.span_data.triggered = not check.allowed
            ctx.input_check = check
            _emit(ctx, on_event, "input_checked", **check.summary())
            if not check.allowed:
                store.update_run(ctx.run_id, status="blocked",
                                 state={"input_check": check.summary(), "trace_ids": ctx.trace_ids})
                raise InputBlockedError(check, ctx.run_id)
            if check.requires_review:
                ctx.flag_review(check.reason)

            # 1. CODE-ORCHESTRATED routing (keywords + the Guard's domain)
            with custom_span("Router (code)"):
                route = route_question(question, guard_domain=check.domain,
                                       sensitive_domains=settings.sensitive_domains)
                if route.requires_review:
                    ctx.flag_review(route.reason)
            _emit(ctx, on_event, "routed", path=route.path, domains=route.domains, matched=route.matched_terms)

            # 1b. Long-term memory: similar past reports (display only — a past conclusion is not evidence)
            related = related_reports(ctx, mstore, question) if settings.memory_mode != "off" else []
            if related:
                _emit(ctx, on_event, "memory_related", reports=related)

            # 2. Planner (chain step 1); for a follow-up the SDK session supplies the earlier turns
            try:
                plan: ResearchPlan = await run_agent(_planner(), follow_up_prompt(question, turns), ctx, 2)
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

            # 3b. Long-term memory reuse (code): remembered evidence enters through save_evidence (re-verified);
            #     a worker whose subquestions are all memory-covered still researches, at quick depth.
            with custom_span("Memory reuse (code)"):
                # search_outage demo: no memory reuse, so the outage's effect is not hidden by remembered evidence
                memory_use = (MemoryUse(enabled=False) if demo_fault == "search_outage"
                              else await reuse_for_plan(ctx, mstore, plan))
            memory_use.related = related
            covered = memory_use.covered()
            depths = {kind: (lower_depth(decision.depth, "quick") if sqs and all(sq.id in covered for sq in sqs)
                             else decision.depth) for kind, sqs in assignment.by_specialist.items()}
            if memory_use.enabled and memory_use.candidates:
                _emit(ctx, on_event, "memory_reused", **{k: v for k, v in memory_use.summary().items()
                                                         if k != "related_reports"}, depths=depths)

            # 4. Specialists in parallel
            tasks = []
            for kind, sqs in assignment.by_specialist.items():
                prompt = _researcher_prompt(plan.main_question, plan.decision_context, sqs, depths[kind])
                agent = _researcher(kind, depths[kind])
                max_turns = 10 + 2 * len(sqs)  # batched process needs ~4 turns; the rest is slack

                def make_run(agent=agent, prompt=prompt, max_turns=max_turns) -> Awaitable[ResearcherOutput]:
                    return run_agent(agent, prompt, ctx, max_turns)

                tasks.append(WorkerTask(kind, [sq.id for sq in sqs], make_run))
            _emit(ctx, on_event, "research_started", workers=[t.specialist for t in tasks])
            with custom_span("Parallel research", data={"workers": [t.specialist for t in tasks]}):
                workers = await run_parallel(ctx, tasks, timeout=settings.worker_timeout_seconds)
            tl = timeline(workers)
            ctx.reliability = build_reliability(ctx, workers, rel_start)
            _emit(ctx, on_event, "research_done", workers=[w.model_dump() for w in workers], timeline=tl,
                  reliability=ctx.reliability)

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

            if demo_fault == "unsupported_claim":
                report, planted = plant_unsupported_claim(ctx, report)
                _emit(ctx, on_event, "fault_planted", kind=demo_fault, statement=planted)

            # 7. Evaluator-optimizer: Writer ⇄ Critic (bounded; verdict decided in code).
            with custom_span("Writer ⇄ Critic loop", data={"max_rounds": settings.max_critic_rounds}):
                loop = await writer_critic_loop(
                    ctx, plan, analysis, workers, writer_prompt, report,
                    run_writer=lambda agent, prompt: run_agent(agent, prompt, ctx, 2),
                    run_critic=lambda agent, prompt: run_agent(agent, prompt, ctx, 2),
                    writer_agent=_writer, critic_agent=_critic, max_rounds=settings.max_critic_rounds,
                    on_event=lambda stage, data: _emit(ctx, on_event, stage, **data))
            report = loop.report
            if loop.verdict == "FAIL":
                ctx.flag_review(f"Critic FAIL after {len(loop.rounds)} round(s) ({loop.stop_reason})")
            _emit(ctx, on_event, "critic_done", verdict=loop.verdict, rounds=len(loop.rounds),
                  stop_reason=loop.stop_reason, best_round=loop.best_round)
            if demo_fault in ("fake_citation", "tampered_quote"):  # after the Critic: only the output guardrail is left
                report, planted = (plant_fake_citation(report) if demo_fault == "fake_citation"
                                   else tamper_cited_quote(ctx, report))
                _emit(ctx, on_event, "fault_planted", kind=demo_fault, statement=planted)
            invalid = _invalid_citations(ctx, report)
            ctx.final_report = report

            with custom_span("Confidence (code)"):
                confidence = compute_confidence(ctx, plan, report, workers)
            _emit(ctx, on_event, "checked", invalid_citations=invalid, confidence=confidence)

            draft = (f"> **DRAFT** — Critic verdict **{loop.verdict}** after {len(loop.rounds)} round(s) "
                     f"({loop.stop_reason}).\n\n" + render_markdown(ctx))

            # 8. Release: output guardrail → token → impact gate → review queue or final report (all code)
            save_run_state(ctx, plan=plan, analysis=analysis, workers=workers, confidence=confidence,
                           extra={"critic_verdict": loop.verdict, "critic_rounds": len(loop.rounds),
                                  "critic_stop_reason": loop.stop_reason, "route": route.path,
                                  "memory_lookup": memory_use.summary(), "worker_depths": depths})
            release = release_report(ctx, workers, confidence, plan=plan,
                                     on_event=lambda stage, data: _emit(ctx, on_event, stage, **data))
            ctx.reliability = build_reliability(ctx, workers, rel_start)  # + LLM/embedding events up to release
            save_run_state(ctx, plan=plan, analysis=analysis, workers=workers, confidence=confidence,
                           release=release, extra={"memory_commit": release.memory_commit, "timeline": tl})
            await record_turn(ctx.session, run_id=ctx.run_id, question=question, main_question=plan.main_question,
                              status=release.status, report=ctx.final_report)
    except InputBlockedError:
        raise  # already recorded as "blocked"
    except Exception as exc:
        _mark_failed(store, ctx.run_id, exc, ctx.trace_ids)
        raise

    return RunResult(
        run_id=ctx.run_id, question=question, route=route, plan=plan, decision=decision, assignment=assignment,
        workers=workers, timeline=tl, analysis=analysis, report=report, invalid_citations=invalid,
        confidence=confidence, draft_markdown=draft, critic=loop, stage_log=ctx.stage_log, ctx=ctx,
        status=release.status, input_check=check, output_check=release.output_check,
        gate_reasons=release.gate_reasons, final_markdown=release.final_markdown,
        memory={"lookup": memory_use.summary(), "commit": release.memory_commit}, session_id=ctx.session_id,
    )


def _mark_failed(store: Store, run_id: str, exc: Exception, trace_ids: list[str] | None = None) -> None:
    """A run that crashed must not stay "running" forever: record where and why it stopped."""
    row = store.get_run(run_id)
    state = json.loads(row["state_json"]) if row and row["state_json"] else {}
    state["error"] = {"stage": getattr(exc, "stage", "unknown"), "message": f"{type(exc).__name__}: {exc}"[:500]}
    if trace_ids:
        state["trace_ids"] = list(dict.fromkeys(state.get("trace_ids", []) + trace_ids))
    store.update_run(run_id, status="failed", state=state)


def save_run_state(ctx: AppContext, *, plan: ResearchPlan, analysis: EvidenceAnalysis,
                   workers: list[WorkerFindings], confidence: dict, release: Release | None = None,
                   extra: dict | None = None) -> None:
    """Everything resume needs, persisted with the run. Evidence and sources are already in the store.

    The approval token is never persisted: a resumed run must pass the output guardrail again.
    """
    row = ctx.store.get_run(ctx.run_id)
    state = json.loads(row["state_json"]) if row and row["state_json"] else {}
    state.update(extra or {})
    state.update({
        "plan": plan.model_dump(), "analysis": analysis.model_dump(),
        "workers": [w.model_dump() for w in workers], "report": ctx.final_report.model_dump() if ctx.final_report else None,
        "confidence": confidence, "review_reasons": ctx.review_reasons, "updated_at": time.time(),
        "input_check": ctx.input_check.summary() if ctx.input_check else None,
        "demo_fault": ctx.demo_fault, "memory_reuse": ctx.memory_reuse, "session_id": ctx.session_id,
        "trace_ids": list(dict.fromkeys(state.get("trace_ids", []) + ctx.trace_ids)),
        "reliability": ctx.reliability, "stage_log": _stage_log_for_state(ctx.stage_log),
    })
    if release is not None:
        state.update({"release_status": release.status, "output_check": release.output_check.summary(),
                      "gate_reasons": release.gate_reasons, "final_markdown": release.final_markdown})
    ctx.store.update_run(ctx.run_id, state=state)


def _stage_log_for_state(log: list[dict]) -> list[dict]:
    """Compact stage timeline for the UI (stage name + time only; full payloads stay in the trace)."""
    return [{"stage": e["stage"], "at": e["at"]} for e in log][-200:]


PLANTED_STATEMENT = ("Self-hosting an open-source model is always cheaper than cloud APIs, "
                     "for any team size and any usage volume.")


def plant_unsupported_claim(ctx: AppContext, report: ResearchReport) -> tuple[ResearchReport, str]:
    """Demo scenario 6: add a finding the cited evidence does not support, so the Critic must catch it.

    It cites a real, verified evidence id (so the citation check passes): only the support check can
    catch it. Prefers evidence that says the opposite (hosted APIs cheaper for small teams).
    """
    from ..models import Finding
    items = ctx.store.list_evidence(ctx.run_id)
    opposite = [e for e in items if any(w in e.quote.lower() for w in ("lower tco", "cheapest", "hosted apis had"))]
    target = (opposite or items)[0]
    planted = Finding(statement=PLANTED_STATEMENT, evidence_ids=[target.id], strength="strong")
    return report.model_copy(update={"findings": list(report.findings) + [planted]}), PLANTED_STATEMENT


FAKE_CITATION_STATEMENT = "An independent 2026 audit found hosted APIs cut total costs by 70% for every team size."


def plant_fake_citation(report: ResearchReport) -> tuple[ResearchReport, str]:
    """Demo scenario 9: after the Critic, add a finding citing an evidence id that does not exist."""
    from ..models import Finding
    fake = Finding(statement=FAKE_CITATION_STATEMENT, evidence_ids=["ev999"], strength="strong")
    return report.model_copy(update={"findings": list(report.findings) + [fake]}), FAKE_CITATION_STATEMENT


def tamper_cited_quote(ctx: AppContext, report: ResearchReport) -> tuple[ResearchReport, str]:
    """Demo scenario 9b: after the Critic, alter a cited quote in the store (as a bug or tampering would).
    The id still exists and is marked verified; only re-checking the quote against the page catches it."""
    eid = report.findings[0].evidence_ids[0]
    ctx.store._exec("UPDATE evidence SET quote = quote || ' Hosted APIs are 90% cheaper in all cases.' "
                    "WHERE run_id=? AND id=?", (ctx.run_id, eid))
    return report, f"quote of {eid} altered after verification"


def _invalid_citations(ctx: AppContext, report: ResearchReport) -> list[dict]:
    out = []
    for eid in sorted(report.cited_evidence_ids()):
        det = check_citation(ctx, eid)
        if not det["valid"]:
            out.append({"evidence_id": eid, "reason": det["reason"]})
    return out


# Indirection so tests can run the pipeline without building real agents.
def _guard() -> Agent:
    from ..agents.guard import guard
    return guard()


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


def _critic() -> Agent:
    from ..agents.critic import critic
    return critic()
