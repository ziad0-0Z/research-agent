"""Human review queue: decide, then resume a paused run (possibly in a new process).

    paused_for_review ──human──► approved           → release again (output guardrail + new token) → completed
                                 rejected           → rejected (nothing released)
                                 changes_requested  → Writer revises with the reviewer's note
                                                      → Writer ⇄ Critic loop → release → back to the queue

Everything resume needs is in the store (run state, sources, evidence, review), so a run paused
yesterday resumes today. Tokens are never stored: every resume re-runs the output guardrail and
gets a fresh token. A human-revised report always goes back to a human before release.
"""
from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from agents import Agent, custom_span, trace

from ..config import Settings, load_settings
from ..context import AppContext
from ..guardrails.input import InputCheck
from ..models import EvidenceAnalysis, HumanReview, ResearchPlan, ResearchReport, WorkerFindings, utcnow
from ..store import Store
from .confidence import compute_confidence
from .eval_loop import revision_prompt, writer_critic_loop
from .release import Release, release_report
from .session import open_session, record_turn

Decision = Literal["approved", "rejected", "changes_requested"]
RunAgent = Callable[[Agent, str, AppContext, int], Awaitable[Any]]
EventCallback = Callable[[str, dict], None]
MAX_NOTE_CHARS = 1000


class ReviewError(RuntimeError):
    pass


def list_pending(store: Store) -> list[dict]:
    out = []
    for review in store.pending_reviews():
        row = store.get_run(review.run_id)
        state = json.loads(row["state_json"]) if row else {}
        report = state.get("report") or {}
        out.append({"run_id": review.run_id, "question": row["question"] if row else "", "reasons": review.reasons,
                    "requested_at": review.created_at.isoformat(), "title": report.get("title", ""),
                    "impact": report.get("impact"), "confidence": (state.get("confidence") or {}).get("score"),
                    "rounds": len(state.get("review_history", [])) + 1})
    return out


def decide_review(store: Store, run_id: str, decision: Decision, note: str = "") -> HumanReview:
    """Record a human decision. Only a pending review can be decided, and only once."""
    if decision not in ("approved", "rejected", "changes_requested"):
        raise ReviewError(f"unknown decision {decision!r}")
    review = store.get_review(run_id)
    if review is None:
        raise ReviewError(f"run {run_id} has no review request")
    if review.status != "pending":
        raise ReviewError(f"review for {run_id} is already {review.status}")
    note = (note or "").strip()[:MAX_NOTE_CHARS]
    if decision == "changes_requested" and not note:
        raise ReviewError("say what should change (the note is sent to the Writer)")
    review.status, review.reviewer_note, review.decided_at = decision, note, utcnow()
    store.upsert_review(review)

    row = store.get_run(run_id)
    state = json.loads(row["state_json"]) if row and row["state_json"] else {}
    state.setdefault("review_history", []).append(
        {"decision": decision, "note": note, "reasons": review.reasons, "decided_at": review.decided_at.isoformat()})
    store.update_run(run_id, state=state)
    return review


@dataclass
class ResumeResult:
    run_id: str
    status: str                       # completed | paused_for_review | rejected | rejected_by_guardrail
    decision: Decision
    release: Release | None
    report: ResearchReport | None
    final_markdown: str | None
    confidence: dict | None
    ctx: AppContext


def _restore(settings: Settings, store: Store, run_id: str) -> tuple[AppContext, dict]:
    row = store.get_run(run_id)
    if row is None:
        raise ReviewError(f"unknown run {run_id}")
    if row["status"] != "paused_for_review":
        raise ReviewError(f"run {run_id} is {row['status']}, not paused_for_review")
    state = json.loads(row["state_json"] or "{}")
    if not state.get("report") or not state.get("plan"):
        raise ReviewError(f"run {run_id} has no saved report to resume")
    ctx = AppContext(settings=settings, store=store, run_id=run_id, question=row["question"])
    ctx.final_report = ResearchReport.model_validate(state["report"])
    ctx.analysis = EvidenceAnalysis.model_validate(state["analysis"])
    for reason in state.get("review_reasons", []):
        ctx.flag_review(reason)
    ic = state.get("input_check") or {}
    ctx.input_check = InputCheck(True, "valid_research", ic.get("reason", "checked when the run started"),
                                 ic.get("checked_by", "guard_agent"), ic.get("domain"))
    ctx.demo_fault = state.get("demo_fault")
    ctx.memory_reuse = dict(state.get("memory_reuse") or {})
    ctx._counters["ev"] = len(store.list_evidence(run_id))
    ctx._counters["src"] = len(store.list_sources(run_id))
    return ctx, state


async def _record_in_conversation(settings: Settings, state: dict, run_id: str, question: str, plan: ResearchPlan,
                                  status: str, report: ResearchReport | None) -> None:
    """Tell the run's conversation (session memory) how the paused run ended."""
    if state.get("session_id"):
        await record_turn(open_session(settings, state["session_id"]), run_id=run_id, question=question,
                          main_question=plan.main_question, status=status, report=report)


async def resume_run(run_id: str, *, settings: Settings | None = None, store: Store | None = None,
                     run_agent: RunAgent | None = None, on_event: EventCallback | None = None) -> ResumeResult:
    from .pipeline import _critic, _emit, _writer, _writer_prompt, default_run_agent, save_run_state

    settings = settings or load_settings()
    store = store or Store(settings.db_path)
    run_agent = run_agent or default_run_agent
    review = store.get_review(run_id)
    if review is None:
        raise ReviewError(f"run {run_id} has no review request")
    if review.status == "pending":
        raise ReviewError(f"run {run_id} is still waiting for a human decision")
    ctx, state = _restore(settings, store, run_id)
    plan = ResearchPlan.model_validate(state["plan"])
    workers = [WorkerFindings.model_validate(w) for w in state.get("workers", [])]
    decision: Decision = review.status  # type: ignore[assignment]

    def emit(stage: str, data: dict) -> None:
        _emit(ctx, on_event, stage, **data)

    with trace("Resume run", group_id=run_id, metadata={"run_id": run_id, "decision": decision}):
        emit("resumed", {"decision": decision, "note": review.reviewer_note})
        if decision == "rejected":
            store.update_run(run_id, status="rejected")
            emit("rejected_by_human", {"note": review.reviewer_note})
            await _record_in_conversation(settings, state, run_id, ctx.question, plan, "rejected", None)
            return ResumeResult(run_id, "rejected", decision, None, ctx.final_report, None, state.get("confidence"), ctx)

        if decision == "changes_requested":
            writer_prompt = _writer_prompt(ctx, plan, ctx.analysis, workers)
            instructions = [f"Human reviewer: {review.reviewer_note}"]
            with custom_span("Writer revision (human feedback)"):
                revised = await run_agent(_writer(), revision_prompt(writer_prompt, ctx.final_report, instructions,
                                                                     source="a human reviewer"), ctx, 2)
            emit("revised", {"round": "human", "findings": len(revised.findings)})
            with custom_span("Writer ⇄ Critic loop", data={"max_rounds": settings.max_critic_rounds}):
                loop = await writer_critic_loop(
                    ctx, plan, ctx.analysis, workers, writer_prompt, revised,
                    run_writer=lambda agent, prompt: run_agent(agent, prompt, ctx, 2),
                    run_critic=lambda agent, prompt: run_agent(agent, prompt, ctx, 2),
                    writer_agent=_writer, critic_agent=_critic, max_rounds=settings.max_critic_rounds,
                    on_event=emit)
            ctx.final_report = loop.report
            if loop.verdict == "FAIL":
                ctx.flag_review(f"Critic FAIL after {len(loop.rounds)} round(s) ({loop.stop_reason})")
            ctx.flag_review("revised after a reviewer requested changes")  # a human must see the new version
            emit("critic_done", {"verdict": loop.verdict, "rounds": len(loop.rounds),
                                 "stop_reason": loop.stop_reason, "best_round": loop.best_round})

        with custom_span("Confidence (code)"):
            confidence = compute_confidence(ctx, plan, ctx.final_report, workers)
        release = release_report(ctx, workers, confidence, on_event=emit, plan=plan)
        save_run_state(ctx, plan=plan, analysis=ctx.analysis, workers=workers, confidence=confidence,
                       release=release, extra={"resumed_at": time.time(), "memory_commit": release.memory_commit})
        await _record_in_conversation(settings, state, run_id, ctx.question, plan, release.status, ctx.final_report)
    return ResumeResult(run_id, release.status, decision, release, ctx.final_report, release.final_markdown,
                        confidence, ctx)
