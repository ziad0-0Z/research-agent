"""Advanced parallel execution of research workers.

- Workers run concurrently with asyncio.gather.
- Each worker has its own timeout; a slow worker never blocks the others.
- A worker that crashes or times out is isolated: the run continues.
- Status per worker comes from the evidence actually saved (not from what the
  LLM says it did): success (every subquestion has evidence) / partial / failed / timeout.
  If a worker stops early (turn limit, timeout) after covering everything, it still
  succeeded; the reason is kept in `error` as a note.
- Start and finish times are recorded for the parallel timeline.
- LLM concurrency is capped by the shared rate limiter in llm.py.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ..context import CURRENT_SPECIALIST, AppContext
from ..models import ResearcherOutput, WorkerFindings


@dataclass
class WorkerTask:
    specialist: str
    subquestion_ids: list[str]
    run: Callable[[], Awaitable[ResearcherOutput]]


def _evidence_for(ctx: AppContext, subquestion_ids: list[str]) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {sq: [] for sq in subquestion_ids}
    for ev in ctx.store.list_evidence(ctx.run_id):
        if ev.subquestion_id in found and ev.status == "active":
            found[ev.subquestion_id].append(ev.id)
    return found


def _last_failed_tool(ctx: AppContext, specialist: str, since: float) -> str | None:
    for r in reversed(ctx.tool_log):
        if r.specialist == specialist and r.at >= since and not r.ok:
            return f"last failed tool: {r.tool} ({r.summary[:120]})"
    return None


async def _attempt(ctx: AppContext, task: WorkerTask, timeout: float
                   ) -> tuple[ResearcherOutput | None, str | None, bool]:
    started = time.time()
    try:
        return await asyncio.wait_for(task.run(), timeout=timeout), None, False
    except asyncio.TimeoutError:
        return None, f"timed out after {timeout:.0f}s", True
    except Exception as exc:  # noqa: BLE001 - isolate any worker failure
        error = f"{type(exc).__name__}: {str(exc)[:200]}"
        hint = _last_failed_tool(ctx, task.specialist, started)
        return None, f"{error}; {hint}" if hint else error, False


async def _run_one(ctx: AppContext, task: WorkerTask, timeout: float, retries: int = 0) -> WorkerFindings:
    CURRENT_SPECIALIST.set(task.specialist)  # this coroutine runs in its own task: the value stays local
    started = time.time()
    output, error, timed_out = await _attempt(ctx, task, timeout)
    attempts = 1
    # A worker that crashed fast without saving anything gets one more try (a transient schema or provider
    # error is common on free tiers). A timeout is not retried: it would only double the wait.
    while (error and not timed_out and attempts <= retries
           and not any(_evidence_for(ctx, task.subquestion_ids).values())):
        ctx.log_stage("worker_retry", specialist=task.specialist, error=error[:200], attempt=attempts + 1)
        remaining = max(30.0, timeout - (time.time() - started))
        output, error, timed_out = await _attempt(ctx, task, remaining)
        attempts += 1
    finished = time.time()

    per_sq = _evidence_for(ctx, task.subquestion_ids)
    evidence_ids = [e for ids in per_sq.values() for e in ids]
    covered = sum(1 for ids in per_sq.values() if ids)

    all_covered = covered == len(task.subquestion_ids)
    if all_covered:
        status = "success"  # judged by saved evidence; a turn limit or timeout after that is only a note
    elif timed_out:
        status = "timeout"
    elif error and not evidence_ids:
        status = "failed"
    elif evidence_ids:
        status = "partial"
    else:
        status = "failed"
        error = error or "no evidence saved"

    summaries: dict[str, str] = {}
    if output is not None:
        for r in output.results:
            if r.subquestion_id in per_sq:
                note = f" Open issues: {r.open_issues}" if r.open_issues else ""
                summaries[r.subquestion_id] = (r.summary + note).strip()

    findings = WorkerFindings(
        specialist=task.specialist, status=status, subquestion_ids=task.subquestion_ids,
        evidence_ids=evidence_ids, summaries=summaries, error=error,
        started_at=started, finished_at=finished, attempts=attempts,
    )
    ctx.log_stage("worker_done", specialist=task.specialist, status=status,
                  evidence=len(evidence_ids), seconds=findings.duration)
    return findings


async def run_parallel(ctx: AppContext, tasks: list[WorkerTask], timeout: float,
                       retries: int = 0) -> list[WorkerFindings]:
    return list(await asyncio.gather(*(_run_one(ctx, t, timeout, retries) for t in tasks)))


def timeline(workers: list[WorkerFindings]) -> dict:
    """Numbers for the sequential-vs-parallel comparison (UI chart + presentation)."""
    if not workers:
        return {"wall_seconds": 0.0, "sequential_seconds": 0.0, "speedup": 1.0, "bars": []}
    t0 = min(w.started_at for w in workers)
    t1 = max(w.finished_at for w in workers)
    wall = t1 - t0
    sequential = sum(w.duration for w in workers)
    return {
        "wall_seconds": round(wall, 1),
        "sequential_seconds": round(sequential, 1),
        "speedup": round(sequential / wall, 2) if wall > 0 and sequential > 0 else 1.0,
        "bars": [{"specialist": w.specialist, "start": round(w.started_at - t0, 1),
                  "end": round(w.finished_at - t0, 1), "status": w.status} for w in workers],
    }
