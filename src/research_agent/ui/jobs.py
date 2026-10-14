"""Background jobs for the UI: research runs and review resumes, off the request thread.

One event loop in one daemon thread runs every job, so the shared rate-limited model (llm.py) always lives in
the same loop. The pipeline's real `on_event` callback feeds each job's event list; the UI polls it with a
timer. Progress is therefore exactly what the pipeline reported: nothing is simulated.

Only one job runs at a time (free-tier rate limits make parallel runs slower, not faster).
"""
from __future__ import annotations

import asyncio
import dataclasses
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..config import Settings
from ..store import Store


@dataclass
class Job:
    id: str
    kind: str                     # research | resume
    run_id: str
    question: str
    started: float = field(default_factory=time.time)
    finished: float | None = None
    events: list[tuple[float, str, dict]] = field(default_factory=list)
    status: str = "running"       # running | done | error
    outcome: str | None = None    # run status when done: completed | paused_for_review | blocked | ...
    error: str | None = None
    error_stage: str | None = None

    @property
    def elapsed(self) -> float:
        return (self.finished or time.time()) - self.started

    def add(self, stage: str, data: dict) -> None:
        self.events.append((time.time(), stage, data))


class JobManager:
    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="research-jobs", daemon=True)
        self._thread.start()
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def active(self) -> Job | None:
        with self._lock:
            return next((j for j in self.jobs.values() if j.status == "running"), None)

    def get(self, job_id: str | None) -> Job | None:
        return self.jobs.get(job_id or "")

    def _submit(self, job: Job, coro_factory) -> Job:
        with self._lock:
            if any(j.status == "running" for j in self.jobs.values()):
                raise RuntimeError("Another research job is still running. Wait for it to finish.")
            self.jobs[job.id] = job

        async def wrapper() -> None:
            try:
                job.outcome = await coro_factory()
                job.status = "done"
            except Exception as exc:  # noqa: BLE001 - every failure is shown to the user
                from ..workflows.pipeline import InputBlockedError, PipelineError
                if isinstance(exc, InputBlockedError):
                    job.status, job.outcome = "done", "blocked"
                    job.add("blocked", {"category": exc.check.category, "reason": exc.check.reason,
                                        "stage": exc.check.stage})
                else:
                    job.status = "error"
                    job.error = f"{type(exc).__name__}: {exc}"[:600]
                    job.error_stage = getattr(exc, "stage", None) if isinstance(exc, PipelineError) else None
                    job.add("job_error", {"error": job.error, "stage": job.error_stage,
                                          "trace": traceback.format_exc(limit=3)[-800:]})
            finally:
                job.finished = time.time()

        asyncio.run_coroutine_threadsafe(wrapper(), self._loop)
        return job

    def start_research(self, settings: Settings, question: str, *, live_search: bool, demo_fault: str | None,
                       session_id: str | None) -> Job:
        from ..workflows.pipeline import run_research
        run_id = f"run_{uuid.uuid4().hex[:10]}"
        job = Job(id=uuid.uuid4().hex[:8], kind="research", run_id=run_id, question=question)
        run_settings = settings if live_search else dataclasses.replace(settings, search_mode="offline")

        async def go() -> str:
            store = Store(run_settings.db_path)
            result = await run_research(question, settings=run_settings, store=store, on_event=job.add,
                                        demo_fault=demo_fault, run_id=run_id,
                                        session_id=session_id or f"s_{uuid.uuid4().hex[:10]}")
            return result.status

        return self._submit(job, go)

    def start_resume(self, settings: Settings, run_id: str, question: str) -> Job:
        from ..workflows.resume import resume_run
        job = Job(id=uuid.uuid4().hex[:8], kind="resume", run_id=run_id, question=question)

        async def go() -> str:
            result = await resume_run(run_id, settings=settings, store=Store(settings.db_path), on_event=job.add)
            return result.status

        return self._submit(job, go)


_manager: JobManager | None = None


def manager() -> JobManager:
    global _manager
    if _manager is None:
        _manager = JobManager()
    return _manager


def event_counts(job: Job) -> dict[str, Any]:
    """Live numbers for the workspace, read from the events received so far."""
    out: dict[str, Any] = {"sources": None, "evidence": None, "workers": [], "critic_rounds": 0}
    for _, stage, data in job.events:
        if stage == "research_done":
            out["workers"] = data.get("workers", [])
            out["evidence"] = sum(len(w.get("evidence_ids", [])) for w in out["workers"])
        elif stage == "critic_round":
            out["critic_rounds"] = data.get("round", out["critic_rounds"])
    return out
