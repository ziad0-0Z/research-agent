"""AppContext: shared run state passed to tools via RunContextWrapper.

Nothing in here is sent to the LLM. Tools read it to know the run, reach the
store, and log what actually happened (the output guardrail checks that log so
the report can never claim an action no tool confirmed).
"""
from __future__ import annotations

import hashlib
import secrets
import time
import uuid
from contextvars import ContextVar
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .config import Settings
from .models import EvidenceAnalysis, ResearchReport, utcnow
from .store import Store

# Which specialist is running the current coroutine (set by workflows/parallel.py; asyncio tasks copy it,
# so every tool call inside one worker sees its own value). Used for labelled search events and demo faults.
CURRENT_SPECIALIST: ContextVar[str | None] = ContextVar("current_specialist", default=None)


@dataclass
class ToolCallRecord:
    tool: str
    ok: bool
    summary: str
    at: float = field(default_factory=time.time)
    specialist: str | None = None  # which research worker made the call (None outside parallel research)


@dataclass
class AppContext:
    settings: Settings
    store: Store
    run_id: str = field(default_factory=lambda: f"run_{uuid.uuid4().hex[:10]}")
    question: str = ""
    requires_review: bool = False
    review_reasons: list[str] = field(default_factory=list)
    review_requested: bool = False
    analysis: EvidenceAnalysis | None = None
    final_report: ResearchReport | None = None
    tool_log: list[ToolCallRecord] = field(default_factory=list)
    stage_log: list[dict[str, Any]] = field(default_factory=list)
    input_check: Any = None  # guardrails.input.InputCheck, set by the pipeline before planning
    demo_fault: str | None = None  # test-mode fault: such a run never writes to long-term memory
    clock: Callable[[], datetime] = utcnow  # injected so memory TTL tests control time
    memory_reuse: dict[str, dict] = field(default_factory=dict)  # run evidence id → memory item it came from
    session: Any = None  # agents.SQLiteSession for follow-up questions (session memory), or None
    session_id: str | None = None
    follow_up: bool = False  # the session already has earlier turns: only then does the Planner get it
    trace_ids: list[str] = field(default_factory=list)  # SDK traces of this run (pipeline + every resume)
    search_log: list[dict[str, Any]] = field(default_factory=list)  # one entry per search_web call
    reliability: dict[str, Any] = field(default_factory=dict)  # what degraded in this run (workers, search, LLM)
    _approval_token: str | None = None
    _approved_hash: str | None = None
    _counters: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.store.get_run(self.run_id) is None:
            self.store.create_run(self.run_id, self.question or "(no question)")

    def next_id(self, prefix: str) -> str:
        """Short, readable, run-unique ids: src1, ev1, ev2..."""
        n = self._counters.get(prefix, 0) + 1
        self._counters[prefix] = n
        return f"{prefix}{n}"

    def log_tool(self, tool: str, ok: bool, summary: str) -> None:
        self.tool_log.append(ToolCallRecord(tool, ok, summary[:300], specialist=CURRENT_SPECIALIST.get()))

    def tools_succeeded(self, tool: str) -> int:
        return sum(1 for r in self.tool_log if r.tool == tool and r.ok)

    def flag_review(self, reason: str) -> None:
        self.requires_review = True
        if reason not in self.review_reasons:
            self.review_reasons.append(reason)

    def log_stage(self, stage: str, **data: Any) -> None:
        self.stage_log.append({"stage": stage, "at": time.time(), **data})

    # ---- approval token: issued by Python only, after the output guardrail passes ----
    # The token is bound to the exact report it approved: if the report changes afterwards, the token
    # is void. It lives only in memory (never stored), so a resumed run must pass the checks again.
    def _report_hash(self) -> str | None:
        if self.final_report is None:
            return None
        return hashlib.sha256(self.final_report.model_dump_json().encode("utf-8")).hexdigest()

    def issue_approval_token(self) -> str:
        self._approval_token = secrets.token_hex(8)
        self._approved_hash = self._report_hash()
        return self._approval_token

    def revoke_approval_token(self) -> None:
        self._approval_token = None
        self._approved_hash = None

    def approval_token_valid(self, token: str) -> bool:
        return (bool(self._approval_token) and secrets.compare_digest(self._approval_token, token or "")
                and self._approved_hash == self._report_hash())
