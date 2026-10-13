"""AppContext: shared run state passed to tools via RunContextWrapper.

Nothing in here is sent to the LLM. Tools read it to know the run, reach the
store, and log what actually happened (the output guardrail checks that log so
the report can never claim an action no tool confirmed).
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .config import Settings
from .store import Store


@dataclass
class ToolCallRecord:
    tool: str
    ok: bool
    summary: str
    at: float = field(default_factory=time.time)


@dataclass
class AppContext:
    settings: Settings
    store: Store
    run_id: str = field(default_factory=lambda: f"run_{uuid.uuid4().hex[:10]}")
    question: str = ""
    requires_review: bool = False
    review_reasons: list[str] = field(default_factory=list)
    approval_token: str | None = None
    tool_log: list[ToolCallRecord] = field(default_factory=list)
    stage_log: list[dict[str, Any]] = field(default_factory=list)
    _counters: dict[str, int] = field(default_factory=dict)

    def next_id(self, prefix: str) -> str:
        """Short, readable, run-unique ids: src1, ev1, ev2..."""
        n = self._counters.get(prefix, 0) + 1
        self._counters[prefix] = n
        return f"{prefix}{n}"

    def log_tool(self, tool: str, ok: bool, summary: str) -> None:
        self.tool_log.append(ToolCallRecord(tool, ok, summary[:300]))

    def flag_review(self, reason: str) -> None:
        self.requires_review = True
        if reason not in self.review_reasons:
            self.review_reasons.append(reason)

    def log_stage(self, stage: str, **data: Any) -> None:
        self.stage_log.append({"stage": stage, "at": time.time(), **data})
