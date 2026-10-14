"""Tracing setup.

The SDK's trace() and spans always run. Where they go:
- Always: a local JSONL processor (data/traces/<trace_id>.jsonl) that powers the
  Trace tab in the UI, so tracing works without paid OpenAI credits.
- Optionally: the OpenAI dashboard, when OPENAI_API_KEY is set and
  TRACE_EXPORT_OPENAI=1.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents import add_trace_processor, set_trace_processors, set_tracing_export_api_key
from agents.tracing import Span, Trace, TracingProcessor

from .config import Settings


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class LocalJsonlProcessor(TracingProcessor):
    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _write(self, trace_id: str, record: dict[str, Any]) -> None:
        path = self.directory / f"{trace_id}.jsonl"
        with self._lock, path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def on_trace_start(self, trace: Trace) -> None:
        exported = trace.export() or {}
        self._write(trace.trace_id, {"event": "trace_start", "name": trace.name, "trace_id": trace.trace_id,
                                     "group_id": exported.get("group_id"), "metadata": exported.get("metadata"),
                                     "started_at": _now_iso()})

    def on_trace_end(self, trace: Trace) -> None:
        self._write(trace.trace_id, {"event": "trace_end", "name": trace.name, "trace_id": trace.trace_id,
                                     "ended_at": _now_iso()})

    def on_span_start(self, span: Span[Any]) -> None:
        pass

    def on_span_end(self, span: Span[Any]) -> None:
        exported = span.export() or {}
        self._write(span.trace_id, {"event": "span", **exported})

    def shutdown(self) -> None:
        pass

    def force_flush(self) -> None:
        pass


_configured = False


def configure_tracing(settings: Settings) -> None:
    global _configured
    if _configured:
        return
    local = LocalJsonlProcessor(settings.traces_dir)
    if settings.trace_export_openai and settings.openai_api_key:
        set_tracing_export_api_key(settings.openai_api_key)
        add_trace_processor(local)  # keep default OpenAI exporter + local
    else:
        set_trace_processors([local])  # replace exporter: no OpenAI upload attempts
    _configured = True


def read_trace(directory: Path, trace_id: str) -> list[dict[str, Any]]:
    path = directory / f"{trace_id}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
