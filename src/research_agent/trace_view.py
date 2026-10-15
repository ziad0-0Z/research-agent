"""Read the local JSONL traces back: find a run's traces and summarise them (Traces page).

Newer runs store their trace ids in the run state ("linked"), and new trace files carry the
run id as group_id. Older trace files have neither, so they are matched by time: a "Research run" / "Resume
run" trace whose spans fall inside the run's created..updated window. The UI says which match was used.

Everything here is read from what the SDK recorded: no timing, token count or error is estimated.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

RUN_TRACE_NAMES = ("Research run", "Resume run")


def _ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def read_records(path: Path) -> list[dict[str, Any]]:
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a half-written last line (process killed mid-write) must not hide the rest
    return out


@dataclass
class TraceIndexEntry:
    trace_id: str
    name: str
    group_id: str | None
    started: datetime | None
    ended: datetime | None
    path: Path


_index_cache: dict[str, tuple[float, TraceIndexEntry]] = {}


def index_traces(directory: Path) -> list[TraceIndexEntry]:
    """One entry per trace file, cached by modification time (files can be several hundred KB)."""
    if not directory.exists():
        return []
    entries = []
    for path in sorted(directory.glob("trace_*.jsonl")):
        mtime = path.stat().st_mtime
        cached = _index_cache.get(str(path))
        if cached and cached[0] == mtime:
            entries.append(cached[1])
            continue
        records = read_records(path)
        start = next((r for r in records if r.get("event") == "trace_start"), {})
        times = [t for r in records if r.get("event") == "span"
                 for t in (_ts(r.get("started_at")), _ts(r.get("ended_at"))) if t]
        entry = TraceIndexEntry(
            trace_id=start.get("trace_id") or path.stem, name=start.get("name", "?"), group_id=start.get("group_id"),
            started=min(times) if times else _ts(start.get("started_at")),
            ended=max(times) if times else None, path=path)
        _index_cache[str(path)] = (mtime, entry)
        entries.append(entry)
    return entries


def find_run_traces(directory: Path, run_id: str, state: dict, created_at: str, updated_at: str
                    ) -> list[tuple[TraceIndexEntry, str]]:
    """[(trace, how)] where how is linked | group | time. Oldest first."""
    entries = index_traces(directory)
    linked = set(state.get("trace_ids") or [])
    out = [(e, "linked") for e in entries if e.trace_id in linked]
    if not out:
        out = [(e, "group") for e in entries if e.group_id == run_id]
    if not out:
        lo, hi = _ts(created_at), _ts(updated_at)
        if lo and hi:
            hi = hi + timedelta(seconds=5)
            out = [(e, "time") for e in entries
                   if e.name in RUN_TRACE_NAMES and e.group_id is None and e.started and e.ended
                   and lo - timedelta(seconds=5) <= e.started and e.ended <= hi]
    return sorted(out, key=lambda x: x[0].started.timestamp() if x[0].started else 0.0)


@dataclass
class SpanNode:
    id: str
    parent_id: str | None
    kind: str          # agent | generation | function | custom | guardrail | ...
    name: str
    start: datetime
    end: datetime
    error: str | None = None
    detail: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    children: list["SpanNode"] = field(default_factory=list)

    @property
    def seconds(self) -> float:
        return max(0.0, (self.end - self.start).total_seconds())


def _span_name(sd: dict) -> str:
    kind = sd.get("type", "?")
    if kind == "generation":
        return f"LLM call · {sd.get('model') or 'model'}"
    if kind == "custom" and sd.get("name") == "turn":
        data = sd.get("data") or {}
        return f"Turn {data.get('turn', '?')} · {data.get('agent_name', '')}".strip()
    if kind == "custom" and sd.get("name") == "task":
        return "Agent workflow"
    return sd.get("name") or kind


def _detail(sd: dict) -> str:
    kind = sd.get("type")
    if kind == "function":
        return str(sd.get("input") or "")[:220]
    if kind == "custom" and sd.get("data") and sd.get("name") not in ("turn", "task"):
        return json.dumps(sd["data"], default=str)[:220]
    if kind == "guardrail":
        return "triggered" if sd.get("triggered") else "passed"
    return ""


def _error(rec: dict) -> str | None:
    err = rec.get("error")
    if not err:
        return None
    msg = err.get("message", "error") if isinstance(err, dict) else str(err)
    data = err.get("data") if isinstance(err, dict) else None
    if isinstance(data, dict) and data.get("error"):
        msg = f"{msg}: {str(data['error'])[:160]}"
    return msg[:240]


@dataclass
class TraceSummary:
    trace_id: str
    name: str
    roots: list[SpanNode]
    start: datetime | None
    end: datetime | None
    llm_calls: int
    tokens_in: int
    tokens_out: int
    tool_calls: dict[str, int]
    errors: list[dict]
    models: dict[str, int]
    spans: int

    @property
    def seconds(self) -> float:
        return (self.end - self.start).total_seconds() if self.start and self.end else 0.0


def summarize(records: list[dict[str, Any]]) -> TraceSummary:
    start_rec = next((r for r in records if r.get("event") == "trace_start"), {})
    nodes: dict[str, SpanNode] = {}
    tool_calls: dict[str, int] = {}
    models: dict[str, int] = {}
    errors: list[dict] = []
    tin = tout = calls = 0
    for r in records:
        if r.get("event") != "span":
            continue
        sd = r.get("span_data") or {}
        s, e = _ts(r.get("started_at")), _ts(r.get("ended_at"))
        if not s or not e:
            continue
        node = SpanNode(id=r.get("id", ""), parent_id=r.get("parent_id"), kind=sd.get("type", "?"),
                        name=_span_name(sd), start=s, end=e, error=_error(r), detail=_detail(sd))
        if node.kind == "generation":
            calls += 1
            usage = sd.get("usage") or {}
            node.tokens_in, node.tokens_out = int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)
            tin, tout = tin + node.tokens_in, tout + node.tokens_out
            m = sd.get("model") or "?"
            models[m] = models.get(m, 0) + 1
        if node.kind == "function":
            tool_calls[node.name] = tool_calls.get(node.name, 0) + 1
        if node.error and node.kind in ("generation", "function", "agent"):
            errors.append({"span": node.name, "kind": node.kind, "at": s.isoformat(), "error": node.error})
        nodes[node.id] = node
    roots = []
    for n in nodes.values():
        parent = nodes.get(n.parent_id or "")
        (parent.children if parent else roots).append(n)
    for n in nodes.values():
        n.children.sort(key=lambda c: c.start)
        if n.name == "Agent workflow":  # the SDK's task span: name it after the agent it ran
            agent = next((c for c in n.children if c.kind == "agent"), None)
            if agent:
                n.name = f"{agent.name} (agent)"
    roots.sort(key=lambda c: c.start)
    starts = [n.start for n in nodes.values()]
    ends = [n.end for n in nodes.values()]
    return TraceSummary(
        trace_id=start_rec.get("trace_id", ""), name=start_rec.get("name", "?"), roots=roots,
        start=min(starts) if starts else None, end=max(ends) if ends else None, llm_calls=calls,
        tokens_in=tin, tokens_out=tout, tool_calls=tool_calls, errors=errors, models=models, spans=len(nodes))


def load_summary(entry: TraceIndexEntry) -> TraceSummary:
    return summarize(read_records(entry.path))
