"""Trace page rendering: totals, a Gantt chart from real span timestamps, and a collapsible span tree."""
from __future__ import annotations

import html

from ..trace_view import SpanNode, TraceSummary
from .components import badge, stat, stats

e = html.escape

HOW = {"linked": ("Linked to this run", "ok"), "group": ("Matched by run id", "ok"),
       "time": ("Matched by time (older run, recorded before traces carried the run id)", "warn")}
KIND_LABEL = {"agent": "agent", "generation": "LLM call", "function": "tool", "custom": "step",
              "guardrail": "guardrail", "handoff": "handoff"}


def totals(summaries: list[TraceSummary], app_errors: int = 0) -> str:
    secs = sum(s.seconds for s in summaries)
    calls = sum(s.llm_calls for s in summaries)
    tin = sum(s.tokens_in for s in summaries)
    tout = sum(s.tokens_out for s in summaries)
    tools = sum(sum(s.tool_calls.values()) for s in summaries)
    # Failures the SDK did not mark as span errors (a worker that saved nothing, a fetch that returned no text)
    # are recorded by the app itself; show whichever count is larger so a failed run never reads "0 errors".
    errs = max(sum(len(s.errors) for s in summaries), app_errors)
    return stats([
        stat("Traced time", f"{secs:.0f} s"),
        stat("LLM calls", calls),
        stat("Tokens in / out", f"{tin:,} / {tout:,}" if calls else None,
             "as reported by the provider" if calls else "no LLM calls traced"),
        stat("Tool calls", tools),
        stat("Errors and fallbacks", errs, "failed workers, tools or model calls" if errs else "",
             "warn" if errs else ""),
    ])


def _gantt_rows(summary: TraceSummary) -> list[tuple[SpanNode, int]]:
    """Top-level steps plus the agents inside parallel research: the level that shows concurrency."""
    rows: list[tuple[SpanNode, int]] = []
    for root in summary.roots:
        rows.append((root, 0))
        if root.name.startswith("Parallel research") or root.name.startswith("Writer ⇄ Critic"):
            for c in root.children:
                rows.append((c, 1))
    return rows


def _failed_worker(node: SpanNode, failed_workers: set[str]) -> bool:
    return any(node.name.lower().startswith(f"{w} researcher") for w in failed_workers)


def gantt(summary: TraceSummary, failed_workers: set[str] | None = None) -> str:
    if not summary.start or not summary.end or not summary.roots:
        return ""
    total = max(summary.seconds, 0.001)
    out = []
    for node, depth in _gantt_rows(summary)[:60]:
        left = (node.start - summary.start).total_seconds() / total * 100
        width = max(0.4, node.seconds / total * 100)
        failed = bool(failed_workers) and _failed_worker(node, failed_workers)
        tone = "bad" if node.error or failed else ("model" if node.kind in ("agent", "generation") or "(agent)" in node.name
                                         else "code")
        out.append(f'<div class="rp-gantt-row d{depth}"><span class="rp-gantt-name" title="{e(node.name)}">'
                   f'{e(node.name)}</span><span class="rp-gantt-track"><span class="rp-gantt-bar rp-{tone}" '
                   f'style="left:{left:.2f}%;width:{width:.2f}%" title="{e(node.name)}: {node.seconds:.1f}s'
                   f'{" — " + e(node.error) if node.error else (" — worker failed (see Reliability)" if failed else "")}"></span></span>'
                   f'<span class="rp-gantt-val">{node.seconds:.1f} s</span></div>')
    axis = (f'<div class="rp-gantt-axis"><span>0 s</span><span>{total / 2:.0f} s</span><span>{total:.0f} s</span></div>')
    legend = ('<div class="rp-rail-legend"><span><i class="ci"></i>Agent / model</span><span><i class="sq"></i>Code step'
              '</span><span><i class="er"></i>Error</span></div>')
    return f'<div class="rp-gantt wide">{"".join(out)}{axis}{legend}</div>'


def _node(n: SpanNode, depth: int) -> str:
    kind = KIND_LABEL.get(n.kind, n.kind)
    meta = [f"{n.seconds:.2f} s"]
    if n.tokens_in or n.tokens_out:
        meta.append(f"{n.tokens_in:,} in / {n.tokens_out:,} out tokens")
    err = f'<div class="rp-span-err">{e(n.error)}</div>' if n.error else ""
    det = f'<div class="rp-span-detail">{e(n.detail)}</div>' if n.detail else ""
    head = (f'<span class="rp-span-kind k-{e(n.kind)}">{e(kind)}</span><span class="rp-span-name">{e(n.name)}</span>'
            f'<span class="rp-span-meta">{" · ".join(meta)}</span>')
    if not n.children:
        return f'<div class="rp-span leaf{" has-err" if n.error else ""}">{head}{det}{err}</div>'
    kids = "".join(_node(c, depth + 1) for c in n.children)
    opened = " open" if depth == 0 and len(n.children) < 12 else ""
    return (f'<details class="rp-span{" has-err" if n.error else ""}"{opened}><summary>{head}</summary>{det}{err}'
            f'<div class="rp-span-kids">{kids}</div></details>')


def tree(summary: TraceSummary) -> str:
    return f'<div class="rp-spans">{"".join(_node(r, 0) for r in summary.roots)}</div>'


def errors(summary: TraceSummary) -> str:
    if not summary.errors:
        return ""
    rows = "".join(f"<li><b>{e(x['span'])}</b> <span>{e(x['error'])}</span></li>" for x in summary.errors[:20])
    if any("redacted" in x["error"] for x in summary.errors):
        rows += ('<li class="rp-caption">The SDK hides tool error details; the reason the app recorded for each '
                 'failed page fetch is listed under Reliability above.</li>')
    fallback = any(code in x["error"] for x in summary.errors for code in ("503", "429", "Timeout"))
    note = ('<p class="rp-caption">A failed LLM call followed by a successful one is the model fallback chain at '
            'work (503 / 429 / timeout → next model).</p>') if fallback else ""
    return f'<div class="rp-errlist"><h4>Errors and fallbacks in this trace</h4><ul>{rows}</ul>{note}</div>'



def trace_block(summary: TraceSummary, how: str, failed_workers: set[str] | None = None) -> str:
    label, tone = HOW.get(how, (how, "muted"))
    models = ", ".join(f"{m} ×{n}" for m, n in summary.models.items()) or "none"
    tools = ", ".join(f"{t} ×{n}" for t, n in sorted(summary.tool_calls.items())) or "none"
    head = (f'<div class="rp-trace-head"><h3>{e(summary.name)}</h3>{badge(label, tone)}'
            f'<code class="rp-id">{e(summary.trace_id)}</code></div>'
            f'<p class="rp-caption">{summary.spans} spans · models: {e(models)} · tools: {e(tools)}</p>')
    return f'<section class="rp-section rp-trace">{head}{gantt(summary, failed_workers)}{errors(summary)}{tree(summary)}</section>'
