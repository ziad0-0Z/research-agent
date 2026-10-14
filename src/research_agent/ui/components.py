"""HTML building blocks for the UI (design system: styles.css).

Every value that came from a model, a web page or a user is escaped here: page text is untrusted.
Components only render data they are given; they never compute or invent a metric.
"""
from __future__ import annotations

import html
from typing import Any, Iterable
from urllib.parse import urlparse

from .data import ago, short_time, status_meta

e = html.escape


# --------------------------------------------------------------------------- atoms

def badge(label: str, tone: str = "muted", title: str = "") -> str:
    t = f' title="{e(title)}"' if title else ""
    return f'<span class="rp-badge rp-{tone}"{t}><i></i>{e(label)}</span>'


def status_badge(status: str) -> str:
    label, tone = status_meta(status)
    return badge(label, tone)


def mono(text: str) -> str:
    return f'<code class="rp-id">{e(text)}</code>'


SUPPORT = {"supported": ("Supported", "ok"), "partial": ("Partly supported", "warn"),
           "not_supported": ("Not supported", "bad"), "unchecked": ("Not judged", "muted"),
           "unjudged": ("Not judged", "muted")}
STRENGTH = {"strong": ("Strong", "ok"), "moderate": ("Moderate", "info"), "weak": ("Weak", "warn")}
WORKER = {"success": ("Complete", "ok"), "partial": ("Partial", "warn"), "failed": ("Failed", "bad"),
          "timeout": ("Timed out", "bad"), "running": ("Researching", "info")}
MEM_STATUS = {"active": ("Active", "ok"), "superseded": ("Superseded", "muted"),
              "unverifiable": ("Unverifiable", "bad"), "retired": ("Retired", "muted")}
ORIGIN = {"tavily": "Tavily", "ddg": "DuckDuckGo", "offline": "Offline corpus", "memory": "Memory"}


def support_badge(level: str) -> str:
    label, tone = SUPPORT.get(level, (level, "muted"))
    return badge(label, tone)


def empty(title: str, body: str = "", tone: str = "") -> str:
    return (f'<div class="rp-empty {tone}"><div class="rp-empty-mark"></div><h4>{e(title)}</h4>'
            f'{f"<p>{e(body)}</p>" if body else ""}</div>')


def notice(text: str, tone: str = "info", title: str = "") -> str:
    head = f"<strong>{e(title)}</strong> " if title else ""
    return f'<div class="rp-notice rp-{tone}">{head}{e(text)}</div>'


def notice_html(inner: str, tone: str = "info") -> str:
    return f'<div class="rp-notice rp-{tone}">{inner}</div>'


def meter(value: float | None, threshold: float | None = None) -> str:
    if value is None:
        return '<span class="rp-meter-none">not computed</span>'
    pct = max(0.0, min(1.0, float(value))) * 100
    tone = "ok" if value >= 0.7 else "warn" if value >= (threshold or 0.6) else "bad"
    mark = (f'<span class="rp-meter-mark" style="left:{threshold * 100:.0f}%" title="review threshold {threshold}"></span>'
            if threshold is not None else "")
    return (f'<span class="rp-meter rp-{tone}"><span class="rp-meter-track"><span class="rp-meter-fill" '
            f'style="width:{pct:.0f}%"></span>{mark}</span><b>{value:.2f}</b></span>')


def kv(rows: Iterable[tuple[str, str]], raw: bool = False) -> str:
    items = "".join(f"<dt>{e(k)}</dt><dd>{v if raw else e(str(v))}</dd>" for k, v in rows)
    return f'<dl class="rp-kv">{items}</dl>'


def section(title: str, body: str, aside: str = "", cls: str = "") -> str:
    a = f'<span class="rp-section-aside">{aside}</span>' if aside else ""
    return f'<section class="rp-section {cls}"><header><h3>{e(title)}</h3>{a}</header>{body}</section>'


def stat(label: str, value: Any, note: str = "", tone: str = "") -> str:
    shown = "—" if value is None else e(str(value))
    return (f'<div class="rp-stat {tone}"><span class="rp-stat-value">{shown}</span>'
            f'<span class="rp-stat-label">{e(label)}</span>{f"<span class=rp-stat-note>{e(note)}</span>" if note else ""}</div>')


def stats(items: list[str], cls: str = "") -> str:
    return f'<div class="rp-stats {cls}">{"".join(items)}</div>'


def page_header(title: str, context: str, crumbs: list[str] | None = None, right: str = "") -> str:
    crumb = ""
    if crumbs:
        crumb = '<nav class="rp-crumbs">' + "<span>/</span>".join(e(c) for c in crumbs) + "</nav>"
    return (f'<div class="rp-pagehead"><div>{crumb}<h1>{e(title)}</h1><p>{e(context)}</p></div>'
            f'<div class="rp-pagehead-right">{right}</div></div>')


def urlhost(url: str) -> str:
    return urlparse(url).hostname or url[:40]


def is_live_url(url: str) -> bool:
    host = urlparse(url).hostname or ""
    return bool(host) and not host.endswith(".example") and urlparse(url).scheme in ("http", "https")


def source_link(url: str, title: str, origin: str) -> str:
    if is_live_url(url) and origin != "offline":
        return (f'<a class="rp-src" href="{e(url)}" target="_blank" rel="noopener noreferrer">{e(title or url)}</a>'
                f'<span class="rp-host">{e(urlparse(url).hostname or "")}</span>')
    return (f'<span class="rp-src rp-src-offline" title="{e(url)}">{e(title or url)}</span>'
            f'<span class="rp-host">offline corpus page (bundled with the project, not on the live web)</span>')


# --------------------------------------------------------------------------- pipeline rail (signature element)
# Square node = Python decides (code). Round node = a model decides. Mixed = both (e.g. guardrail).

STAGES = [
    ("guard", "Input guardrail", "mixed", ("input_checked",)),
    ("route", "Router", "code", ("routed",)),
    ("memory", "Memory", "code", ("memory_related", "memory_reused")),
    ("plan", "Planner", "model", ("planned",)),
    ("delegate", "Orchestrator", "model", ("delegated",)),
    ("research", "Specialists", "model", ("research_started", "research_done")),
    ("analyse", "Evidence analyst", "model", ("analysed", "analysis_failed")),
    ("write", "Writer", "model", ("drafted",)),
    ("critic", "Critic loop", "mixed", ("critic_round", "critic_done")),
    ("release", "Release checks", "code", ("checked", "output_checked", "gate")),
    ("outcome", "Outcome", "code", ("completed", "paused", "rejected", "blocked", "job_error")),
]
RESUME_STAGES = [
    ("decision", "Human decision", "code", ("resumed",)),
    ("revise", "Writer revision", "model", ("revised",)),
    ("critic", "Critic loop", "mixed", ("critic_round", "critic_done")),
    ("release", "Release checks", "code", ("output_checked", "gate")),
    ("outcome", "Outcome", "code", ("completed", "paused", "rejected", "rejected_by_human", "job_error")),
]
DONE_WHEN = {"research": "research_done", "critic": "critic_done"}


def stage_states(events: list[str], running: bool, kind: str = "research") -> list[tuple[str, str, str, str]]:
    """[(key, label, decider, state)] with state done | active | skipped | pending | failed."""
    stages = RESUME_STAGES if kind == "resume" else STAGES
    seen = set(events)
    reached = []
    for key, _, _, evs in stages:
        done_ev = DONE_WHEN.get(key)
        started = any(ev in seen for ev in evs)
        finished = (done_ev in seen) if done_ev else started
        reached.append((started, finished))
    last_idx = max((i for i, (s, _) in enumerate(reached) if s), default=-1)
    out = []
    failed = "job_error" in seen or "blocked" in seen or "rejected" in seen
    for i, ((key, label, decider, _), (started, finished)) in enumerate(zip(stages, reached)):
        if finished and not (key == "outcome" and failed):
            state = "done"
        elif key == "outcome" and failed:
            state = "failed"
        elif started:
            state = "active" if running else "done"
        elif i < last_idx:
            state = "skipped"
        else:
            state = "pending"
        out.append((key, label, decider, state))
    if running:  # the first pending stage after the last reached one is what is happening now
        nxt = next((i for i, s in enumerate(out) if s[3] == "pending"), None)
        if nxt is not None and not any(s[3] == "active" for s in out):
            k, lab, d, _ = out[nxt]
            out[nxt] = (k, lab, d, "active")
    return out


def rail(states: list[tuple[str, str, str, str]], legend: bool = True, show_state: bool = True) -> str:
    nodes = []
    for key, label, decider, state in states:
        word = f'<span class="rp-node-state">{e(STATE_WORD[state])}</span>' if show_state else ""
        nodes.append(f'<li class="rp-node {state} {decider}" title="{e(label)}: {state}"><span class="rp-dot"></span>'
                     f'<span class="rp-node-label">{e(label)}</span>{word}</li>')
    leg = ('<div class="rp-rail-legend"><span><i class="sq"></i>Python decides</span>'
           '<span><i class="ci"></i>A model decides</span><span><i class="mx"></i>Both</span></div>') if legend else ""
    return f'<div class="rp-rail"><ol>{"".join(nodes)}</ol>{leg}</div>'


STATE_WORD = {"done": "done", "active": "in progress", "skipped": "not run", "pending": "waiting", "failed": "stopped"}


# --------------------------------------------------------------------------- event feed

def event_line(stage: str, data: dict) -> tuple[str, str] | None:
    """(text, tone) for the live feed. Text is built from the event payload the pipeline emitted."""
    d = data
    if stage == "input_checked":
        return (f"Input guardrail ({d.get('checked_by')}): {'allowed' if d.get('allowed') else 'blocked'} as "
                f"{d.get('category')}" + (f", domain {d['domain']}" if d.get("domain") else ""),
                "ok" if d.get("allowed") else "bad")
    if stage == "routed":
        return (f"Router: {d.get('path')} path" + (f" — sensitive domain {', '.join(d['domains'])}" if d.get("domains") else ""),
                "warn" if d.get("domains") else "")
    if stage == "memory_related":
        return (f"Memory: {len(d.get('reports', []))} related past report(s), shown to you only", "")
    if stage == "memory_reused":
        return (f"Memory: {d.get('reused')} remembered evidence item(s) re-verified into this run", "")
    if stage == "planned":
        return (f"Planner: {len(d.get('subquestions', []))} subquestions", "")
    if stage == "delegated":
        a = d.get("assignments", {})
        return (f"Orchestrator: {', '.join(f'{k} ({len(v)})' for k, v in a.items())} at {d.get('depth')} depth", "")
    if stage == "research_started":
        return (f"{len(d.get('workers', []))} specialists researching in parallel", "info")
    if stage == "research_done":
        ws = d.get("workers", [])
        bad = [w["specialist"] for w in ws if w["status"] != "success"]
        s = (d.get("reliability") or {}).get("search") or {}
        extra = f"; {s['fallback_calls']} searches used a fallback provider" if s.get("fallback_calls") else ""
        return (f"Research finished: {sum(len(w.get('evidence_ids', [])) for w in ws)} verified evidence items"
                + (f"; degraded: {', '.join(bad)}" if bad else "") + extra, "warn" if bad or extra else "ok")
    if stage == "analysed":
        return (f"Evidence analyst: {d.get('agreements')} agreements, {d.get('contradictions')} contradictions", "")
    if stage == "analysis_failed":
        return ("Evidence analysis failed; the report continues without it", "warn")
    if stage == "drafted":
        return (f"Writer: first draft with {d.get('findings')} findings, impact {d.get('impact')}", "")
    if stage == "fault_planted":
        return (f"Demo fault planted: {d.get('kind')}", "warn")
    if stage == "critic_round":
        return (f"Critic round {d.get('round')}: {d.get('verdict')} ({d.get('blocking')} blocking issues)",
                {"PASS": "ok", "REVISE": "warn", "FAIL": "bad"}.get(d.get("verdict"), ""))
    if stage == "revised":
        return (f"Writer revision {d.get('round')}: {d.get('findings')} findings", "")
    if stage == "critic_done":
        return (f"Critic loop finished: {d.get('verdict')} after {d.get('rounds')} round(s)", "")
    if stage == "checked":
        return (f"Confidence computed in code: {(d.get('confidence') or {}).get('score')}", "")
    if stage == "output_checked":
        return ("Output guardrail passed" if d.get("passed") else
                f"Output guardrail rejected the report ({len(d.get('issues', []))} issue(s))",
                "ok" if d.get("passed") else "bad")
    if stage == "gate":
        if d.get("needs_review"):
            return ("Impact gate: human review required — " + "; ".join(d.get("reasons", []))[:220], "warn")
        return ("Impact gate: " + ("approved by a human" if d.get("approved") else "no review needed"), "ok")
    if stage == "paused":
        return ("Paused for human review — the report is held until someone decides", "warn")
    if stage == "completed":
        return ("Report released", "ok")
    if stage == "memory_committed":
        return ((f"Long-term memory: +{d.get('added')} evidence" if d.get("committed")
                 else f"Long-term memory: nothing written ({d.get('reason')})"), "")
    if stage == "rejected":
        return ("Nothing released: the output guardrail rejected the report", "bad")
    if stage == "blocked":
        return (f"Blocked by the input guardrail ({d.get('category')}): {d.get('reason')}", "bad")
    if stage == "resumed":
        return (f"Resuming after the reviewer's decision: {d.get('decision')}", "info")
    if stage == "rejected_by_human":
        return ("Rejected by the reviewer; nothing released", "muted")
    if stage == "job_error":
        return (f"Stopped{' at ' + d['stage'] if d.get('stage') else ''}: {d.get('error')}", "bad")
    return None


def feed(events: list[tuple[float, str, dict]], t0: float) -> str:
    rows = []
    for t, stage, data in events:
        line = event_line(stage, data)
        if line is None:
            continue
        text, tone = line
        rows.append(f'<li class="rp-feed-{tone or "plain"}"><time>{t - t0:5.0f}s</time><span>{e(text)}</span></li>')
    if not rows:
        return '<ol class="rp-feed"><li class="rp-feed-plain"><time>   0s</time><span>Starting…</span></li></ol>'
    return f'<ol class="rp-feed">{"".join(reversed(rows))}</ol>'


def worker_cards(workers: list[dict], running_names: list[str] | None = None) -> str:
    cards = []
    for w in workers:
        label, tone = WORKER.get(w.get("status", ""), (w.get("status", "?"), "muted"))
        dur = (w.get("finished_at") or 0) - (w.get("started_at") or 0)
        note = ""
        if w.get("error"):
            err = w["error"]
            if w.get("status") == "success" and "MaxTurns" in err:
                # parallel.py judges success by saved evidence: the turn limit hit after that is only a note
                note = '<p class="rp-worker-note rp-note-quiet">Stopped at its turn limit after covering every subquestion.</p>'
            else:
                note = f'<p class="rp-worker-note">{e(err[:140])}</p>'

        cards.append(f'<div class="rp-worker"><div class="rp-worker-top"><b>{e(w["specialist"].capitalize())}</b>'
                     f'{badge(label, tone)}</div><div class="rp-worker-meta">{len(w.get("evidence_ids", []))} evidence · '
                     f'{dur:.0f} s · {e(", ".join(w.get("subquestion_ids", [])))}</div>{note}</div>')
    for name in running_names or []:
        cards.append(f'<div class="rp-worker"><div class="rp-worker-top"><b>{e(name.capitalize())}</b>'
                     f'{badge("Researching", "info")}</div><div class="rp-worker-meta">searching, reading, verifying quotes'
                     f'</div></div>')
    return f'<div class="rp-workers">{"".join(cards)}</div>' if cards else ""


def timeline_bars(tl: dict | None) -> str:
    if not tl or not tl.get("bars"):
        return ""
    total = max(b["end"] for b in tl["bars"]) or 1
    rows = []
    for b in tl["bars"]:
        label, tone = WORKER.get(b["status"], (b["status"], "muted"))
        left, width = b["start"] / total * 100, max(1.5, (b["end"] - b["start"]) / total * 100)
        rows.append(f'<div class="rp-gantt-row"><span class="rp-gantt-name">{e(b["specialist"])}</span>'
                    f'<span class="rp-gantt-track"><span class="rp-gantt-bar rp-{tone}" style="left:{left:.1f}%;'
                    f'width:{width:.1f}%" title="{e(label)} {b["end"] - b["start"]:.1f}s"></span></span>'
                    f'<span class="rp-gantt-val">{b["end"] - b["start"]:.0f} s</span></div>')
    caption = (f'Wall time {tl["wall_seconds"]} s; the same work one after another would take '
               f'{tl["sequential_seconds"]} s ({tl["speedup"]}× faster in parallel).')
    return f'<div class="rp-gantt">{"".join(rows)}<p class="rp-caption">{e(caption)}</p></div>'


# --------------------------------------------------------------------------- report

def _cite_chip(eid: str, ev: dict | None, src: dict | None) -> str:
    if ev is None:
        return f'<span class="rp-cite rp-cite-bad" title="This evidence id does not exist in the run">{e(eid)}</span>'
    tip = f'“{ev["quote"][:260]}” — {src.get("title", "") if src else ""}'
    mem = " rp-cite-mem" if ev.get("memory") else ""
    return f'<a class="rp-cite{mem}" href="#ev-{e(eid)}" title="{e(tip)}">{e(eid)}</a>'


def report_body(detail, compact: bool = False) -> str:
    """The reading view: built from the stored report JSON, evidence and sources of the run."""
    r = detail.report
    if not r:
        return empty("No report for this run", _no_report_reason(detail))
    ev, src = detail.evidence, detail.sources
    parts = [f'<article class="rp-report{" compact" if compact else ""}">']
    if not compact:  # compact views already show the title in their own header
        parts.append(f'<h2 class="rp-report-title">{e(r["title"])}</h2>')
    parts.append(f'<p class="rp-lede">{e(r["executive_summary"])}</p>')
    parts.append('<h3>Findings</h3><ol class="rp-findings">')
    for f in r["findings"]:
        sl, st = STRENGTH.get(f["strength"], (f["strength"], "muted"))
        chips = "".join(_cite_chip(i, ev.get(i), src.get(ev[i]["source_id"]) if i in ev else None)
                        for i in f["evidence_ids"])
        parts.append(f'<li><p>{e(f["statement"])}</p><div class="rp-finding-meta">{chips}'
                     f'<span class="rp-strength rp-{st}">{e(sl)} evidence</span></div></li>')
    parts.append("</ol>")
    if r.get("contradictions"):
        parts.append('<h3>Where sources disagree</h3><ul class="rp-list">' +
                     "".join(f"<li>{e(c)}</li>" for c in r["contradictions"]) + "</ul>")
    parts.append(f'<div class="rp-reco"><h3>Recommendation</h3><p>{e(r["recommendation"])}</p>'
                 f'<span>Impact if acted on: <b>{e(r["impact"])}</b></span></div>')
    if r.get("limitations"):
        parts.append('<h3>Limitations and missing information</h3><ul class="rp-list">' +
                     "".join(f"<li>{e(x)}</li>" for x in r["limitations"]) + "</ul>")
    rel = detail.state.get("reliability")
    from ..workflows.reliability import coverage_lines
    cov = coverage_lines(rel)
    if cov:
        parts.append('<h3>Research coverage</h3><ul class="rp-list rp-cov">' +
                     "".join(f"<li>{e(x.replace('**', ''))}</li>" for x in cov) + "</ul>")
    if not compact:
        parts.append(appendix(detail, r))
    parts.append("</article>")
    return "".join(parts)


def appendix(detail, r: dict) -> str:
    cited = []
    for f in r["findings"]:
        for i in f["evidence_ids"]:
            if i not in cited:
                cited.append(i)
    cited.sort(key=lambda x: int(x[2:]) if x[2:].isdigit() else 10**6)
    rows = []
    for i in cited:
        ev = detail.evidence.get(i)
        if ev is None:
            rows.append(f'<li id="ev-{e(i)}" class="rp-app-bad">{mono(i)} <span>cited but does not exist in this run'
                        f'</span></li>')
            continue
        s = detail.sources.get(ev["source_id"], {})
        mem = ev.get("memory")
        memo = (f'<span class="rp-memnote">Reused from memory · last verified '
                f'{e(short_time(mem.get("last_verified_at")))} ({e(mem.get("freshness", ""))})</span>') if mem else ""
        rows.append(
            f'<li id="ev-{e(i)}"><div class="rp-app-head">{mono(i)}{support_badge(ev["support"])}'
            f'{badge("Quote verified" if ev["verified"] else "Unverified", "ok" if ev["verified"] else "bad")}</div>'
            f'<blockquote>“{e(ev["quote"])}”</blockquote>'
            f'<div class="rp-app-src">{source_link(s.get("url", ""), s.get("title", ""), s.get("origin", ""))}'
            f'<span class="rp-host">{e(s.get("domain_type", ""))} · quality {s.get("quality_score", "—")}</span>{memo}</div></li>')
    return f'<h3>Evidence appendix</h3><ol class="rp-appendix">{"".join(rows)}</ol>'


def _no_report_reason(detail) -> str:
    st = detail.status
    if st == "blocked":
        ic = detail.state.get("input_check") or {}
        return f"The input guardrail blocked this question ({ic.get('category', '?')}): {ic.get('reason', '')}. No agent ran."
    if st == "failed":
        err = detail.state.get("error") or {}
        return f"The run stopped at stage “{err.get('stage', '?')}”: {err.get('message', 'no error recorded')}"
    if st == "running":
        return "This run has not produced a report. If it is not running now, the process that ran it was stopped."
    return f"No report was stored for this run (status: {st})."


def outcome_banner(detail) -> str:
    st = detail.status
    st_ = detail.state
    if st == "completed":
        hist = st_.get("review_history") or []
        if hist and hist[-1]["decision"] == "approved":
            return notice_html(f'<strong>Released after human approval.</strong> {e(hist[-1].get("note") or "")}', "ok")
        return notice_html("<strong>Released.</strong> Passed the Critic, the output guardrail and the impact gate.", "ok")
    if st == "paused_for_review":
        reasons = "; ".join(st_.get("gate_reasons") or st_.get("review_reasons") or [])
        return notice_html(f"<strong>Held for human review.</strong> {e(reasons)}. Decide in the Review queue.", "warn")
    if st == "rejected_by_guardrail":
        issues = (st_.get("output_check") or {}).get("issues") or []
        lis = "".join(f"<li><b>{e(i.get('check', ''))}</b> {e(i.get('message', ''))}</li>" for i in issues)
        return notice_html(f"<strong>Not released: the output guardrail rejected this report.</strong>"
                           f"<ul>{lis}</ul>Shown below as the last draft, for inspection only.", "bad")
    if st == "rejected":
        hist = st_.get("review_history") or []
        note = hist[-1].get("note", "") if hist else ""
        return notice_html(f"<strong>Rejected by a human reviewer.</strong> {e(note)} Nothing was released.", "muted")
    if st == "failed":
        return notice_html(f"<strong>Run failed.</strong> {e(_no_report_reason(detail))}", "bad")
    if st == "blocked":
        return notice_html(f"<strong>Blocked at input.</strong> {e(_no_report_reason(detail))}", "bad")
    if st == "running":
        return notice_html("<strong>No result recorded.</strong> " + e(_no_report_reason(detail)), "muted")
    return notice_html(f"<strong>Recorded by an early build</strong> (status “{e(st)}”), before release checks existed. "
                       "Shown for the record; it was never released.", "muted")


# --------------------------------------------------------------------------- evidence chain (detail panel)

def evidence_chain(item: dict, page: dict | None) -> str:
    """claim → quote → stored page → source. The quote is re-checked against the page text right now."""
    ctx = ""
    if page and page.get("has_text") and page.get("match"):
        ctx = (f'<div class="rp-pagetext"><div class="rp-pagetext-head">Stored page text '
               f'{badge("quote still matches" if page["still_matches"] else "quote no longer matches", "ok" if page["still_matches"] else "bad")}'
               f'</div><p>{e(page["before"])}<mark>{e(page["match"])}</mark>{e(page["after"])}</p></div>')
    elif page and page.get("has_text"):
        ctx = notice("The quote was verified when saved; its exact position in the page text could not be located "
                     "for display.", "muted")
    mem = item.get("memory")
    memo = ""
    if mem:
        memo = notice_html(f"<strong>From long-term memory</strong> ({e(mem.get('mem_id', ''))}): found in earlier "
                           f"research, last verified {e(short_time(mem.get('last_verified_at')))}, reuse was "
                           f"{e(mem.get('freshness', ''))}. It entered this run through save_evidence, so it was "
                           "verified again like any new quote.", "info")
    flags = [support_badge(item["support"]),
             badge("Quote verified" if item["verified"] else "Unverified", "ok" if item["verified"] else "bad"),
             badge(ORIGIN.get(item["origin"], item["origin"]), "muted")]
    if item.get("injection"):
        flags.append(badge("Injection detected in page", "bad"))
    if item.get("status") and item["status"] != "active":
        flags.append(badge(item["status"].capitalize(), "muted"))
    return (f'<div class="rp-chain"><div class="rp-chain-step"><span>Claim</span><p>{e(item["claim"])}</p></div>'
            f'<div class="rp-chain-step"><span>Verbatim quote</span><blockquote>“{e(item["quote"])}”</blockquote></div>'
            f'<div class="rp-chain-flags">{"".join(flags)}</div>{ctx}{memo}'
            f'<div class="rp-chain-step"><span>Source</span><p>{source_link(item["url"], item["source_title"], item["origin"] if item["origin"] != "memory" else "")}'
            f'<span class="rp-host">{e(item["domain_type"])} · quality {item["quality"]}</span></p></div>'
            f'{kv([("Evidence", item["id"]), ("Run", item["run_id"]), ("Subquestion", item["subquestion_id"]), ("Saved", short_time(item["retrieved_at"]))])}'
            f'</div>')


# --------------------------------------------------------------------------- run lists

def run_table_rows(rows) -> list[list[str]]:
    out = []
    for r in rows:
        label, tone = status_meta(r.status)
        flags = []
        if r.partial:
            flags.append(badge("Partial", "warn"))
        if r.reused:
            flags.append(badge(f"{r.reused} from memory", "info"))
        if r.stale:
            flags.append(badge("Interrupted", "muted"))
        q = r.title or r.question
        out.append([f'<div class="rp-cell-q"><b>{e(q)}</b>{"<span>" + e(r.question) + "</span>" if r.title else ""}</div>',
                    badge(label, tone) + "".join(flags),
                    "—" if r.confidence is None else f"{r.confidence:.2f}",
                    str(r.evidence), f'<span title="{e(r.created_at)}">{e(ago(r.created_at))}</span>',
                    f'<code class="rp-id">{e(r.run_id)}</code>'])
    return out


def attention_list(pending: list[dict], attention_rows) -> str:
    items = []
    for p in pending:
        items.append(f'<li><div>{badge("Awaiting review", "warn")}<b>{e(p.get("title") or p["question"])}</b></div>'
                     f'<p>{e("; ".join(p["reasons"]))}</p></li>')
    for r in attention_rows:
        if r.status == "paused_for_review":
            continue
        label, tone = status_meta(r.status)
        items.append(f'<li><div>{badge(label, tone)}<b>{e(r.title or r.question)}</b></div>'
                     f'<p>{e(ago(r.created_at))} · {mono(r.run_id)}</p></li>')
    if not items:
        return empty("Nothing needs attention", "No paused, failed or rejected runs.")
    return f'<ul class="rp-attention">{"".join(items[:6])}</ul>'
