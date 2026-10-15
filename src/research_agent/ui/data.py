"""Read models for the UI: everything a screen shows comes from here, read from the real stores.

No function in this module invents a number. When something cannot be measured (no runs yet, an eval that
was never executed), it returns None and the screen shows an empty or "not measured" state instead.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import PROJECT_ROOT, Settings
from ..memory import MemoryStore
from ..models import utcnow
from ..store import Store
from ..tools.quotes import verify_quote
from ..workflows.resume import list_pending

# status → (label, tone). Tones map to CSS classes; the label always carries the meaning (not color alone).
STATUS = {
    "completed": ("Released", "ok"),
    "paused_for_review": ("Awaiting review", "warn"),
    "running": ("Running", "info"),
    "failed": ("Failed", "bad"),
    "blocked": ("Blocked at input", "bad"),
    "rejected_by_guardrail": ("Rejected by guardrail", "bad"),
    "rejected": ("Rejected by reviewer", "muted"),
}
LEGACY_STATUSES = ("drafted", "reviewed")  # statuses from early runs (kept for the audit trail)


def status_meta(status: str) -> tuple[str, str]:
    if status in STATUS:
        return STATUS[status]
    if status in LEGACY_STATUSES:
        return (f"Early build ({status})", "muted")
    return (status.replace("_", " ").capitalize(), "muted")


def _state(row) -> dict:
    try:
        return json.loads(row["state_json"] or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def ago(value: str | None, now: datetime | None = None) -> str:
    dt = parse_time(value)
    if dt is None:
        return "—"
    secs = int(((now or utcnow()) - dt).total_seconds())
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{secs // 60} min ago"
    if secs < 86400:
        return f"{secs // 3600} h ago"
    return f"{secs // 86400} d ago"


def short_time(value: str | None) -> str:
    dt = parse_time(value)
    return dt.astimezone().strftime("%b %d, %H:%M") if dt else "—"


# --------------------------------------------------------------------------- runs

@dataclass
class RunRow:
    run_id: str
    question: str
    title: str
    status: str
    created_at: str
    updated_at: str
    confidence: float | None
    evidence: int
    partial: bool
    reused: int
    session_id: str | None
    stale: bool  # "running" but not updated for a long time: the process that ran it is gone
    demo_fault: str | None = None  # a test fault was planted on purpose (scenarios 6, 9, 10)


def demo_fault_of(state: dict) -> str | None:
    fault = state.get("demo_fault") or (state.get("reliability") or {}).get("demo_fault")
    if fault:
        return fault
    # Older runs (before the fault was stored in the state): the planted fake citation always cites ev999,
    # an id the pipeline never creates, so a rejection for exactly that id identifies the demo.
    issues = (state.get("output_check") or {}).get("issues") or []
    if issues and all("ev999" in (i.get("message") or "") for i in issues):
        return "fake_citation"
    return None


def needs_attention(status: str, state: dict) -> bool:
    """Paused, failed, or rejected by a guardrail for a REAL problem. A planted demo fault that the guardrail
    stopped is the system working as designed, so it is labelled, not raised as an alarm."""
    if status in ("paused_for_review", "failed"):
        return True
    return status == "rejected_by_guardrail" and not demo_fault_of(state)


STALE_RUNNING_SECONDS = 30 * 60


def run_rows(store: Store, query: str = "", status: str = "all", limit: int = 300) -> list[RunRow]:
    counts = {r["run_id"]: r["n"] for r in store._query(
        "SELECT run_id, COUNT(*) n FROM evidence WHERE verified=1 GROUP BY run_id")}
    q = (query or "").strip().lower()
    now = utcnow()
    out = []
    filtering = bool(q) or status != "all"
    # the limit applies to what is SHOWN: with a filter, scan further back so older matches are not missed
    for row in store.latest_runs(max(limit, 2000) if filtering else limit):
        if len(out) >= limit:
            break
        state = _state(row)
        report = state.get("report") or {}
        title = report.get("title") or ""
        if q and q not in row["question"].lower() and q not in title.lower() and q not in row["id"]:
            continue
        st = row["status"]
        if status == "attention" and not needs_attention(st, state):
            continue
        if status not in ("all", "attention") and st != status and not (status == "legacy" and st in LEGACY_STATUSES):
            continue
        updated = parse_time(row["updated_at"])
        out.append(RunRow(
            run_id=row["id"], question=row["question"], title=title, status=st, created_at=row["created_at"],
            updated_at=row["updated_at"], confidence=(state.get("confidence") or {}).get("score"),
            evidence=counts.get(row["id"], 0), partial=bool((state.get("reliability") or {}).get("partial")),
            reused=len(state.get("memory_reuse") or {}), session_id=state.get("session_id"),
            stale=st == "running" and updated is not None and (now - updated).total_seconds() > STALE_RUNNING_SECONDS,
            demo_fault=demo_fault_of(state),
        ))
    return out


def overview_metrics(store: Store) -> dict[str, Any]:
    by_status = {r["status"]: r["n"] for r in store._query("SELECT status, COUNT(*) n FROM runs GROUP BY status")}
    total = sum(by_status.values())
    ms = MemoryStore(store)
    mem = ms.stats()
    verified = store._query("SELECT COUNT(*) n FROM evidence WHERE verified=1")[0]["n"]
    return {
        "runs": total,
        "completed": by_status.get("completed", 0),
        "pending_reviews": len(store.pending_reviews()),
        "needs_attention": sum(1 for r in store._query(
            "SELECT status, state_json FROM runs WHERE status IN ('failed', 'rejected_by_guardrail')")
            if needs_attention(r["status"], _state(r))),
        "demo_faults_stopped": sum(1 for r in store._query(
            "SELECT status, state_json FROM runs WHERE status='rejected_by_guardrail'") if demo_fault_of(_state(r))),
        "blocked": by_status.get("blocked", 0),
        "verified_evidence": verified,
        "memory_active": mem["evidence_by_status"].get("active", 0),
        "memory_reports": mem["reports"],
    }


@dataclass
class RunDetail:
    run_id: str
    question: str
    status: str
    created_at: str
    updated_at: str
    state: dict
    report: dict | None
    sources: dict[str, dict]
    evidence: dict[str, dict]
    review: dict | None
    final_markdown: str | None


def run_detail(store: Store, run_id: str) -> RunDetail | None:
    row = store.get_run(run_id)
    if row is None:
        return None
    state = _state(row)
    sources = {}
    for r in store._query("SELECT * FROM sources WHERE run_id=? ORDER BY rowid", (run_id,)):
        sources[r["id"]] = {k: r[k] for k in r.keys()}
    reuse = state.get("memory_reuse") or {}
    evidence = {}
    for e in store.list_evidence(run_id):
        evidence[e.id] = {**e.model_dump(mode="json"), "memory": reuse.get(e.id)}
    rv = store.get_review(run_id)
    review = rv.model_dump(mode="json") if rv else None
    return RunDetail(run_id=run_id, question=row["question"], status=row["status"], created_at=row["created_at"],
                     updated_at=row["updated_at"], state=state, report=state.get("report"), sources=sources,
                     evidence=evidence, review=review, final_markdown=state.get("final_markdown"))


def report_markdown(detail: RunDetail) -> str:
    """Text for copy / download: the released report, or a clearly marked unreleased draft."""
    if detail.final_markdown:
        return detail.final_markdown
    r = detail.report
    if not r:
        return f"# {detail.question}\n\nNo report was produced for this run (status: {detail.status})."
    label, _ = status_meta(detail.status)
    lines = [f"> **NOT RELEASED** — status: {label}. This is the last draft, kept for inspection.", "",
             f"# {r['title']}", "", "## Executive summary", r["executive_summary"], "", "## Findings"]
    for i, f in enumerate(r["findings"], 1):
        lines.append(f"{i}. {f['statement']} " + ", ".join(f"[{e}]" for e in f["evidence_ids"]) +
                     f" _(evidence: {f['strength']})_")
    if r.get("contradictions"):
        lines += ["", "## Where sources disagree"] + [f"- {c}" for c in r["contradictions"]]
    lines += ["", "## Recommendation", r["recommendation"], "", f"**Impact:** {r['impact']}"]
    if r.get("limitations"):
        lines += ["", "## Limitations"] + [f"- {x}" for x in r["limitations"]]
    lines += ["", "## Evidence appendix", ""]
    seen: set[str] = set()
    for f in r["findings"]:
        for eid in f["evidence_ids"]:
            ev = detail.evidence.get(eid)
            if ev and eid not in seen:
                seen.add(eid)
                src = detail.sources.get(ev["source_id"], {})
                where = ("offline corpus page bundled with the project, not on the live web"
                         if src.get("origin") == "offline" else src.get("url", ""))
                lines.append(f"- [{eid}] “{ev['quote']}” — {src.get('title', '?')} ({where})")
    return "\n".join(lines)


def quote_in_page(store: Store, run_id: str, evidence: dict, window: int = 420) -> dict:
    """Where the quote sits in the stored page text, re-checked now (the evidence chain, shown to the user)."""
    text = store.get_source_text(run_id, evidence["source_id"]) or ""
    ok, score = verify_quote(evidence["quote"], text)
    idx = text.lower().find(evidence["quote"][:60].lower()) if text else -1
    if idx < 0:
        return {"still_matches": ok, "score": score, "before": "", "match": "", "after": "", "has_text": bool(text)}
    end = idx + len(evidence["quote"])
    return {"still_matches": ok, "score": score, "has_text": True,
            "before": ("…" if idx > window else "") + text[max(0, idx - window):idx],
            "match": text[idx:end], "after": text[end:end + window] + ("…" if end + window < len(text) else "")}


# --------------------------------------------------------------------------- evidence explorer

def evidence_rows(store: Store, query: str = "", support: str = "all", origin: str = "all",
                  run_id: str = "all", limit: int = 800) -> list[dict]:
    reuse_by_run = {r["id"]: (_state(r).get("memory_reuse") or {}) for r in store.latest_runs(500)}
    sql = ("SELECT e.*, s.url, s.title AS source_title, s.domain_type, s.quality_score, s.origin, "
           "s.injection_detected, r.question FROM evidence e JOIN sources s ON s.run_id=e.run_id AND s.id=e.source_id "
           "JOIN runs r ON r.id=e.run_id")
    params: tuple = ()
    if run_id != "all":
        sql += " WHERE e.run_id=?"
        params = (run_id,)
    sql += " ORDER BY r.created_at DESC, e.rowid LIMIT ?"
    q = (query or "").strip().lower()
    out = []
    for r in store._query(sql, params + (limit,)):
        mem = reuse_by_run.get(r["run_id"], {}).get(r["id"])
        org = "memory" if mem else r["origin"]
        if support != "all" and r["support"] != support:
            continue
        if origin != "all" and org != origin:
            continue
        if q and q not in r["claim"].lower() and q not in r["quote"].lower() and q not in r["url"].lower():
            continue
        out.append({"key": f"{r['run_id']}/{r['id']}", "run_id": r["run_id"], "id": r["id"], "claim": r["claim"],
                    "quote": r["quote"], "url": r["url"], "source_title": r["source_title"], "source_id": r["source_id"],
                    "domain_type": r["domain_type"], "quality": r["quality_score"], "origin": org,
                    "verified": bool(r["verified"]), "support": r["support"], "status": r["status"],
                    "injection": bool(r["injection_detected"]), "retrieved_at": r["retrieved_at"],
                    "subquestion_id": r["subquestion_id"], "memory": mem, "question": r["question"]})
    return out


# --------------------------------------------------------------------------- memory

def memory_rows(store: Store, status: str = "all", claim_type: str = "all", query: str = "") -> list[dict]:
    ms = MemoryStore(store)
    now = utcnow()
    q = (query or "").strip().lower()
    out = []
    for i in ms.list(status=None if status == "all" else status,
                     claim_type=None if claim_type == "all" else claim_type):
        if q and q not in i.claim.lower() and q not in i.quote.lower() and q not in i.source_host.lower():
            continue
        d = i.as_dict()
        d["fresh"] = i.is_fresh(now)
        d["days_left"] = (i.expires_at - now).days
        out.append(d)
    return out


def memory_reports(store: Store) -> list[dict]:
    return [{k: r[k] for k in r.keys() if k not in ("report_json", "final_markdown")}
            for r in reversed(MemoryStore(store).reports())]


def memory_context(store: Store) -> list[dict]:
    return [{k: r[k] for k in r.keys()} for r in MemoryStore(store).context()]


def delete_memory(store: Store, mem_id: str) -> bool:
    return MemoryStore(store).delete(mem_id, utcnow())


# --------------------------------------------------------------------------- review queue

def pending(store: Store) -> list[dict]:
    return list_pending(store)


# --------------------------------------------------------------------------- conversations (session memory)

def conversations(store: Store, limit: int = 30) -> list[dict]:
    """Runs that belong to a conversation and can be followed up (blocked questions never enter one)."""
    seen, out = set(), []
    for row in store.latest_runs(200):
        state = _state(row)
        sid = state.get("session_id")
        if not sid or row["status"] == "blocked" or sid in seen:
            continue
        seen.add(sid)
        title = (state.get("report") or {}).get("title") or row["question"]
        out.append({"session_id": sid, "run_id": row["id"], "label": title if len(title) <= 90 else title[:89].rsplit(" ", 1)[0] + "…", "status": row["status"],
                    "created_at": row["created_at"]})
        if len(out) >= limit:
            break
    return out


# --------------------------------------------------------------------------- evaluation

EVAL_DIR = PROJECT_ROOT / "eval" / "results"


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def grounding_metrics(store: Store) -> dict[str, Any]:
    """Citation validity and grounding over every released report (shared with eval/run_eval.py)."""
    from ..evaluation import grounding_metrics as _g
    return _g(store)


def eval_results() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("guardrails", "guardrails_live", "retrieval", "memory", "tests", "routing", "routing_live",
                 "claims_live", "summary"):
        p = EVAL_DIR / f"{name}.json"
        data = _load(p)
        if data is not None:
            out[name] = {"data": data, "measured_at": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat()}
    return out


def parse_pytest_summary(text: str) -> dict[str, int]:
    counts = {}
    for n, word in re.findall(r"(\d+) (passed|failed|errors?|skipped|xfailed|xpassed)", text):
        counts["error" if word.startswith("error") else word] = int(n)
    return counts


# --------------------------------------------------------------------------- settings (read-only, no secrets)

def settings_view(settings: Settings) -> list[tuple[str, list[tuple[str, str]]]]:
    def key(v: str) -> str:
        return "set" if v else "not set"
    fallbacks = " → ".join(f"{p}/{m}" for p, m in settings.llm_fallbacks) or "none"
    return [
        ("Language model", [("Provider", settings.llm_provider), ("Model", settings.model_name),
                            ("Fallback chain", fallbacks), ("Mode", settings.llm_mode),
                            ("Rate limit", f"{settings.llm_requests_per_minute} requests/min, "
                                           f"{settings.llm_max_concurrent} concurrent"),
                            ("Gemini key", key(settings.gemini_api_key)), ("Groq key", key(settings.groq_api_key))]),
        ("Search", [("Default chain", {"tavily": "Tavily → DuckDuckGo → offline corpus", "ddg": "DuckDuckGo → offline corpus",
                                       "offline": "offline corpus"}[settings.search_mode]),
                    ("Tavily key", key(settings.tavily_api_key))]),
        ("Retrieval", [("Embeddings", settings.embed_mode), ("Ollama model", settings.embed_model),
                       ("Gemini embedding model", settings.gemini_embed_model)]),
        ("Guardrails & review", [("Review below confidence", str(settings.review_confidence_threshold)),
                                 ("Sensitive domains", ", ".join(settings.sensitive_domains)),
                                 ("Max Critic rounds", str(settings.max_critic_rounds)),
                                 ("Worker timeout", f"{settings.worker_timeout_seconds} s"),
                                 ("Worker retries", str(settings.worker_retries)),
                                 ("Independent sites needed", str(settings.min_independent_sources)),
                                 ("No verdict below coverage", f"{settings.min_coverage_for_verdict:.0%}")]),
        ("Memory", [("Long-term memory", settings.memory_mode),
                    ("Freshness (days)", ", ".join(f"{k} {v}" for k, v in settings.memory_ttl_days)),
                    ("Reuse match threshold", str(settings.memory_min_match))]),
        ("Tracing", [("Local traces", str(settings.traces_dir)),
                     ("OpenAI dashboard export", "on" if settings.trace_export_openai and settings.openai_api_key
                      else "off")]),
    ]
