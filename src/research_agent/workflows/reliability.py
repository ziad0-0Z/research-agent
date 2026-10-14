"""Reliability record (Phase 7): what degraded during a run, measured in code, and how the report says so.

    workers      success / partial / failed / timeout per specialist (from saved evidence, parallel.py)
    search       which provider answered each search_web call; fallbacks; searches where every provider failed
    llm          model fallbacks during this run (503 / 429 / timeout → next model)
    embeddings   embedding-provider failures (hybrid retrieval falls back to BM25)

The record is saved in the run state (UI + audit) and the report gets a "Research coverage" section
whenever research was partial or a fallback answered, so a degraded report is labelled in its own text.
"""
from __future__ import annotations

from typing import Any

from ..context import AppContext
from ..models import WorkerFindings


def _model_events() -> list[dict]:
    from .. import llm
    return list(llm._model.fallback_events) if llm._model is not None else []


def _embed_events() -> list[dict]:
    from ..rag import embeddings
    return list(embeddings._chain.events) if embeddings._chain is not None else []


def snapshot() -> dict[str, int]:
    """Counts at run start, so the record keeps only this run's events (the model is shared per process)."""
    return {"llm": len(_model_events()), "embed": len(_embed_events())}


def build_reliability(ctx: AppContext, workers: list[WorkerFindings], start: dict[str, int] | None = None) -> dict:
    start = start or {"llm": 0, "embed": 0}
    by_provider: dict[str, int] = {}
    unavailable: dict[str, None] = {}  # chain order, no duplicates
    for e in ctx.search_log:
        if e["provider"]:
            by_provider[e["provider"]] = by_provider.get(e["provider"], 0) + 1
        unavailable.update(dict.fromkeys(e["unavailable"]))
    fallback_calls = sum(1 for e in ctx.search_log if e["fallback_used"] and not e["failed"])
    failed_calls = sum(1 for e in ctx.search_log if e["failed"])
    llm_events = _model_events()[start["llm"]:]
    embed_events = _embed_events()[start["embed"]:]
    degraded_workers = [w for w in workers if w.status != "success"]
    return {
        "partial": bool(degraded_workers),
        "workers": [{"specialist": w.specialist, "status": w.status, "subquestions": w.subquestion_ids,
                     "evidence": len(w.evidence_ids)} for w in workers],
        "search": {"calls": len(ctx.search_log), "by_provider": by_provider, "fallback_calls": fallback_calls,
                   "failed_calls": failed_calls, "unavailable": list(unavailable)},
        "llm_fallbacks": [{"from": e["from"], "to": e["to"], "error": e["error"][:120]} for e in llm_events],
        "embedding_failures": [{"provider": e["provider"], "error": e["error"][:120]} for e in embed_events],
        "demo_fault": ctx.demo_fault,
    }


def degraded(rel: dict[str, Any] | None) -> bool:
    if not rel:
        return False
    s = rel.get("search") or {}
    return bool(rel.get("partial") or s.get("fallback_calls") or s.get("failed_calls"))


def coverage_lines(rel: dict[str, Any] | None) -> list[str]:
    """Markdown bullets for the report. Only our own words and provider names: no error text, no URLs."""
    if not degraded(rel):
        return []
    assert rel is not None
    out = []
    for w in rel.get("workers", []):
        if w["status"] != "success":
            what = {"partial": "covered only part of its subquestions", "failed": "saved no evidence",
                    "timeout": "timed out"}.get(w["status"], w["status"])
            out.append(f"**Partial research:** the {w['specialist']} specialist {what} "
                       f"(subquestions {', '.join(w['subquestions'])}); findings for them may be missing.")
    s = rel.get("search") or {}
    if s.get("fallback_calls"):
        used = ", ".join(f"{p} {n}" for p, n in sorted(s["by_provider"].items()))
        down = ", ".join(p for p in s.get("unavailable", []) if p not in s["by_provider"]) or "the first choice"
        out.append(f"**Search fallback:** {down} unavailable; {s['fallback_calls']} of {s['calls']} searches were "
                   f"answered by a fallback provider (by provider: {used}).")
    if s.get("failed_calls"):
        out.append(f"**Search failures:** {s['failed_calls']} search(es) failed on every provider.")
    return out
