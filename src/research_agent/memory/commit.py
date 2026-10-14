"""Commit a completed run to long-term memory, then apply supersede rules (supersede.py). All code, no LLM.

Called from release.py only when the run is `completed`, so it runs once for a normal release
and once when a human approves a paused run. Idempotent: committing the same run twice adds nothing.
"""
from __future__ import annotations

from datetime import datetime

from ..context import AppContext
from ..models import ResearchPlan, WorkerFindings
from .policy import (
    admit_evidence, can_commit, claim_type, expires_at, host, memory_hash, memory_support, page_hash, same_fact,
)
from .store import MemoryItem, MemoryStore
from .supersede import apply_supersede
from .user_context import extract_stated_context


def _same_fact_item(mstore: MemoryStore, url: str, quote: str, claim: str) -> MemoryItem | None:
    for item in mstore.by_url(url):
        if same_fact(item.source_url, item.quote, item.claim, url, quote, claim):
            return item
    return None


def commit_completed_run(ctx: AppContext, *, plan: ResearchPlan | None, workers: list[WorkerFindings],
                         confidence: dict, final_markdown: str) -> dict:
    mstore = MemoryStore(ctx.store)
    row = ctx.store.get_run(ctx.run_id)
    ok, reason = can_commit(ctx.settings, row["status"] if row else None, ctx.demo_fault)
    if not ok:
        return {"committed": False, "reason": reason}

    now: datetime = ctx.clock()
    specialist_of = {sq: w.specialist for w in workers for sq in w.subquestion_ids}
    sq_text = {sq.id: sq.question for sq in (plan.subquestions if plan else [])}
    added, refreshed, skipped = [], [], []
    run_to_mem: dict[str, str] = {}

    for ev in ctx.store.list_evidence(ctx.run_id):
        src = ctx.store.get_source(ctx.run_id, ev.source_id)
        admitted, why = admit_evidence(ev, src)
        h = memory_hash(str(src.url), ev.quote) if src else ""
        existing = mstore.get_by_hash(h) if src else None
        if not admitted:
            # A remembered item the Critic now judged not_supported must not be reused again.
            if existing and ev.support == "not_supported":
                mstore.update(existing.mem_id, status="retired", status_reason=f"judged not_supported in {ctx.run_id}")
            skipped.append({"evidence_id": ev.id, "reason": why})
            continue
        assert src is not None
        if mstore.is_tombstoned(h):
            skipped.append({"evidence_id": ev.id, "reason": "deleted by a human earlier"})
            continue
        existing = existing or _same_fact_item(mstore, str(src.url), ev.quote, ev.claim)
        if existing:  # the same quote, or an excerpt of the same fact: refresh, never a second copy
            support = memory_support(ev.support)
            fields: dict = {"last_run_id": ctx.run_id}
            if support != "unjudged":
                fields["support"] = support
            mstore.update(existing.mem_id, **fields)
            run_to_mem[ev.id] = existing.mem_id
            refreshed.append(existing.mem_id)
            continue
        text = ctx.store.get_source_text(ctx.run_id, ev.source_id) or ""
        ph = page_hash(str(src.url), text)
        mstore.put_page(ph, str(src.url), src.title, text, src.retrieved_at)
        ct = claim_type(specialist_of.get(ev.subquestion_id, ""), ev.claim, ev.quote)
        item = MemoryItem(
            mem_id=f"m{h[:10]}", content_hash=h, claim=ev.claim, quote=ev.quote, claim_type=ct,
            specialist=specialist_of.get(ev.subquestion_id, "unknown"),
            subquestion_text=sq_text.get(ev.subquestion_id, ""), source_url=str(src.url).rstrip("/"),
            source_host=host(str(src.url)), source_title=src.title, source_domain_type=src.domain_type,
            source_published=src.published.isoformat() if src.published else None,
            source_quality=src.quality_score,
            origin=src.origin if src.origin != "memory" else "offline", page_hash=ph,
            support=memory_support(ev.support), first_run_id=ctx.run_id, last_run_id=ctx.run_id,
            created_at=now, last_verified_at=src.retrieved_at if src.retrieved_at <= now else now,
            expires_at=expires_at(ctx.settings, ct, src.retrieved_at if src.retrieved_at <= now else now))
        mstore.insert(item)
        run_to_mem[ev.id] = item.mem_id
        added.append(item.mem_id)

    report = ctx.final_report
    review = ctx.store.get_review(ctx.run_id)
    stored_report = False
    if report is not None:
        stored_report = mstore.put_report(
            run_id=ctx.run_id, question=ctx.question, title=report.title, report_json=report.model_dump_json(),
            final_markdown=final_markdown, confidence=float(confidence.get("score", 0.0)),
            review_status="approved" if review and review.status == "approved" else "released",
            released_at=now, mem_ids=sorted(set(run_to_mem.values())))

    facts = extract_stated_context(ctx.question)
    for f in facts:
        mstore.put_context(f["key"], f["value"], f["span"], ctx.run_id, now)

    sup = apply_supersede(ctx, mstore, run_to_mem)
    return {"committed": True, "added": len(added), "refreshed": len(refreshed), "skipped": skipped,
            "report_stored": stored_report, "context_facts": facts, **sup}
