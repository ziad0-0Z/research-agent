"""Supersede: the Evidence Analyst spots a contradiction; code decides what it means. No LLM."""
from __future__ import annotations

from datetime import date

from ..context import AppContext
from .store import MemoryItem, MemoryStore


def _date_of(item: MemoryItem) -> date:
    """Newer means: later publication date; when unknown, when we first saw it."""
    if item.source_published:
        return date.fromisoformat(item.source_published)
    return item.created_at.date()


def apply_supersede(ctx: AppContext, mstore: MemoryStore, run_to_mem: dict[str, str]) -> dict:
    """The Evidence Analyst spotted the contradiction; code decides what it means.

    Old item X is superseded by new item Y only if all hold:
      1. the Analyst listed X and Y on opposite sides of one contradiction
      2. Y was judged `supported` in this run
      3. Y is newer than X (publication date, else first seen)
      4. same site, or both are pricing and Y's source quality ≥ X's
    Otherwise both stay usable and are marked as contested with each other.
    Mechanical case: an `unverifiable` item (quote gone from its page) is superseded by a new item
    from the same URL.
    """
    superseded, contested = [], []
    analysis = ctx.analysis
    run_support = {e.id: e.support for e in ctx.store.list_evidence(ctx.run_id)}
    for c in (analysis.contradictions if analysis else []):
        for a in c.evidence_ids_a:
            for b in c.evidence_ids_b:
                ma, mb = run_to_mem.get(a), run_to_mem.get(b)
                if not ma or not mb or ma == mb:
                    continue
                xa, xb = mstore.get(ma), mstore.get(mb)
                if xa is None or xb is None:
                    continue
                old, new, new_ev = (xa, xb, b) if _date_of(xa) < _date_of(xb) else (xb, xa, a)
                rule_ok = (
                    old.first_run_id != ctx.run_id                       # X existed before this run
                    and run_support.get(new_ev) == "supported"
                    and _date_of(old) < _date_of(new)
                    and (old.source_host == new.source_host
                         or (old.claim_type == new.claim_type == "pricing"
                             and new.source_quality >= old.source_quality)))
                if rule_ok and old.status == "active":
                    mstore.update(old.mem_id, status="superseded", superseded_by=new.mem_id,
                                  status_reason=f"newer evidence in {ctx.run_id}: {c.topic}"[:200])
                    superseded.append({"old": old.mem_id, "new": new.mem_id, "topic": c.topic})
                else:
                    mstore.update(xa.mem_id, contested_with=set(xa.contested_with) | {xb.mem_id})
                    mstore.update(xb.mem_id, contested_with=set(xb.contested_with) | {xa.mem_id})
                    contested.append({"a": xa.mem_id, "b": xb.mem_id, "topic": c.topic})

    unverifiable_by_url: dict[str, list[MemoryItem]] = {}
    for old in mstore.list(status="unverifiable"):  # one query, not one per new item
        unverifiable_by_url.setdefault(old.source_url, []).append(old)
    for mid in set(run_to_mem.values()):
        item = mstore.get(mid)
        if item is None or item.first_run_id != ctx.run_id:
            continue
        for old in unverifiable_by_url.get(item.source_url, []):
            if old.mem_id != item.mem_id:
                mstore.update(old.mem_id, status="superseded", superseded_by=item.mem_id,
                              status_reason="quote gone from the page; replaced by newer evidence from the same page")
                superseded.append({"old": old.mem_id, "new": item.mem_id, "topic": "page changed"})
    return {"superseded": superseded, "contested": contested}



def merge_duplicate_excerpts(mstore: MemoryStore) -> list[dict]:
    """One-off cleanup for items stored before the same-fact rule: within each page, a later item that is
    the same fact as an earlier one (policy.same_fact) is marked superseded by it. Nothing is deleted."""
    from .policy import same_fact

    by_url: dict[str, list[MemoryItem]] = {}
    for item in mstore.list(status="active"):  # ordered by created_at: the earlier item is kept
        by_url.setdefault(item.source_url, []).append(item)
    merged = []
    for items in by_url.values():
        kept: list[MemoryItem] = []
        for item in items:
            original = next((k for k in kept if same_fact(k.source_url, k.quote, k.claim,
                                                          item.source_url, item.quote, item.claim)), None)
            if original is None:
                kept.append(item)
                continue
            mstore.update(item.mem_id, status="superseded", superseded_by=original.mem_id,
                          status_reason=f"same fact as {original.mem_id} (excerpt of the same quote)")
            merged.append({"duplicate": item.mem_id, "kept": original.mem_id})
    return merged
