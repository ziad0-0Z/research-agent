"""Memory lookup: reuse remembered evidence in a new run, without skipping any check.

    [before the Planner]  related_reports()     similar past questions — shown in the trace/UI only,
                                                never sent to an agent (a past conclusion is not evidence)
    [after the plan]      reuse_for_plan()
        1. match       each active memory item against each subquestion: IDF-weighted term gate (generic
                       words count little), then the hybrid retriever the Critic uses ranks what passed
        2. freshness   fresh (before expires_at) → use the stored page snapshot
                       stale → re-fetch the page and re-verify the quote (code, no LLM, capped)
                               quote still there → refresh TTL and snapshot; gone → `unverifiable`
        3. inject      page → run source (origin=memory) → save_evidence (quote verified AGAIN) →
                       a normal run evidence id. The Critic and the output guardrail treat it like any other.
        4. coverage    ≥ N items from ≥ M sites → the subquestion is "memory-covered": its worker still
                       researches, at quick depth (code may lower the Orchestrator's depth, never raise it)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from ..context import AppContext
from ..guardrails.injection import detect_injection
from ..guardrails.url_safety import check_url
from ..models import ResearchPlan, Source
from ..tools.evidence import save_evidence
from ..tools.quality import score_quality
from ..tools.quotes import verify_quote
from ..tools.web import fetch_page
from .matching import assign_items, question_similarity
from .policy import expires_at, page_hash
from .store import MemoryItem, MemoryStore

DEPTH_ORDER = ("quick", "standard", "deep")


def related_reports(ctx: AppContext, mstore: MemoryStore, question: str) -> list[dict]:
    """Past released reports on a similar question. Display only."""
    out = []
    for r in mstore.reports():
        sim = question_similarity(question, r["question"])
        if sim >= ctx.settings.memory_similarity_threshold:
            out.append({"run_id": r["run_id"], "question": r["question"], "title": r["title"],
                        "similarity": sim, "confidence": r["confidence"], "released_at": r["released_at"],
                        "review_status": r["review_status"]})
    return sorted(out, key=lambda x: -x["similarity"])[:5]


@dataclass
class Reuse:
    mem_id: str
    evidence_id: str
    subquestion_id: str
    source_host: str
    freshness: str            # fresh | reverified
    last_verified_at: str
    match: float


@dataclass
class MemoryUse:
    enabled: bool
    related: list[dict] = field(default_factory=list)
    candidates: int = 0
    reused: list[Reuse] = field(default_factory=list)
    reverified: int = 0
    unverifiable: list[str] = field(default_factory=list)
    skipped_reverify_budget: int = 0
    rejected: list[dict] = field(default_factory=list)
    coverage: dict[str, dict] = field(default_factory=dict)
    retrieval_method: str = "none"
    assignment_method: str = "none"   # dense | lexical: how items were filed under subquestions

    def covered(self) -> set[str]:
        return {sq for sq, c in self.coverage.items() if c["covered"]}

    def summary(self) -> dict:
        return {"enabled": self.enabled, "related_reports": self.related, "candidates": self.candidates,
                "reused": len(self.reused), "reverified": self.reverified, "unverifiable": self.unverifiable,
                "skipped_reverify_budget": self.skipped_reverify_budget, "rejected": self.rejected,
                "coverage": self.coverage, "covered_subquestions": sorted(self.covered()),
                "retrieval": self.retrieval_method, "assignment": self.assignment_method,
                "items": [r.__dict__ for r in self.reused]}


async def fetch_text(ctx: AppContext, url: str) -> str:
    """Fresh page text for re-verification: the same URL policy and fetch path as scrape_page."""
    check = check_url(url, resolve_dns=ctx.settings.search_mode != "offline")
    if not check.allowed:
        raise ValueError(f"blocked by URL safety policy: {check.reason}")
    return (await fetch_page(ctx, url)).text


def _rank(ctx: AppContext, query: str, scored: list[tuple[MemoryItem, float]], k: int) -> tuple[list, str]:
    """Order the items that passed the term-match gate with the RAG retriever (hybrid when embeddings run)."""
    from ..rag import Doc, get_embedder, retrieve

    if not scored:
        return [], "none"
    docs = [Doc(id=it.mem_id, text=f"{it.claim} {it.quote}", quality=it.source_quality, meta={"item": it, "m": m})
            for it, m in scored]
    embedder = get_embedder(ctx.settings) if ctx.settings.embed_mode != "off" else None
    found = retrieve(query, docs, k=10, embedder=embedder)
    ranked = [(h.doc.meta["item"], h.doc.meta["m"]) for h in found.hits]
    seen = {it.mem_id for it, _ in ranked}
    ranked += [(it, m) for it, m in sorted(scored, key=lambda x: -x[1]) if it.mem_id not in seen]
    return ranked[:k], found.method


async def reuse_for_plan(ctx: AppContext, mstore: MemoryStore, plan: ResearchPlan) -> MemoryUse:
    s = ctx.settings
    use = MemoryUse(enabled=s.memory_mode != "off")
    if not use.enabled:
        return use
    items = mstore.list(status="active")
    use.candidates = len(items)
    filed = assign_items(s, items, plan)
    if filed:
        use.assignment_method = next(iter(filed.values())).method
    by_id = {it.mem_id: it for it in items}
    budget = _ReverifyBudget(s.memory_max_reverify)

    for sq in plan.subquestions:
        scored = [(by_id[mid], a.match) for mid, a in filed.items() if a.subquestion_id == sq.id]
        ranked, method = _rank(ctx, sq.question, scored, s.memory_max_reuse_per_subquestion)
        if method != "none":
            use.retrieval_method = method
        for item, match_score in ranked:
            page = await _verified_page_text(ctx, mstore, item, budget, use)
            if page is None:
                continue
            text, freshness = page
            _inject(ctx, item, sq.id, text, freshness, match_score, use)

    use.coverage = _coverage(ctx, plan, use.reused)
    ctx.log_tool("memory_lookup", True, f"{len(use.reused)} reused, {use.reverified} re-verified, "
                 f"{len(use.unverifiable)} unverifiable, covered: {sorted(use.covered()) or 'none'}")
    return use


@dataclass
class _ReverifyBudget:
    left: int


def _mark_unverifiable(mstore: MemoryStore, item: MemoryItem, reason: str, use: MemoryUse) -> None:
    mstore.update(item.mem_id, status="unverifiable", status_reason=reason[:200])
    use.unverifiable.append(item.mem_id)


async def _verified_page_text(ctx: AppContext, mstore: MemoryStore, item: MemoryItem, budget: _ReverifyBudget,
                              use: MemoryUse) -> tuple[str, str] | None:
    """The page text this item may be reused against, and whether it was fresh or re-verified.
    None when the item cannot be reused now (URL blocked, re-verify budget spent, page changed or gone)."""
    url_check = check_url(item.source_url, resolve_dns=False)  # memory never skips the URL policy
    if not url_check.allowed:
        _mark_unverifiable(mstore, item, f"URL blocked: {url_check.reason}", use)
        return None
    now = ctx.clock()
    if item.is_fresh(now):
        page = mstore.get_page(item.page_hash)
        return (page["text"] if page else ""), "fresh"
    if budget.left <= 0:
        use.skipped_reverify_budget += 1
        return None
    budget.left -= 1
    try:
        text = await fetch_text(ctx, item.source_url)
    except Exception as exc:  # dead page, blocked URL, network error
        _mark_unverifiable(mstore, item, f"re-fetch failed: {exc}", use)
        return None
    quote_still_there, _ = verify_quote(item.quote, text)
    if not quote_still_there:
        _mark_unverifiable(mstore, item, "quote no longer on the page", use)
        return None
    if detect_injection(text).detected:
        _mark_unverifiable(mstore, item, "page now contains injected instructions", use)
        return None
    new_hash = page_hash(item.source_url, text)
    mstore.put_page(new_hash, item.source_url, item.source_title, text, now)
    mstore.update(item.mem_id, page_hash=new_hash, last_verified_at=now,
                  expires_at=expires_at(ctx.settings, item.claim_type, now))
    use.reverified += 1
    return text, "reverified"


def _inject(ctx: AppContext, item: MemoryItem, subquestion_id: str, text: str, freshness: str,
            match_score: float, use: MemoryUse) -> None:
    """Page → run source (origin=memory) → save_evidence, which verifies the quote again."""
    source = ctx.store.find_source_by_url(ctx.run_id, item.source_url)
    if source is None:
        published = date.fromisoformat(item.source_published) if item.source_published else None
        quality = score_quality(domain_type=item.source_domain_type, published=published,
                                text_length=len(text), injection_detected=False)["score"]
        source = Source(id=ctx.next_id("src"), url=item.source_url, title=item.source_title,
                        domain_type=item.source_domain_type, published=published, origin="memory",
                        quality_score=quality, injection_detected=False)
        ctx.store.add_source(ctx.run_id, source, text=text)
    out = save_evidence(ctx, source_id=source.id, subquestion_id=subquestion_id, claim=item.claim, quote=item.quote)
    if not out.get("saved"):
        use.rejected.append({"mem_id": item.mem_id, "reason": out.get("reason", "")})
        return
    verified_at = ctx.clock().isoformat() if freshness == "reverified" else item.last_verified_at.isoformat()
    use.reused.append(Reuse(item.mem_id, out["evidence_id"], subquestion_id, item.source_host, freshness,
                            verified_at, match_score))
    ctx.memory_reuse[out["evidence_id"]] = {"mem_id": item.mem_id, "last_verified_at": verified_at,
                                            "freshness": freshness}


def _coverage(ctx: AppContext, plan: ResearchPlan, reused: list[Reuse]) -> dict[str, dict]:
    """Memory-covered = enough reused items from enough different sites (decided in code)."""
    s = ctx.settings
    out = {}
    for sq in plan.subquestions:
        mine = [r for r in reused if r.subquestion_id == sq.id]
        sites = {r.source_host for r in mine}
        out[sq.id] = {"items": len(mine), "sites": len(sites),
                      "covered": len(mine) >= s.memory_coverage_min_items and len(sites) >= s.memory_coverage_min_sources}
    return out


def lower_depth(depth: str, cap: str) -> str:
    """Code may only make research shallower than the Orchestrator asked, never deeper."""
    return DEPTH_ORDER[min(DEPTH_ORDER.index(depth), DEPTH_ORDER.index(cap))]
