"""save_evidence, score_source_quality, retrieve_evidence and the deterministic
half of validate_citation. All pure Python: the rules here cannot be talked
around by a model."""
from __future__ import annotations

from ..context import AppContext
from ..models import Evidence
from ..store import DuplicateEvidenceError
from .quality import score_quality
from .quotes import verify_quote
from .web import ToolError

MAX_CLAIM_CHARS = 400
MAX_QUOTE_CHARS = 600


def save_evidence(ctx: AppContext, *, source_id: str, subquestion_id: str, claim: str, quote: str) -> dict:
    """Protected action: only grounded evidence enters the store.

    Rejects unknown sources, quotes that are not in the source text, quotes
    from injection-flagged sources and duplicates. Never trusts flags set by
    earlier steps: the quote is re-verified here against the stored text.
    """
    claim, quote = (claim or "").strip(), (quote or "").strip()
    if len(claim) < 10 or len(claim) > MAX_CLAIM_CHARS:
        return _reject(ctx, "claim must be 10-400 characters")
    if len(quote) < 10 or len(quote) > MAX_QUOTE_CHARS:
        return _reject(ctx, "quote must be 10-600 characters")

    source = ctx.store.get_source(ctx.run_id, source_id)
    if source is None:
        return _reject(ctx, f"unknown source_id '{source_id}': scrape the page first")
    text = ctx.store.get_source_text(ctx.run_id, source_id) or ""

    verified, similarity = verify_quote(quote, text)
    if not verified:
        return _reject(ctx, f"quote not found in {source_id} (best match {similarity:.0%}); copy the exact words")

    evidence = Evidence.build(
        evidence_id=ctx.next_id("ev"), run_id=ctx.run_id, subquestion_id=subquestion_id,
        claim=claim, quote=quote, source_id=source_id, verified=True,
    )
    try:
        ctx.store.add_evidence(evidence)
    except DuplicateEvidenceError:
        ctx._counters["ev"] -= 1  # keep ids dense; nothing was saved
        return _reject(ctx, "duplicate: this quote from this source is already saved")

    warning = "source contains suspected prompt injection; treat as low-trust" if source.injection_detected else ""
    ctx.log_tool("save_evidence", True, f"{evidence.id} from {source_id} ({subquestion_id})")
    return {"saved": True, "evidence_id": evidence.id, "source_id": source_id,
            "quality_score": source.quality_score, "warning": warning}


def _reject(ctx: AppContext, reason: str) -> dict:
    ctx.log_tool("save_evidence", False, reason)
    return {"saved": False, "reason": reason}


def score_source_quality(ctx: AppContext, source_id: str) -> dict:
    source = ctx.store.get_source(ctx.run_id, source_id)
    if source is None:
        raise ToolError(f"unknown source_id '{source_id}'")
    text = ctx.store.get_source_text(ctx.run_id, source_id) or ""
    breakdown = score_quality(domain_type=source.domain_type, published=source.published,
                              text_length=len(text), injection_detected=source.injection_detected)
    ctx.store.update_source_quality(ctx.run_id, source_id, breakdown["score"])
    ctx.log_tool("score_source_quality", True, f"{source_id}={breakdown['score']}")
    return {"source_id": source_id, **breakdown}


def retrieve_evidence(ctx: AppContext, query: str, k: int = 5, exclude_ids: set[str] | None = None) -> dict:
    """Hybrid retrieval (BM25 + embeddings, rank-fused, quality-weighted) over this run's evidence.

    The RAG bonus: the Critic calls this per finding to see which saved evidence is most relevant,
    so it can spot findings that cite the wrong item or ignore a better or contradicting one.
    Falls back to BM25-only when no embedding provider is reachable; `method` says which ran.
    """
    from ..rag import Doc, get_embedder, retrieve

    items = [e for e in ctx.store.list_evidence(ctx.run_id)
             if e.status == "active" and e.id not in (exclude_ids or set())]
    if not items:
        return {"query": query, "results": [], "method": "none", "embed_provider": None}
    docs = []
    for e in items:
        source = ctx.store.get_source(ctx.run_id, e.source_id)
        docs.append(Doc(id=e.id, text=f"{e.claim} {e.quote}", quality=source.quality_score if source else 0.5,
                        meta={"evidence": e}))
    embedder = get_embedder(ctx.settings) if ctx.settings.embed_mode != "off" else None
    found = retrieve(query, docs, k=k, embedder=embedder)
    results = []
    for h in found.hits:
        ev = h.doc.meta["evidence"]
        results.append({"evidence_id": ev.id, "score": h.score, "claim": ev.claim, "quote": ev.quote,
                        "source_id": ev.source_id, "source_quality": h.doc.quality, "support": ev.support,
                        "subquestion_id": ev.subquestion_id, "bm25_rank": h.bm25_rank, "dense_rank": h.dense_rank})
    ctx.log_tool("retrieve_evidence", True, f"{len(results)} hits ({found.method}) for {query[:80]!r}")
    return {"query": query, "results": results, "method": found.method, "embed_provider": found.embed_provider}


def check_citation(ctx: AppContext, evidence_id: str) -> dict:
    """Deterministic citation checks (used by validate_citation and the output guardrail)."""
    ev = ctx.store.get_evidence(ctx.run_id, evidence_id)
    if ev is None:
        return {"evidence_id": evidence_id, "valid": False, "reason": "evidence id does not exist in this run"}
    if not ev.verified:
        return {"evidence_id": evidence_id, "valid": False, "reason": "quote was never verified"}
    source = ctx.store.get_source(ctx.run_id, ev.source_id)
    if source is None:
        return {"evidence_id": evidence_id, "valid": False, "reason": "source missing"}
    text = ctx.store.get_source_text(ctx.run_id, ev.source_id) or ""
    still_there, _ = verify_quote(ev.quote, text)
    if not still_there:
        return {"evidence_id": evidence_id, "valid": False, "reason": "quote no longer matches source text"}
    return {"evidence_id": evidence_id, "valid": True, "reason": "ok", "evidence": ev, "source": source}
