"""LLM-backed tool steps: extract_claims, compare_sources, validate_citation.

The LLM proposes; Python checks. Every LLM output here is followed by a
deterministic check (quote verification, id validation, citation checks).
`run_agent` is injectable so tests can replace the model with a fake.
"""
from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from agents import Agent, Runner

from ..context import AppContext
from ..guardrails.injection import sanitize, wrap_untrusted
from ..models import ClaimExtraction, ClaimSupportVerdict, Contradiction, EvidenceAnalysis
from .evidence import check_citation
from .quotes import verify_quote
from .web import ToolError

MAX_EXTRACT_CHARS = 9_000
AgentRunner = Callable[[Agent, str], Awaitable[Any]]


async def default_run_agent(agent: Agent, prompt: str) -> Any:
    result = await Runner.run(agent, prompt, max_turns=2)
    return result.final_output


async def extract_claims(ctx: AppContext, *, source_id: str, subquestion_id: str, subquestion: str,
                         run_agent: AgentRunner | None = None, auto_save: bool = False) -> dict:
    """Ask the extractor for claims, then verify every quote against the source text in code.

    auto_save=False: returns candidates only; nothing is saved until save_evidence accepts them.
    auto_save=True (the agent tool): every verified candidate goes straight through the protected
    save_evidence check. This saves one LLM turn per subquestion and means evidence survives
    even if the worker later times out.
    """
    from ..agents.tool_agents import claim_extractor

    source = ctx.store.get_source(ctx.run_id, source_id)
    if source is None:
        raise ToolError(f"unknown source_id '{source_id}': scrape the page first")
    text = ctx.store.get_source_text(ctx.run_id, source_id) or ""
    clean, _ = sanitize(text)
    prompt = (
        f"Subquestion id: {subquestion_id}\nSubquestion: {subquestion}\n\n"
        f"{wrap_untrusted(clean[:MAX_EXTRACT_CHARS], source_id)}"
    )
    output: ClaimExtraction = await (run_agent or default_run_agent)(claim_extractor(), prompt)

    candidates = []
    for c in output.claims:
        verified, similarity = verify_quote(c.quote, text)
        candidates.append({
            "claim": c.claim, "quote": c.quote, "subquestion_id": subquestion_id,
            "quote_verified": verified, "match": similarity,
        })
    n_ok = sum(c["quote_verified"] for c in candidates)
    ctx.log_tool("extract_claims", True, f"{source_id}: {len(candidates)} claims, {n_ok} quotes verified")
    if auto_save:
        return _save_verified(ctx, source_id, candidates)
    return {
        "source_id": source_id,
        "candidates": candidates,
        "note": "Call save_evidence for each candidate you want to keep. Unverified quotes will be rejected.",
    }


def _save_verified(ctx: AppContext, source_id: str, candidates: list[dict]) -> dict:
    """Send verified candidates through the protected save_evidence check (which re-verifies)."""
    from .evidence import save_evidence

    results = []
    for c in candidates:
        if not c["quote_verified"]:
            results.append({"claim": c["claim"], "saved": False,
                            "reason": f"quote not found in source (best match {c['match']:.0%})"})
            continue
        out = save_evidence(ctx, source_id=source_id, subquestion_id=c["subquestion_id"],
                            claim=c["claim"], quote=c["quote"])
        results.append({"claim": c["claim"], "saved": out.get("saved", False),
                        "evidence_id": out.get("evidence_id"), "reason": out.get("reason")})
    # Short answer for the researcher: no quotes, so its context stays small.
    return {
        "source_id": source_id,
        "saved_evidence": [{"evidence_id": r["evidence_id"], "claim": r["claim"]} for r in results if r["saved"]],
        "rejected": [{"claim": r["claim"][:100], "reason": r["reason"]} for r in results if not r["saved"]],
    }


async def compare_sources(ctx: AppContext, *, subquestion_ids: list[str] | None = None,
                          run_agent: AgentRunner | None = None) -> dict:
    """Run the Evidence Analyst over saved evidence; drop any reference to unknown ids."""
    from ..agents.tool_agents import evidence_analyst

    evidence = [e for e in ctx.store.list_evidence(ctx.run_id) if e.status == "active"]
    if subquestion_ids:
        evidence = [e for e in evidence if e.subquestion_id in subquestion_ids]
    if not evidence:
        raise ToolError("no saved evidence to compare yet")

    items = []
    for e in evidence:
        src = ctx.store.get_source(ctx.run_id, e.source_id)
        items.append({
            "id": e.id, "subquestion_id": e.subquestion_id, "claim": e.claim, "quote": e.quote,
            "source_id": e.source_id, "source_type": src.domain_type if src else "unknown",
            "source_quality": src.quality_score if src else 0.0,
            "published": src.published.isoformat() if src and src.published else None,
        })
    sq_ids = sorted({e.subquestion_id for e in evidence} | set(subquestion_ids or []))
    prompt = (
        f"Subquestion ids: {', '.join(sq_ids)}\n\nEvidence items (JSON):\n"
        + json.dumps(items, ensure_ascii=False, indent=1)
    )
    analysis: EvidenceAnalysis = await (run_agent or default_run_agent)(evidence_analyst(), prompt)

    known = {e.id for e in evidence}
    dropped = 0
    valid_contradictions: list[Contradiction] = []
    for c in analysis.contradictions:
        a = [i for i in c.evidence_ids_a if i in known]
        b = [i for i in c.evidence_ids_b if i in known]
        if a and b:
            valid_contradictions.append(c.model_copy(update={"evidence_ids_a": a, "evidence_ids_b": b}))
        else:
            dropped += 1
    analysis = analysis.model_copy(update={
        "contradictions": valid_contradictions,
        "gaps": [g for g in analysis.gaps if g in sq_ids],
    })
    ctx.analysis = analysis
    ctx.log_tool("compare_sources", True,
                 f"{len(evidence)} items: {len(analysis.agreements)} agreements, "
                 f"{len(analysis.contradictions)} contradictions, {dropped} dropped (unknown ids)")
    return {"analysis": analysis.model_dump(), "evidence_count": len(evidence), "dropped_invalid": dropped}


async def validate_citation(ctx: AppContext, *, evidence_id: str, claim: str,
                            run_agent: AgentRunner | None = None) -> dict:
    """Check 1 (code): the evidence exists and its quote is in the source.
    Check 2 (LLM judge): the quote actually supports this claim."""
    from ..agents.tool_agents import support_judge

    det = check_citation(ctx, evidence_id)
    if not det["valid"]:
        ctx.log_tool("validate_citation", False, f"{evidence_id}: {det['reason']}")
        return {"evidence_id": evidence_id, "valid": False, "support": "not_supported", "reason": det["reason"]}

    ev = det["evidence"]
    prompt = f"Claim: {claim}\n\nQuote: {ev.quote}"
    verdict: ClaimSupportVerdict = await (run_agent or default_run_agent)(support_judge(), prompt)
    ctx.store.set_support(ctx.run_id, evidence_id, verdict.support)
    ok = verdict.support in ("supported", "partial")
    ctx.log_tool("validate_citation", ok, f"{evidence_id}: {verdict.support}")
    return {
        "evidence_id": evidence_id, "valid": ok, "support": verdict.support,
        "reason": verdict.reason, "overreach": verdict.overreach,
        "source_id": ev.source_id, "source_quality": det["source"].quality_score,
    }
