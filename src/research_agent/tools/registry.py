"""The 11 domain tools as OpenAI Agents SDK function tools.

Each wrapper is thin: it pulls AppContext from RunContextWrapper, calls the
core function, and turns ToolError into a structured error the agent can read
and recover from (instead of crashing the run).
"""
from __future__ import annotations

from typing import Any

from agents import RunContextWrapper, function_tool
from pydantic import BaseModel, Field

from ..context import AppContext
from . import analysis, evidence, review, web
from .web import ToolError


def _error(exc: Exception) -> dict[str, Any]:
    return {"error": str(exc), "recoverable": True}


@function_tool
async def search_web(ctx: RunContextWrapper[AppContext], query: str, max_results: int = 5) -> dict:
    """Search the web for sources about a research question.

    Args:
        query: A short, specific search query (3-300 characters).
        max_results: Number of results to return (1-8).
    """
    try:
        return await web.search_web(ctx.context, query, max_results)
    except ToolError as exc:
        return _error(exc)


@function_tool
async def validate_source(ctx: RunContextWrapper[AppContext], url: str) -> dict:
    """Check whether a URL is safe and reachable, and classify its source type, before scraping it.

    Args:
        url: The full http(s) URL to check.
    """
    return await web.validate_source(ctx.context, url)


@function_tool
async def scrape_page(ctx: RunContextWrapper[AppContext], url: str) -> dict:
    """Fetch a web page and register it as a source. Returns its source_id and a short preview.

    The full text is stored; extract_claims reads it from there. Blocked for local files,
    localhost and private network addresses.

    Args:
        url: The full http(s) URL of the page.
    """
    try:
        payload = await web.scrape_page(ctx.context, url)
    except ToolError as exc:
        return _error(exc)
    return _with_preview(payload)


PREVIEW_CHARS = 500


def _with_preview(payload: dict) -> dict:
    """Keep the researcher's context small: it sees a preview, not the whole page.

    Every later turn re-sends the conversation, so 3 full pages (~18k chars) would make
    every following model call slower. extract_claims reads the full stored text.
    """
    out = dict(payload)
    content = out.pop("content", "")
    out["preview"] = content[:PREVIEW_CHARS] + ("…" if len(content) > PREVIEW_CHARS else "")
    out["note"] = "Full text stored. Call extract_claims with this source_id to get verified evidence."
    return out


@function_tool
async def extract_claims(ctx: RunContextWrapper[AppContext], source_id: str, subquestion_id: str,
                         subquestion: str) -> dict:
    """Extract claims with verbatim quotes from a scraped source for one subquestion.

    Every quote is verified against the stored page in code; verified claims are saved
    as evidence automatically and their evidence ids are returned.

    Args:
        source_id: The id returned by scrape_page, e.g. "src3".
        subquestion_id: The subquestion id, e.g. "sq2".
        subquestion: The subquestion text.
    """
    try:
        return await analysis.extract_claims(ctx.context, source_id=source_id,
                                             subquestion_id=subquestion_id, subquestion=subquestion,
                                             auto_save=True)
    except ToolError as exc:
        return _error(exc)


class EvidenceInput(BaseModel):
    source_id: str = Field(description='The source the quote comes from, e.g. "src3"')
    subquestion_id: str = Field(description='The subquestion this evidence answers, e.g. "sq2"')
    claim: str = Field(description="The claim in your own words (10-400 characters)")
    quote: str = Field(description="The exact supporting text copied from the source (10-600 characters)")


MAX_BATCH = 12


@function_tool
def save_evidence(ctx: RunContextWrapper[AppContext], items: list[EvidenceInput]) -> dict:
    """Save evidence items you copied yourself (up to 12 per call). Not needed after extract_claims,
    which saves verified claims automatically. Each quote must be copied exactly; it is verified.

    Args:
        items: The evidence items to save.
    """
    if not items:
        return {"error": "no items given", "recoverable": True}
    results = []
    for item in items[:MAX_BATCH]:
        out = evidence.save_evidence(ctx.context, source_id=item.source_id, subquestion_id=item.subquestion_id,
                                     claim=item.claim, quote=item.quote)
        results.append(out)
    saved = [r["evidence_id"] for r in results if r.get("saved")]
    return {
        "saved_ids": saved,
        "rejected": [{"claim": i.claim[:80], "reason": r["reason"]}
                     for i, r in zip(items, results) if not r.get("saved")],
        "skipped_over_limit": max(0, len(items) - MAX_BATCH),
    }


@function_tool
def score_source_quality(ctx: RunContextWrapper[AppContext], source_id: str) -> dict:
    """Score a source's reliability (0-1) from its type, recency, length and injection flags.

    Args:
        source_id: The source id, e.g. "src3".
    """
    try:
        return evidence.score_source_quality(ctx.context, source_id)
    except ToolError as exc:
        return _error(exc)


@function_tool
async def compare_sources(ctx: RunContextWrapper[AppContext], subquestion_ids: list[str] | None = None) -> dict:
    """Compare saved evidence across sources: agreements, contradictions and gaps.

    Args:
        subquestion_ids: Limit to these subquestions; omit to compare all evidence.
    """
    try:
        return await analysis.compare_sources(ctx.context, subquestion_ids=subquestion_ids)
    except ToolError as exc:
        return _error(exc)


@function_tool
def retrieve_evidence(ctx: RunContextWrapper[AppContext], query: str, k: int = 5) -> dict:
    """Find the saved evidence most relevant to a statement or question.

    Args:
        query: The statement or question to find evidence for.
        k: Number of results (1-10).
    """
    return evidence.retrieve_evidence(ctx.context, query, k)


@function_tool
async def validate_citation(ctx: RunContextWrapper[AppContext], evidence_id: str, claim: str) -> dict:
    """Check that an evidence id exists, its quote is real, and the quote supports the given claim.

    Args:
        evidence_id: The evidence id cited, e.g. "ev4".
        claim: The statement that cites this evidence.
    """
    return await analysis.validate_citation(ctx.context, evidence_id=evidence_id, claim=claim)


@function_tool
def request_human_approval(ctx: RunContextWrapper[AppContext], reasons: list[str]) -> dict:
    """Pause the run and ask a human to review before the recommendation is finalised.

    Args:
        reasons: Why human review is needed.
    """
    try:
        return review.request_human_approval(ctx.context, reasons)
    except ToolError as exc:
        return _error(exc)


@function_tool
def generate_report(ctx: RunContextWrapper[AppContext], approval_token: str) -> dict:
    """Render the final validated report. Requires the approval token issued after all checks pass.

    Args:
        approval_token: The token issued by the system after validation.
    """
    try:
        return review.generate_report(ctx.context, approval_token)
    except ToolError as exc:
        return _error(exc)


# Researchers do not get save_evidence: extract_claims already saves verified claims through the
# same protected check, and giving them both made them re-save duplicates and run out of turns.
RESEARCH_TOOLS = [search_web, validate_source, scrape_page, extract_claims, score_source_quality]
ALL_TOOLS = RESEARCH_TOOLS + [save_evidence, compare_sources, retrieve_evidence, validate_citation,
                              request_human_approval, generate_report]
