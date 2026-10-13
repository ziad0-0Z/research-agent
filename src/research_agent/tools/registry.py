"""The 11 domain tools as OpenAI Agents SDK function tools.

Each wrapper is thin: it pulls AppContext from RunContextWrapper, calls the
core function, and turns ToolError into a structured error the agent can read
and recover from (instead of crashing the run).
"""
from __future__ import annotations

from typing import Any

from agents import RunContextWrapper, function_tool

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
    """Fetch a web page, register it as a source and return its text as untrusted content.

    Blocked for local files, localhost and private network addresses.

    Args:
        url: The full http(s) URL of the page.
    """
    try:
        return await web.scrape_page(ctx.context, url)
    except ToolError as exc:
        return _error(exc)


@function_tool
async def extract_claims(ctx: RunContextWrapper[AppContext], source_id: str, subquestion_id: str,
                         subquestion: str) -> dict:
    """Extract candidate claims with verbatim quotes from a scraped source for one subquestion.

    Args:
        source_id: The id returned by scrape_page, e.g. "src3".
        subquestion_id: The subquestion id, e.g. "sq2".
        subquestion: The subquestion text.
    """
    try:
        return await analysis.extract_claims(ctx.context, source_id=source_id,
                                             subquestion_id=subquestion_id, subquestion=subquestion)
    except ToolError as exc:
        return _error(exc)


@function_tool
def save_evidence(ctx: RunContextWrapper[AppContext], source_id: str, subquestion_id: str,
                  claim: str, quote: str) -> dict:
    """Save one piece of evidence. The quote must be copied exactly from the source; it is verified.

    Args:
        source_id: The source the quote comes from, e.g. "src3".
        subquestion_id: The subquestion this evidence answers, e.g. "sq2".
        claim: The claim in your own words (10-400 characters).
        quote: The exact supporting text from the source (10-600 characters).
    """
    return evidence.save_evidence(ctx.context, source_id=source_id, subquestion_id=subquestion_id,
                                  claim=claim, quote=quote)


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


RESEARCH_TOOLS = [search_web, validate_source, scrape_page, extract_claims, save_evidence, score_source_quality]
ALL_TOOLS = RESEARCH_TOOLS + [compare_sources, retrieve_evidence, validate_citation,
                              request_human_approval, generate_report]
