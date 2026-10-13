"""search_web, validate_source and scrape_page core logic.

Search chain: Tavily → DuckDuckGo → offline corpus (each result is labelled
with where it came from). Scraping: offline corpus pages are served from disk;
live pages go through URL safety, redirect re-checks, size limits and
trafilatura extraction. Fetched text is always treated as untrusted.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import httpx

from ..context import AppContext
from ..guardrails.injection import detect_injection, sanitize, wrap_untrusted
from ..guardrails.url_safety import check_url
from ..models import Source
from .corpus import load_corpus
from .quality import classify_domain, score_quality

MAX_RESULTS = 8
MAX_DOWNLOAD_BYTES = 2_000_000
MAX_TEXT_TO_MODEL = 6_000
FETCH_TIMEOUT = 15.0
MAX_REDIRECTS = 3


class ToolError(Exception):
    """A tool failure with a message safe to show to the model and the user."""


@dataclass
class SearchResult:
    url: str
    title: str
    snippet: str
    origin: str


# --------------------------------------------------------------------------- search

async def _search_tavily(ctx: AppContext, query: str, n: int) -> list[SearchResult]:
    if not ctx.settings.tavily_api_key:
        raise ToolError("TAVILY_API_KEY is not set")
    from tavily import TavilyClient

    client = TavilyClient(api_key=ctx.settings.tavily_api_key)
    data = await asyncio.wait_for(
        asyncio.to_thread(client.search, query, max_results=n, search_depth="basic"), timeout=20
    )
    return [
        SearchResult(r["url"], r.get("title", ""), (r.get("content") or "")[:300], "tavily")
        for r in data.get("results", [])
    ]


async def _search_ddg(query: str, n: int) -> list[SearchResult]:
    from ddgs import DDGS

    def run() -> list[dict[str, Any]]:
        return list(DDGS().text(query, max_results=n))

    rows = await asyncio.wait_for(asyncio.to_thread(run), timeout=20)
    return [SearchResult(r.get("href", ""), r.get("title", ""), (r.get("body") or "")[:300], "ddg") for r in rows]


def _search_offline(ctx: AppContext, query: str, n: int) -> list[SearchResult]:
    corpus = load_corpus(str(ctx.settings.corpus_dir))
    return [
        SearchResult(p.url, p.title, p.text[:300].replace("\n", " "), "offline")
        for p, _score in corpus.search(query, k=n)
    ]


async def search_web(ctx: AppContext, query: str, max_results: int = 5) -> dict:
    query = (query or "").strip()
    if len(query) < 3:
        raise ToolError("query is too short")
    if len(query) > 300:
        raise ToolError("query is too long (max 300 characters)")
    n = max(1, min(int(max_results), MAX_RESULTS))

    chain = {"tavily": ["tavily", "ddg", "offline"], "ddg": ["ddg", "offline"], "offline": ["offline"]}
    errors: list[str] = []
    for provider in chain[ctx.settings.search_mode]:
        try:
            if provider == "tavily":
                results = await _search_tavily(ctx, query, n)
            elif provider == "ddg":
                results = await _search_ddg(query, n)
            else:
                results = _search_offline(ctx, query, n)
        except Exception as exc:  # noqa: BLE001 - every provider failure falls through
            errors.append(f"{provider}: {type(exc).__name__}: {str(exc)[:120]}")
            continue
        seen: set[str] = set()
        unique = []
        for r in results:
            if r.url and r.url not in seen and check_url(r.url).allowed:
                seen.add(r.url)
                unique.append(r)
        if unique or provider == "offline":
            ctx.log_tool("search_web", True, f"{provider}: {len(unique)} results for {query!r}")
            return {
                "query": query,
                "provider": provider,
                "fallback_used": provider != chain[ctx.settings.search_mode][0],
                "errors": errors,
                "results": [asdict(r) for r in unique],
            }
    ctx.log_tool("search_web", False, f"all providers failed for {query!r}")
    raise ToolError("all search providers failed: " + "; ".join(errors))


# --------------------------------------------------------------------------- validate_source

async def validate_source(ctx: AppContext, url: str) -> dict:
    check = check_url(url, resolve_dns=ctx.settings.search_mode != "offline")
    result = {"url": url, "allowed": check.allowed, "reason": check.reason}
    if not check.allowed:
        ctx.log_tool("validate_source", False, f"blocked {url}: {check.reason}")
        return result

    corpus_page = load_corpus(str(ctx.settings.corpus_dir)).get(url)
    if corpus_page:
        result.update(reachable=True, origin="offline", domain_type=corpus_page.domain_type,
                      published=corpus_page.published.isoformat() if corpus_page.published else None)
    elif ctx.settings.search_mode == "offline":
        result.update(reachable=False, reason="not in the offline corpus", domain_type=classify_domain(url))
    else:
        try:
            async with httpx.AsyncClient(timeout=8.0, follow_redirects=False) as client:
                resp = await client.head(url)
            ctype = resp.headers.get("content-type", "")
            result.update(reachable=resp.status_code < 400, status=resp.status_code,
                          content_type=ctype, domain_type=classify_domain(url))
        except httpx.HTTPError as exc:
            result.update(reachable=False, reason=f"{type(exc).__name__}", domain_type=classify_domain(url))
    ctx.log_tool("validate_source", True, f"{url}: reachable={result.get('reachable')}")
    return result


# --------------------------------------------------------------------------- scrape_page

async def _fetch_live(url: str, resolve_dns: bool) -> tuple[str, str, date | None]:
    import trafilatura

    current = url
    async with httpx.AsyncClient(timeout=FETCH_TIMEOUT, follow_redirects=False,
                                 headers={"User-Agent": "research-agent/1.0 (student project)"}) as client:
        for _ in range(MAX_REDIRECTS + 1):
            async with client.stream("GET", current) as resp:
                if resp.is_redirect:
                    nxt = str(resp.next_request.url) if resp.next_request else ""
                    check = check_url(nxt, resolve_dns=resolve_dns)
                    if not check.allowed:
                        raise ToolError(f"redirect to blocked URL: {check.reason}")
                    current = nxt
                    continue
                if resp.status_code >= 400:
                    raise ToolError(f"HTTP {resp.status_code}")
                ctype = resp.headers.get("content-type", "")
                if not any(t in ctype for t in ("html", "text", "xml")):
                    raise ToolError(f"unsupported content type '{ctype}'")
                body = bytearray()
                async for chunk in resp.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > MAX_DOWNLOAD_BYTES:
                        raise ToolError("page too large")
                html = body.decode(resp.encoding or "utf-8", errors="replace")
                break
        else:
            raise ToolError("too many redirects")

    text = trafilatura.extract(html, include_comments=False, include_tables=True) or ""
    meta = trafilatura.extract_metadata(html)
    title = (meta.title if meta and meta.title else "") or current
    published = None
    if meta and meta.date:
        try:
            published = date.fromisoformat(meta.date[:10])
        except ValueError:
            published = None
    return text, title, published


async def scrape_page(ctx: AppContext, url: str) -> dict:
    """Fetch a page, register it as a Source, return sanitized text for the model."""
    url = (url or "").strip()
    resolve = ctx.settings.search_mode != "offline"
    check = check_url(url, resolve_dns=resolve)
    if not check.allowed:
        ctx.log_tool("scrape_page", False, f"BLOCKED {url}: {check.reason}")
        raise ToolError(f"blocked by URL safety policy: {check.reason}")

    existing = ctx.store.find_source_by_url(ctx.run_id, url)
    if existing:
        text = ctx.store.get_source_text(ctx.run_id, existing.id) or ""
        return _scrape_payload(existing, text, cached=True)

    page = load_corpus(str(ctx.settings.corpus_dir)).get(url)
    if page:
        text, title, published, domain_type, origin = page.text, page.title, page.published, page.domain_type, "offline"
    elif ctx.settings.search_mode == "offline":
        ctx.log_tool("scrape_page", False, f"{url} not in offline corpus")
        raise ToolError("page not available in offline mode")
    else:
        try:
            text, title, published = await _fetch_live(url, resolve)
        except httpx.HTTPError as exc:
            ctx.log_tool("scrape_page", False, f"{url}: {type(exc).__name__}")
            raise ToolError(f"could not fetch page ({type(exc).__name__})") from exc
        domain_type, origin = classify_domain(url), ("tavily" if ctx.settings.search_mode == "tavily" else "ddg")

    if len(text.strip()) < 200:
        ctx.log_tool("scrape_page", False, f"{url}: no usable text")
        raise ToolError("page has no usable text content")

    injection = detect_injection(text)
    quality = score_quality(domain_type=domain_type, published=published, text_length=len(text),
                            injection_detected=injection.detected)
    source = Source(id=ctx.next_id("src"), url=url, title=title[:300], domain_type=domain_type,
                    published=published, origin=origin, quality_score=quality["score"],
                    injection_detected=injection.detected)
    ctx.store.add_source(ctx.run_id, source, text=text)
    ctx.log_tool("scrape_page", True,
                 f"{source.id} {url} quality={quality['score']} injection={injection.detected}")
    return _scrape_payload(source, text, cached=False, injection_patterns=injection.patterns)


def _scrape_payload(source: Source, text: str, *, cached: bool, injection_patterns: list[str] | None = None) -> dict:
    clean, removed = sanitize(text)
    truncated = len(clean) > MAX_TEXT_TO_MODEL
    return {
        "source_id": source.id,
        "url": str(source.url),
        "title": source.title,
        "domain_type": source.domain_type,
        "published": source.published.isoformat() if source.published else None,
        "quality_score": source.quality_score,
        "injection_detected": source.injection_detected,
        "injection_patterns": injection_patterns or [],
        "sentences_removed": removed,
        "truncated": truncated,
        "cached": cached,
        "content": wrap_untrusted(clean[:MAX_TEXT_TO_MODEL], source.id),
    }
