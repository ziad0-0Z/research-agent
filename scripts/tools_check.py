"""Tools check: run every tool for real (live LLM, offline corpus by default).

    python scripts/tools_check.py               # offline corpus (deterministic)
    python scripts/tools_check.py --live-search # Tavily -> DuckDuckGo -> offline
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents import trace  # noqa: E402
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError  # noqa: E402

from research_agent.config import ConfigError, load_settings  # noqa: E402
from research_agent.context import AppContext  # noqa: E402
from research_agent.models import Finding, ResearchReport  # noqa: E402
from research_agent.store import Store  # noqa: E402
from research_agent.tools import analysis, evidence, review, web  # noqa: E402
from research_agent.tools.web import ToolError  # noqa: E402
from research_agent.tracing import configure_tracing  # noqa: E402

SUBQUESTION = ("What are the total costs, including engineering time, of self-hosting an open-source "
               "LLM versus using cloud LLM APIs for a 5-person startup?")
P10 = "https://scaleup-digest.example/hidden-costs-llm-apis"

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"   {'[ok]' if ok else '[fail]'} {name}" + (f" — {detail}" if detail else ""), flush=True)


def step(n: int, title: str) -> None:
    print(f"\n[{n}/9] {title}", flush=True)


async def main(live_search: bool) -> int:
    try:
        settings = load_settings()
        settings.require_llm_key()
    except ConfigError as exc:
        print(f"[config] {exc}")
        return 1
    if not live_search:
        settings = dataclasses.replace(settings, search_mode="offline")
    configure_tracing(settings)
    ctx = AppContext(settings=settings, store=Store(":memory:"), question=SUBQUESTION)
    print(f"Provider: {settings.llm_provider} ({settings.model_name}) | search: {settings.search_mode} | run: {ctx.run_id}")

    with trace("Tools check", group_id=ctx.run_id):
        step(1, "search_web")
        found = await web.search_web(ctx, "total cost of ownership self-hosting LLM small team engineering time")
        urls = [r["url"] for r in found["results"]]
        check("search returns results", bool(urls), f"{found['provider']}: {len(urls)} results")
        for r in found["results"][:4]:
            print(f"      - {r['title'][:70]}")

        step(2, "validate_source + scrape_page (top 4 results)")
        scraped = []
        for url in urls[:4]:
            v = await web.validate_source(ctx, url)
            if not v["allowed"]:
                continue
            try:
                page = await web.scrape_page(ctx, url)
            except ToolError as exc:
                print(f"      skip {url}: {exc}")
                continue
            scraped.append(page)
            print(f"      {page['source_id']}: {page['title'][:55]} | {page['domain_type']} | quality {page['quality_score']}")
        check("pages scraped and registered", len(scraped) >= 2, f"{len(scraped)} sources")

        step(3, "extract_claims (live LLM) — quotes verified in code")
        candidates = []
        for page in scraped:
            out = await analysis.extract_claims(ctx, source_id=page["source_id"], subquestion_id="sq1",
                                                subquestion=SUBQUESTION)
            for c in out["candidates"]:
                c["source_id"] = page["source_id"]
                candidates.append(c)
                mark = "ok" if c["quote_verified"] else "x"
                print(f"      {mark} [{page['source_id']}] {c['claim'][:85]}")
        verified = [c for c in candidates if c["quote_verified"]]
        rate = len(verified) / len(candidates) if candidates else 0
        check("claims extracted", bool(candidates), f"{len(candidates)} candidates")
        check("quote verification rate ≥ 70%", rate >= 0.7, f"{rate:.0%} of quotes found verbatim in the page")

        step(4, "save_evidence (protected: re-verifies every quote)")
        saved = []
        for c in candidates:
            out = evidence.save_evidence(ctx, source_id=c["source_id"], subquestion_id="sq1",
                                         claim=c["claim"], quote=c["quote"])
            if out["saved"]:
                saved.append(out["evidence_id"])
        fake = evidence.save_evidence(ctx, source_id=scraped[0]["source_id"], subquestion_id="sq1",
                                      claim="Self-hosting is free for every startup.",
                                      quote="Self-hosting is completely free for every startup in every scenario.")
        check("verified claims saved", bool(saved), f"{len(saved)} saved: {', '.join(saved)}")
        check("fabricated quote rejected", fake["saved"] is False, fake.get("reason", ""))

        step(5, "score_source_quality + injected page")
        for page in scraped:
            q = evidence.score_source_quality(ctx, page["source_id"])
            print(f"      {page['source_id']}: {q['domain_type']} base {q['base']} recency {q['recency']:+} → {q['score']}")
        injected = await web.scrape_page(ctx, P10)
        check("injected page flagged + sanitized",
              injected["injection_detected"] and "ignore all previous" not in injected["content"].lower(),
              f"quality capped at {injected['quality_score']}, {injected['sentences_removed']} sentence(s) removed")

        step(6, "compare_sources (live LLM)")
        if len(saved) >= 2:
            cmp = await analysis.compare_sources(ctx)
            a = cmp["analysis"]
            print(f"      agreements: {len(a['agreements'])} | contradictions: {len(a['contradictions'])} | "
                  f"dropped invalid ids: {cmp['dropped_invalid']}")
            for c in a["contradictions"][:2]:
                print(f"      [conflict] {c['topic']}: {c['evidence_ids_a']} vs {c['evidence_ids_b']}")
            check("analysis returned", True)
        else:
            check("analysis returned", False, "need at least 2 saved evidence items")

        step(7, "retrieve_evidence")
        hits = evidence.retrieve_evidence(ctx, "engineering time maintenance cost of self-hosting", k=3)["results"]
        for h in hits:
            print(f"      {h['evidence_id']} score {h['score']}: {h['claim'][:70]}")
        check("retrieval returns evidence", bool(hits))

        step(8, "validate_citation (live LLM): faithful vs overreaching claim")
        if saved:
            ev = ctx.store.get_evidence(ctx.run_id, saved[0])
            faithful = await analysis.validate_citation(ctx, evidence_id=ev.id, claim=ev.claim)
            stretched = await analysis.validate_citation(
                ctx, evidence_id=ev.id,
                claim=f"{ev.claim.rstrip('.')} — and this holds for every company of every size, always.")
            print(f"      faithful claim  → {faithful['support']}")
            print(f"      stretched claim → {stretched['support']} ({stretched.get('overreach', '')[:70]})")
            check("judge accepts the faithful claim", faithful["support"] in ("supported", "partial"))
            check("judge flags the overreaching claim", stretched["support"] in ("partial", "not_supported"))
        fake_cite = await analysis.validate_citation(ctx, evidence_id="ev999", claim="Anything.")
        check("invented evidence id rejected", fake_cite["valid"] is False, fake_cite["reason"])

        step(9, "protected actions")
        try:
            await web.scrape_page(ctx, "http://localhost/admin")
            check("scrape localhost blocked", False)
        except ToolError as exc:
            check("scrape localhost blocked", True, str(exc))
        ctx.final_report = ResearchReport(
            title="Tools check report", executive_summary="Placeholder summary used only to test the token gate.",
            findings=[Finding(statement="Placeholder finding for the token gate test.", evidence_ids=saved[:1] or ["ev1"],
                              strength="weak")],
            contradictions=[], limitations=[], recommendation="Placeholder recommendation for the gate test.",
            impact="low")
        try:
            review.generate_report(ctx, "guessed-token")
            check("generate_report without token blocked", False)
        except ToolError as exc:
            check("generate_report without token blocked", True, str(exc)[:60])
        rv = review.request_human_approval(ctx, ["tools check"])
        rv2 = review.request_human_approval(ctx, ["tools check"])
        check("human approval idempotent", rv["status"] == "pending" and rv2["already_requested"])

    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    print("Tool log:", f"{sum(r.ok for r in ctx.tool_log)} ok / {sum(not r.ok for r in ctx.tool_log)} blocked-or-failed")
    print("\nTools check:", "PASSED" if not failed else "FAILED — see [fail] above")
    return 0 if not failed else 1


async def guarded(live_search: bool) -> int:
    try:
        return await main(live_search)
    except RateLimitError as exc:
        print(f"\n[rate limit] {exc.message} — wait a minute and run again.")
    except APITimeoutError:
        print("\n[timeout] The model did not answer in time — run again or lower LLM_REASONING_EFFORT.")
    except APIStatusError as exc:
        print(f"\n[api] {exc.status_code}: {exc.message}")
    except APIConnectionError:
        print("\n[network] Could not reach the LLM provider — check your connection.")
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-search", action="store_true", help="use the SEARCH_MODE from .env instead of offline")
    raise SystemExit(asyncio.run(guarded(parser.parse_args().live_search)))
