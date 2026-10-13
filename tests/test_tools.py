"""The 11 tools, run against the offline corpus with a fake LLM where needed."""
import asyncio
import json
from datetime import date

import pytest
from agents import RunContextWrapper
from agents.tool_context import ToolContext

from research_agent import config as cfg
from research_agent.context import AppContext
from research_agent.models import (
    ClaimExtraction, ClaimSupportVerdict, Contradiction, EvidenceAnalysis, ExtractedClaim, Finding, ResearchReport,
)
from research_agent.store import Store
from research_agent.tools import analysis, evidence, review, web
from research_agent.tools.corpus import Corpus
from research_agent.tools.quality import classify_domain, score_quality
from research_agent.tools.quotes import verify_quote
from research_agent.tools.registry import ALL_TOOLS, generate_report as generate_report_tool
from research_agent.tools.web import ToolError

P01 = "https://northwind-tech-review.example/2026/08/llm-api-pricing-small-teams"
P03 = "https://applied-systems-institute.example/reports/llm-tco-small-teams"
P04 = "https://devpath-blog.example/self-hosting-llm-always-cheaper"
P10 = "https://scaleup-digest.example/hidden-costs-llm-apis"
P03_QUOTE = "Below that threshold, hosted APIs had the lower TCO in every scenario we modelled."


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def ctx(monkeypatch, tmp_path):
    monkeypatch.setenv("SEARCH_MODE", "offline")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-never-used")  # agents are built, the fake runner answers
    from research_agent import llm
    from research_agent.agents import tool_agents
    llm.reset_model()
    for factory in (tool_agents.claim_extractor, tool_agents.evidence_analyst, tool_agents.support_judge):
        factory.cache_clear()
    settings = cfg.load_settings()
    return AppContext(settings=settings, store=Store(":memory:"), question="Cloud APIs or self-hosting?")


def fake_runner(output):
    async def _run(agent, prompt):
        _run.prompts.append(prompt)
        return output
    _run.prompts = []
    return _run


# ---------------------------------------------------------------- registry

def test_eleven_tools_with_strict_schemas():
    names = [t.name for t in ALL_TOOLS]
    assert len(names) == 11 and len(set(names)) == 11
    for required in ["search_web", "scrape_page", "extract_claims", "compare_sources", "save_evidence",
                     "generate_report", "validate_source", "validate_citation"]:
        assert required in names
    assert all(t.strict_json_schema for t in ALL_TOOLS)


# ---------------------------------------------------------------- corpus + search

def test_corpus_loads_all_pages():
    corpus = Corpus(cfg.PROJECT_ROOT / "data" / "corpus")
    assert len(corpus.pages) == 15
    assert all(p.url.split("/")[2].endswith(".example") for p in corpus.pages)


def test_offline_search_finds_relevant_pages(ctx):
    out = run(web.search_web(ctx, "total cost of ownership self-hosting small team engineering time"))
    urls = [r["url"] for r in out["results"]]
    assert out["provider"] == "offline" and P03 in urls[:3]


def test_search_rejects_bad_queries(ctx):
    with pytest.raises(ToolError):
        run(web.search_web(ctx, "a"))


def test_live_mode_falls_back_to_offline(ctx, monkeypatch):
    # Tavily has no key and DuckDuckGo fails: the chain must end in the offline corpus, labelled.
    object.__setattr__(ctx.settings, "search_mode", "tavily")
    object.__setattr__(ctx.settings, "tavily_api_key", "")

    async def ddg_down(query, n):
        raise ConnectionError("network unreachable")

    monkeypatch.setattr(web, "_search_ddg", ddg_down)
    out = run(web.search_web(ctx, "gpu rental break even"))
    assert out["provider"] == "offline" and out["fallback_used"] is True
    assert len(out["errors"]) == 2 and out["results"]


# ---------------------------------------------------------------- scrape + validate

def test_scrape_registers_source(ctx):
    out = run(web.scrape_page(ctx, P03))
    assert out["source_id"] == "src1" and out["domain_type"] == "academic"
    assert out["content"].startswith("<untrusted_content")
    assert ctx.store.get_source(ctx.run_id, "src1").url.host == "applied-systems-institute.example"


def test_scrape_same_url_is_cached(ctx):
    a = run(web.scrape_page(ctx, P03))
    b = run(web.scrape_page(ctx, P03))
    assert a["source_id"] == b["source_id"] and b["cached"] is True


@pytest.mark.parametrize("url", ["http://localhost/admin", "file:///etc/passwd", "http://169.254.169.254/"])
def test_scrape_blocks_unsafe_urls(ctx, url):
    with pytest.raises(ToolError, match="blocked"):
        run(web.scrape_page(ctx, url))
    assert ctx.tool_log[-1].ok is False and "BLOCKED" in ctx.tool_log[-1].summary


def test_scrape_unknown_page_offline(ctx):
    with pytest.raises(ToolError, match="offline"):
        run(web.scrape_page(ctx, "https://not-in-corpus.example/page"))


def test_scrape_injected_page_is_flagged_and_sanitized(ctx):
    out = run(web.scrape_page(ctx, P10))
    assert out["injection_detected"] is True and out["sentences_removed"] >= 1
    assert out["quality_score"] <= 0.10
    assert "ignore all previous instructions" not in out["content"].lower()
    assert "caching repeated prompts" in out["content"]


def test_validate_source(ctx):
    assert run(web.validate_source(ctx, P01))["reachable"] is True
    blocked = run(web.validate_source(ctx, "http://127.0.0.1/"))
    assert blocked["allowed"] is False


# ---------------------------------------------------------------- quotes + save_evidence

def test_verify_quote_exact_and_tolerant():
    text = "Below that threshold, hosted APIs had the lower TCO in every scenario we modelled."
    assert verify_quote(text, text)[0]
    assert verify_quote("“Below that threshold, hosted APIs had the lower TCO”", text)[0]
    assert verify_quote("Below that threshold hosted APIs had the lower TCO in every scenario we modeled", text)[0]


def test_verify_quote_rejects_fabrication():
    text = "Below that threshold, hosted APIs had the lower TCO in every scenario we modelled."
    assert not verify_quote("Self-hosting had the lower TCO in every scenario we modelled for startups", text)[0]
    assert not verify_quote("APIs win", text)[0]  # too short to be meaningful


def test_save_evidence_accepts_real_quote(ctx):
    run(web.scrape_page(ctx, P03))
    out = evidence.save_evidence(ctx, source_id="src1", subquestion_id="sq1",
                                 claim="For small teams, APIs had lower total cost below several billion tokens/month.",
                                 quote=P03_QUOTE)
    assert out["saved"] and out["evidence_id"] == "ev1"


def test_save_evidence_rejects_fabricated_quote(ctx):
    run(web.scrape_page(ctx, P03))
    out = evidence.save_evidence(ctx, source_id="src1", subquestion_id="sq1",
                                 claim="Self-hosting is always cheaper.",
                                 quote="Self-hosting was cheaper in every scenario we modelled for all teams.")
    assert out["saved"] is False and "not found" in out["reason"]
    assert ctx.store.list_evidence(ctx.run_id) == []


def test_save_evidence_rejects_unknown_source_and_duplicates(ctx):
    out = evidence.save_evidence(ctx, source_id="src9", subquestion_id="sq1", claim="Some claim here.", quote=P03_QUOTE)
    assert out["saved"] is False and "unknown source" in out["reason"]
    run(web.scrape_page(ctx, P03))
    args = dict(source_id="src1", subquestion_id="sq1", claim="APIs cheaper for small teams.", quote=P03_QUOTE)
    assert evidence.save_evidence(ctx, **args)["saved"]
    dup = evidence.save_evidence(ctx, **args)
    assert dup["saved"] is False and "duplicate" in dup["reason"]
    nxt = evidence.save_evidence(ctx, source_id="src1", subquestion_id="sq1", claim="Labour dominates for small teams.",
                                 quote="For small teams, labour dominates the comparison.")
    assert nxt["evidence_id"] == "ev2"  # ids stay dense after a rejected duplicate


def test_quote_from_removed_injection_cannot_be_saved_as_clean(ctx):
    run(web.scrape_page(ctx, P10))
    out = evidence.save_evidence(ctx, source_id="src1", subquestion_id="sq1",
                                 claim="Caching cuts API bills.", quote="caching repeated prompts, trimming retrieved context")
    assert out["saved"] and "injection" in out["warning"]


# ---------------------------------------------------------------- quality

def test_quality_scores_rank_sources_sensibly(ctx):
    academic = score_quality(domain_type="academic", published=date(2026, 3, 1), text_length=2000,
                             injection_detected=False, today=date(2026, 10, 14))
    marketing = score_quality(domain_type="vendor", published=date(2026, 9, 1), text_length=400,
                              injection_detected=False, today=date(2026, 10, 14))
    injected = score_quality(domain_type="blog", published=date(2026, 5, 1), text_length=2000,
                             injection_detected=True, today=date(2026, 10, 14))
    old = score_quality(domain_type="news", published=date(2021, 1, 1), text_length=2000,
                        injection_detected=False, today=date(2026, 10, 14))
    assert academic["score"] > old["score"] > marketing["score"] > injected["score"]
    assert injected["score"] <= 0.10


@pytest.mark.parametrize("url,expected", [
    ("https://www.nist.gov/report", "official"), ("https://arxiv.org/abs/1", "academic"),
    ("https://www.reddit.com/r/x", "forum"), ("https://medium.com/@a/b", "blog"),
    ("https://openai.com/pricing", "vendor"), ("https://random-site.example/x", "unknown"),
])
def test_classify_domain(url, expected):
    assert classify_domain(url) == expected


def test_score_source_quality_tool(ctx):
    run(web.scrape_page(ctx, P04))
    out = evidence.score_source_quality(ctx, "src1")
    assert out["domain_type"] == "blog" and "recency" in out


# ---------------------------------------------------------------- retrieval

def test_retrieve_evidence_prefers_relevant_items(ctx):
    run(web.scrape_page(ctx, P03))
    run(web.scrape_page(ctx, P01))
    evidence.save_evidence(ctx, source_id="src1", subquestion_id="sq1", claim="APIs had lower TCO for small teams.",
                           quote=P03_QUOTE)
    evidence.save_evidence(ctx, source_id="src2", subquestion_id="sq2", claim="Output tokens cost 3-5x input tokens.",
                           quote="Output tokens typically cost three to five times more than input tokens.")
    out = evidence.retrieve_evidence(ctx, "output token price compared to input", k=2)
    assert out["results"][0]["evidence_id"] == "ev2"


# ---------------------------------------------------------------- LLM-backed tools (fake model)

def test_extract_claims_verifies_quotes_in_code(ctx):
    run(web.scrape_page(ctx, P03))
    fake = fake_runner(ClaimExtraction(claims=[
        ExtractedClaim(claim="APIs had lower TCO for small teams.", quote=P03_QUOTE, subquestion_id="sq1"),
        ExtractedClaim(claim="Self-hosting is always cheaper.", quote="Self-hosting is always cheaper for every team we studied.",
                       subquestion_id="sq1"),
    ]))
    out = run(analysis.extract_claims(ctx, source_id="src1", subquestion_id="sq1",
                                      subquestion="Which option has lower total cost?", run_agent=fake))
    assert [c["quote_verified"] for c in out["candidates"]] == [True, False]
    assert "<untrusted_content" in fake.prompts[0]
    assert ctx.store.list_evidence(ctx.run_id) == []  # extraction never saves on its own


def test_extract_claims_never_shows_injection_to_model(ctx):
    run(web.scrape_page(ctx, P10))
    fake = fake_runner(ClaimExtraction(claims=[]))
    run(analysis.extract_claims(ctx, source_id="src1", subquestion_id="sq1", subquestion="Hidden API costs?",
                                run_agent=fake))
    assert "ignore all previous instructions" not in fake.prompts[0].lower()


def _two_evidence(ctx):
    run(web.scrape_page(ctx, P03))
    run(web.scrape_page(ctx, P04))
    evidence.save_evidence(ctx, source_id="src1", subquestion_id="sq1", claim="APIs had lower TCO for small teams.",
                           quote=P03_QUOTE)
    evidence.save_evidence(ctx, source_id="src2", subquestion_id="sq1", claim="The author claims self-hosting is always cheaper.",
                           quote="In my view self-hosting is always cheaper than paying per token")


def test_compare_sources_drops_invented_ids(ctx):
    _two_evidence(ctx)
    fake = fake_runner(EvidenceAnalysis(
        agreements=[],
        contradictions=[
            Contradiction(topic="cost", evidence_ids_a=["ev1"], evidence_ids_b=["ev2"],
                          explanation="The report counts engineering labour; the blog ignores it."),
            Contradiction(topic="made up", evidence_ids_a=["ev1"], evidence_ids_b=["ev99"], explanation="invented"),
        ],
        gaps=["sq2", "sq7"],
    ))
    out = run(analysis.compare_sources(ctx, subquestion_ids=["sq1", "sq2"], run_agent=fake))
    assert len(out["analysis"]["contradictions"]) == 1 and out["dropped_invalid"] == 1
    assert out["analysis"]["gaps"] == ["sq2"]
    assert ctx.analysis is not None
    payload = json.loads(fake.prompts[0].split("(JSON):\n", 1)[1])
    assert {i["source_type"] for i in payload} == {"academic", "blog"}


def test_validate_citation_real_quote_but_unsupported_claim(ctx):
    # The key grounding case: the quote is real, the claim overreaches.
    _two_evidence(ctx)
    fake = fake_runner(ClaimSupportVerdict(support="not_supported", reason="Quote limits this to small teams.",
                                           overreach="claims it for all companies"))
    out = run(analysis.validate_citation(ctx, evidence_id="ev1", claim="APIs are always cheaper for every company.",
                                         run_agent=fake))
    assert out["valid"] is False and out["support"] == "not_supported"
    ev = ctx.store.get_evidence(ctx.run_id, "ev1")
    assert ev.verified is True and ev.support == "not_supported"


def test_validate_citation_unknown_id_skips_llm(ctx):
    fake = fake_runner(None)
    out = run(analysis.validate_citation(ctx, evidence_id="ev42", claim="Anything.", run_agent=fake))
    assert out["valid"] is False and "does not exist" in out["reason"] and fake.prompts == []


# ---------------------------------------------------------------- review + protected report

def _report():
    return ResearchReport(
        title="Cloud APIs vs self-hosting for a 5-person startup",
        executive_summary="For a small team at modest volume, hosted APIs cost less overall once labour is counted.",
        findings=[Finding(statement="APIs have lower TCO for small teams below several billion tokens per month.",
                          evidence_ids=["ev1"], strength="strong")],
        contradictions=["One blog claims self-hosting is always cheaper but ignores engineering time."],
        limitations=["Synthetic offline corpus."],
        recommendation="Start with hosted APIs behind an abstraction layer and revisit at higher volume.",
        impact="medium",
    )


def test_human_approval_is_idempotent(ctx):
    first = review.request_human_approval(ctx, ["financial data"])
    second = review.request_human_approval(ctx, ["financial data"])
    assert first["status"] == "pending" and second["already_requested"] is True
    assert len(ctx.store.pending_reviews()) == 1
    assert ctx.store.get_run(ctx.run_id)["status"] == "paused_for_review"


def test_generate_report_blocked_without_token(ctx):
    _two_evidence(ctx)
    ctx.final_report = _report()
    with pytest.raises(ToolError, match="approval token"):
        review.generate_report(ctx, "guessed-token")


def test_generate_report_blocked_until_human_approves(ctx):
    _two_evidence(ctx)
    ctx.final_report = _report()
    ctx.flag_review("financial domain")
    token = ctx.issue_approval_token()
    with pytest.raises(ToolError, match="human approval"):
        review.generate_report(ctx, token)
    rv = review.request_human_approval(ctx, ["financial domain"])
    assert rv["status"] == "pending"
    approved = ctx.store.get_review(ctx.run_id)
    approved.status = "approved"
    ctx.store.upsert_review(approved)
    out = review.generate_report(ctx, token)
    assert out["status"] == "completed"
    assert "## Evidence appendix" in out["markdown"] and "[ev1]" in out["markdown"]


def test_generate_report_tool_wrapper_returns_error_not_crash(ctx):
    tool_ctx = ToolContext(context=ctx, tool_name="generate_report", tool_call_id="call_1",
                           tool_arguments='{"approval_token": "fake"}')
    out = run(generate_report_tool.on_invoke_tool(tool_ctx, '{"approval_token": "fake"}'))
    text = json.dumps(out) if not isinstance(out, str) else out
    assert "approval token" in text and "recoverable" in text
