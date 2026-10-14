"""The three research specialists, built by one factory.

Same tools and process, different focus. Each specialist actually uses the
tools, batched across its subquestions: search → scrape → extract (+ protected save).
"""
from __future__ import annotations

from functools import lru_cache

from agents import Agent, ModelSettings

from ..llm import get_model, precise_settings
from ..models import ResearcherOutput
from ..tools.registry import RESEARCH_TOOLS
from .tool_agents import UNTRUSTED_RULE

FOCUS = {
    "technical": "performance, architecture, model quality, latency, hardware and engineering effort",
    "market": "costs, pricing, total cost of ownership, vendors, volume thresholds and adoption",
    "risk": "privacy, security, compliance, vendor lock-in and operational failure modes",
}

SOURCES_PER_DEPTH = {"quick": 2, "standard": 3, "deep": 4}

RESEARCHER_TEMPLATE = """You are the {kind} research specialist. Your focus: {focus}.

Work on ALL your subquestions at the same time, calling several tools in one turn.
Each turn is slow, so aim to finish in about 4 turns:

Turn 1 — search_web once per subquestion (use its first search query), all in this turn.
Turn 2 — scrape_page for up to {n_sources} relevant results per subquestion, all in this turn.
         Prefer official, academic and reputable news sources; use blogs, forums and vendor
         pages only alongside stronger sources. scrape_page checks URL safety itself, so you
         only need validate_source for a URL you are unsure about.
Turn 3 — extract_claims for every (source, subquestion) pair you scraped, all in this turn.
         Verified claims are saved as evidence automatically; you get their evidence ids.
Turn 4 — if a subquestion still has no saved evidence, search its second query and repeat
         turns 2-3 for it once. Otherwise give your final answer.

If a tool returns an error, adapt (another result, another query) instead of stopping.
Never repeat a tool call you already made. Once every subquestion has evidence, stop and answer.

Final answer: for each subquestion, a short summary of what the saved evidence says
(cite evidence ids like [ev3]), the evidence_ids saved for it, and open_issues (what you could
not find, conflicting sources, or weak-only evidence).

Rules:
- Only report what saved evidence supports. Never claim something that was not saved.
- If you found nothing usable, say so in open_issues and return an empty evidence_ids list.
- Stay within your subquestions.
{untrusted}"""


@lru_cache(maxsize=9)
def researcher(kind: str, depth: str = "standard") -> Agent:
    if kind not in FOCUS:
        raise ValueError(f"unknown specialist {kind!r}")
    instructions = RESEARCHER_TEMPLATE.format(
        kind=kind, focus=FOCUS[kind], n_sources=SOURCES_PER_DEPTH.get(depth, 3), untrusted=UNTRUSTED_RULE)
    base = precise_settings()
    settings = ModelSettings(temperature=base.temperature, reasoning=base.reasoning, parallel_tool_calls=True)
    return Agent(
        name=f"{kind.capitalize()} Researcher",
        instructions=instructions,
        model=get_model(),
        model_settings=settings,
        tools=RESEARCH_TOOLS,
        output_type=ResearcherOutput,
    )
