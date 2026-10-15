"""Answer-quality checks computed in code (no LLM): corroboration, coverage and the recommendation policy.

    independent sources   distinct sites behind the cited, usable evidence. Two pages of one site, or the same
                          sentence quoted twice, count once: a finding is corroborated only by sources that
                          could have disagreed.
    coverage              which subquestions have usable evidence, and which dimensions the question names
                          (cost, latency, lock-in...) no subquestion asks about.
    recommendation policy below MIN_COVERAGE_FOR_VERDICT the report may not pick an option: the code replaces
                          the recommendation with "Insufficient evidence to choose" and the list of what is missing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

from ..context import AppContext
from ..models import ResearchPlan, ResearchReport

INSUFFICIENT_PREFIX = "Insufficient evidence to choose"


def normalize_quote(text: str) -> str:
    """Case, punctuation and British/American spelling folded, so a sentence quoted twice compares equal."""
    t = re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower())
    t = t.replace("isation", "ization").replace("labour", "labor").replace("organis", "organiz")
    return " ".join(t.split())


def site_of(url: str) -> str:
    host = (urlparse(str(url)).hostname or str(url)).lower()
    return host[4:] if host.startswith("www.") else host


def independent_sources(ctx: AppContext, evidence_ids: set[str] | list[str]) -> set[str]:
    """Distinct sites behind the given evidence, ignoring evidence judged not_supported and repeated quotes."""
    sites: set[str] = set()
    seen_quotes: set[str] = set()
    for eid in sorted(set(evidence_ids)):
        ev = ctx.store.get_evidence(ctx.run_id, eid)
        if ev is None or not ev.verified or ev.support == "not_supported":
            continue
        q = normalize_quote(ev.quote)
        if q in seen_quotes:
            continue
        seen_quotes.add(q)
        src = ctx.store.get_source(ctx.run_id, ev.source_id)
        if src is not None:
            sites.add(site_of(str(src.url)))
    return sites


def corroborated_findings(ctx: AppContext, report: ResearchReport, minimum: int = 2) -> int:
    return sum(1 for f in report.findings if len(independent_sources(ctx, f.evidence_ids)) >= minimum)


# --------------------------------------------------------------------------- coverage

# The dimensions decision questions usually name, with the words that show a subquestion covers them.
DIMENSIONS: dict[str, tuple[str, ...]] = {
    "cost": ("cost", "price", "pricing", "tco", "ownership", "budget", "spend", "cheap", "expens"),
    "quality": ("quality", "accuracy", "accurate", "relevance", "retrieval quality", "recall", "precision"),
    "latency": ("latency", "response time", "speed", "fast", "slow"),
    "scalability": ("scal", "growth", "grow", "volume", "throughput"),
    "privacy": ("privacy", "personal data", "gdpr", "data protection"),
    "security": ("security", "secure", "attack", "breach"),
    "compliance": ("compliance", "regulat", "legal", "audit"),
    "lock-in": ("lock-in", "lock in", "lockin", "vendor", "portab", "switching"),
    "operational complexity": ("operational", "maintenance", "ops", "complexity", "operate", "engineering effort"),
    "filtering": ("filter",),
    "time to launch": ("time to launch", "time required", "launch", "time to market", "setup time"),
}


def question_dimensions(question: str) -> list[str]:
    q = (question or "").lower()
    return [name for name, words in DIMENSIONS.items() if any(w in q for w in words)]


def plan_gaps(plan: ResearchPlan) -> list[str]:
    """Dimensions the question names that no subquestion (question, criteria or queries) asks about."""
    text = " ".join(f"{sq.question} {sq.success_criteria} {' '.join(sq.search_queries)}"
                    for sq in plan.subquestions).lower()
    return [d for d in question_dimensions(plan.main_question) if not any(w in text for w in DIMENSIONS[d])]


@dataclass
class Coverage:
    covered: list[str] = field(default_factory=list)       # subquestion ids with usable evidence
    missing: list[str] = field(default_factory=list)       # subquestion ids without
    missing_questions: list[str] = field(default_factory=list)
    unplanned: list[str] = field(default_factory=list)     # named dimensions with no subquestion

    @property
    def share(self) -> float:
        total = len(self.covered) + len(self.missing)
        return len(self.covered) / total if total else 0.0

    def summary(self) -> dict:
        return {"covered": self.covered, "missing": self.missing, "missing_questions": self.missing_questions,
                "unplanned_dimensions": self.unplanned, "share": round(self.share, 2)}


def coverage(ctx: AppContext, plan: ResearchPlan, unplanned: list[str] | None = None) -> Coverage:
    have = {ev.subquestion_id for ev in ctx.store.list_evidence(ctx.run_id)
            if ev.status == "active" and ev.verified and ev.support != "not_supported"}
    cov = Coverage(unplanned=list(unplanned or []))
    for sq in plan.subquestions:
        if sq.id in have:
            cov.covered.append(sq.id)
        else:
            cov.missing.append(sq.id)
            cov.missing_questions.append(sq.question)
    return cov


def coverage_prompt_block(cov: Coverage, min_share: float) -> str:
    """Told to the Writer: what is covered, what is missing, and the policy that applies."""
    lines = [f"Coverage (computed in code): {len(cov.covered)} of {len(cov.covered) + len(cov.missing)} "
             f"subquestions have usable evidence."]
    if cov.missing:
        lines.append("No evidence for: " + "; ".join(f"{i} ({q})" for i, q in zip(cov.missing, cov.missing_questions)))
    if cov.unplanned:
        lines.append("The question names these dimensions but no subquestion researched them: "
                     + ", ".join(cov.unplanned) + ". List them in limitations.")
    if cov.share < min_share:
        lines.append(f"POLICY: coverage is below {min_share:.0%}. Do not choose an option. The recommendation must "
                     f"start with \"{INSUFFICIENT_PREFIX}\" and say what data is missing.")
    return "\n".join(lines)


def enforce_recommendation_policy(report: ResearchReport, cov: Coverage, min_share: float
                                  ) -> tuple[ResearchReport, bool]:
    """Below the coverage threshold the code, not the Writer, has the last word on the recommendation."""
    limitations = list(report.limitations)
    for d in cov.unplanned:
        if not any(d.split()[0] in x.lower() for x in limitations):
            limitations.append(f"The question asks about {d}, but no subquestion researched it.")
    changed = limitations != list(report.limitations)
    if cov.share >= min_share or report.recommendation.strip().startswith(INSUFFICIENT_PREFIX):
        return (report.model_copy(update={"limitations": limitations}) if changed else report), False
    missing = "; ".join(cov.missing_questions[:6]) or "most subquestions"
    text = (f"{INSUFFICIENT_PREFIX}: only {len(cov.covered)} of {len(cov.covered) + len(cov.missing)} subquestions "
            f"have usable evidence. Missing: {missing}. Research these before deciding; the findings above are "
            f"partial and do not justify picking an option.")
    return report.model_copy(update={"recommendation": text, "limitations": limitations}), True
