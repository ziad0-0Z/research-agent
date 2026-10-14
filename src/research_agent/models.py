"""Data models.

Two families:
- LLM output models: what agents must return (`output_type=`). Kept free of
  defaults and exotic types so they convert to strict JSON schemas.
- Internal records: what Python stores and validates (sources, evidence, reviews,
  memory). The LLM never fills these directly.
"""
from __future__ import annotations

import hashlib
import re
from datetime import date, datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

SpecialistType = Literal["technical", "market", "risk"]
Domain = Literal["technology", "business", "financial", "medical", "legal", "other"]


# ---------------------------------------------------------------------------
# LLM output models
# ---------------------------------------------------------------------------

class GuardrailVerdict(BaseModel):
    """Guard Agent output: is the request allowed, and what kind is it."""

    model_config = ConfigDict(extra="forbid")

    verdict: Literal["allow", "block"]
    category: Literal["valid_research", "prompt_injection", "fabrication_request", "harmful", "off_scope"]
    domain: Domain
    reason: str = Field(min_length=3, max_length=400)


class SubQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^sq\d{1,2}$", description="sq1, sq2, ...")
    question: str = Field(min_length=10, max_length=300)
    type: SpecialistType
    priority: int = Field(ge=1, le=3, description="1 = most important")
    success_criteria: str = Field(min_length=10, max_length=300)
    search_queries: list[str] = Field(min_length=1, max_length=4)

    @field_validator("search_queries")
    @classmethod
    def _queries_not_blank(cls, value: list[str]) -> list[str]:
        cleaned = [q.strip() for q in value if q.strip()]
        if not cleaned:
            raise ValueError("at least one non-empty search query is required")
        return cleaned


class ResearchPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    main_question: str = Field(min_length=10)
    decision_context: str = Field(min_length=5, max_length=500)
    subquestions: list[SubQuestion] = Field(min_length=2, max_length=6)

    @model_validator(mode="after")
    def _unique_ids(self) -> "ResearchPlan":
        ids = [sq.id for sq in self.subquestions]
        if len(ids) != len(set(ids)):
            raise ValueError(f"subquestion ids must be unique, got {ids}")
        return self

    def types(self) -> set[str]:
        return {sq.type for sq in self.subquestions}


class Assignment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    specialist: SpecialistType
    subquestion_ids: list[str] = Field(min_length=1)


class OrchestrationDecision(BaseModel):
    """Research Orchestrator output: which specialists run, on which subquestions, how deep."""

    model_config = ConfigDict(extra="forbid")

    assignments: list[Assignment] = Field(min_length=1, max_length=3)
    depth: Literal["quick", "standard", "deep"]
    rationale: str = Field(min_length=5, max_length=500)

    @model_validator(mode="after")
    def _merge_duplicate_specialists(self) -> "OrchestrationDecision":
        merged: dict[str, list[str]] = {}
        for a in self.assignments:
            merged.setdefault(a.specialist, [])
            merged[a.specialist] += [s for s in a.subquestion_ids if s not in merged[a.specialist]]
        self.assignments = [Assignment(specialist=k, subquestion_ids=v) for k, v in merged.items()]  # type: ignore[arg-type]
        return self

    @property
    def specialists(self) -> list[str]:
        return [a.specialist for a in self.assignments]


class SubquestionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subquestion_id: str
    summary: str = Field(description="What the saved evidence says, citing evidence ids like [ev3]")
    evidence_ids: list[str]
    open_issues: str = Field(description="What could not be found or stays uncertain; empty string if none")


class ResearcherOutput(BaseModel):
    """What a specialist returns after using its tools."""

    model_config = ConfigDict(extra="forbid")

    results: list[SubquestionResult]


class ExtractedClaim(BaseModel):
    """One claim pulled from a page, with the verbatim quote that supports it."""

    model_config = ConfigDict(extra="forbid")

    claim: str = Field(min_length=10, max_length=400)
    quote: str = Field(min_length=10, max_length=600)
    subquestion_id: str


class ClaimExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: list[ExtractedClaim] = Field(max_length=8)


class ClaimSupportVerdict(BaseModel):
    """Second, separate check after quote verification.

    Quote verification (Python) proves the quote exists in the page.
    This (LLM judge) decides whether that quote actually supports the claim:
    a real quote can still be stretched into a claim it doesn't support.
    """

    model_config = ConfigDict(extra="forbid")

    support: Literal["supported", "partial", "not_supported"]
    reason: str = Field(min_length=5, max_length=400)
    overreach: str = Field(description="What the claim says beyond the quote; empty string if nothing")


class Contradiction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: str
    evidence_ids_a: list[str] = Field(min_length=1)
    evidence_ids_b: list[str] = Field(min_length=1)
    explanation: str = Field(description="Why the sources may disagree (scope, date, assumptions)")


class EvidenceAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agreements: list[str]
    contradictions: list[Contradiction]
    gaps: list[str] = Field(description="Subquestion ids with weak or no evidence")


_EV_ID = re.compile(r"\bev\s*[-_]?\s*(\d+)\b", re.IGNORECASE)


def normalize_evidence_id(raw: str) -> str:
    """Fix formatting only: "id:ev1", "[EV1]", "ev_1" -> "ev1". Never guesses a missing prefix
    ("v10" stays "v10"), so a wrong id is still caught by the citation check."""
    m = _EV_ID.search(raw or "")
    return f"ev{int(m.group(1))}" if m else (raw or "").strip()


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    statement: str = Field(min_length=10)
    evidence_ids: list[str] = Field(min_length=1)
    strength: Literal["strong", "moderate", "weak"]

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def _normalize_ids(cls, v: Any) -> Any:
        if not isinstance(v, list):
            return v
        return list(dict.fromkeys(normalize_evidence_id(str(x)) for x in v))  # dedupe, keep order


class ResearchReport(BaseModel):
    """Writer output. Confidence is deliberately absent: Python computes it."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=5, max_length=200)
    executive_summary: str = Field(min_length=30)
    findings: list[Finding] = Field(min_length=1)
    contradictions: list[str]
    limitations: list[str]
    recommendation: str = Field(min_length=20)
    impact: Literal["low", "medium", "high"]

    def cited_evidence_ids(self) -> set[str]:
        return {eid for f in self.findings for eid in f.evidence_ids}


class Critique(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["PASS", "REVISE", "FAIL"]
    unsupported_claims: list[str]
    missing_sources: list[str]
    contradictions_ignored: list[str]
    revision_instructions: list[str]

    @model_validator(mode="after")
    def _revise_needs_instructions(self) -> "Critique":
        if self.verdict == "REVISE" and not self.revision_instructions:
            raise ValueError("REVISE verdict requires revision_instructions")
        return self


# ---------------------------------------------------------------------------
# Internal records (filled and validated by Python)
# ---------------------------------------------------------------------------

DomainType = Literal["official", "academic", "news", "vendor", "blog", "forum", "unknown"]
Origin = Literal["tavily", "ddg", "offline", "memory"]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def content_hash(*parts: str) -> str:
    joined = "␟".join(normalize_text(p) for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


class Source(BaseModel):
    id: str
    url: HttpUrl
    title: str
    domain_type: DomainType = "unknown"
    published: date | None = None
    origin: Origin
    quality_score: float = Field(default=0.5, ge=0, le=1)
    injection_detected: bool = False
    retrieved_at: datetime = Field(default_factory=utcnow)


SupportLevel = Literal["unchecked", "supported", "partial", "not_supported"]


class Evidence(BaseModel):
    id: str
    run_id: str
    subquestion_id: str
    claim: str = Field(min_length=10)
    quote: str = Field(min_length=10)
    source_id: str
    verified: bool  # quote verified: the quote exists in the source text (Python check)
    support: SupportLevel = "unchecked"  # claim supported by the quote (LLM judge), set later
    content_hash: str
    retrieved_at: datetime = Field(default_factory=utcnow)
    status: Literal["active", "stale", "superseded"] = "active"

    @classmethod
    def build(cls, *, evidence_id: str, run_id: str, subquestion_id: str, claim: str,
              quote: str, source_id: str, verified: bool) -> "Evidence":
        return cls(
            id=evidence_id, run_id=run_id, subquestion_id=subquestion_id, claim=claim,
            quote=quote, source_id=source_id, verified=verified,
            content_hash=content_hash(source_id, quote),
        )


WorkerStatus = Literal["success", "partial", "failed", "timeout"]


class WorkerFindings(BaseModel):
    specialist: SpecialistType
    status: WorkerStatus
    subquestion_ids: list[str]
    evidence_ids: list[str] = Field(default_factory=list)
    summaries: dict[str, str] = Field(default_factory=dict)
    error: str | None = None
    started_at: float
    finished_at: float

    @property
    def duration(self) -> float:
        return round(self.finished_at - self.started_at, 2)


class HumanReview(BaseModel):
    run_id: str
    reasons: list[str] = Field(min_length=1)
    status: Literal["pending", "approved", "rejected", "changes_requested"] = "pending"
    reviewer_note: str = ""
    created_at: datetime = Field(default_factory=utcnow)
    decided_at: datetime | None = None


class MemoryRecord(BaseModel):
    kind: Literal["evidence", "report", "user_context"]
    ref_id: str
    text: str
    claim_type: Literal["pricing", "technical", "general"] = "general"
    created_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime | None = None
