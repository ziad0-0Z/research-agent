"""CODE-ORCHESTRATED routing decisions (deterministic, auditable, testable).

1. route_question: sensitive domains (financial / medical / legal) take the
   review path, so a human must approve the recommendation. A safety rule like
   this must not depend on an LLM's judgement.
2. validate_assignments: the Orchestrator (LLM) proposes who researches what;
   Python enforces that every subquestion is assigned exactly once to a real
   specialist, and fixes anything that isn't.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from ..models import OrchestrationDecision, ResearchPlan, SubQuestion

SENSITIVE_TERMS: dict[str, list[str]] = {
    "financial": [
        r"banks?", r"banking", r"financial data", r"customer financial", r"credit cards?", r"card data",
        r"payment data", r"transactions? data", r"loans?", r"mortgages?", r"account balances?",
        r"insurance claims?", r"pci", r"anti-money laundering", r"aml", r"kyc", r"trading",
    ],
    "medical": [
        r"patients?", r"medical records?", r"health data", r"health records?", r"clinical", r"diagnos\w*",
        r"hospitals?", r"phi", r"hipaa", r"prescriptions?",
    ],
    "legal": [
        r"legal advice", r"lawsuits?", r"litigation", r"court", r"contract disputes?", r"legal liability",
        r"law firms?", r"attorney",
    ],
}
_COMPILED = {
    domain: re.compile(r"\b(" + "|".join(terms) + r")\b", re.IGNORECASE)
    for domain, terms in SENSITIVE_TERMS.items()
}


@dataclass
class RouteDecision:
    path: Literal["standard", "sensitive"]
    requires_review: bool
    domains: list[str] = field(default_factory=list)
    matched_terms: list[str] = field(default_factory=list)
    reason: str = ""


def route_question(question: str, guard_domain: str | None = None,
                   sensitive_domains: tuple[str, ...] = ("financial", "medical", "legal")) -> RouteDecision:
    domains: list[str] = []
    matched: list[str] = []
    for domain, rx in _COMPILED.items():
        if domain not in sensitive_domains:
            continue
        hits = sorted({m.group(0).lower() for m in rx.finditer(question or "")})
        if hits:
            domains.append(domain)
            matched += hits
    if guard_domain in sensitive_domains and guard_domain not in domains:
        domains.append(guard_domain)
        matched.append(f"guard classified domain as {guard_domain}")
    if domains:
        return RouteDecision("sensitive", True, domains, matched,
                             f"sensitive domain ({', '.join(domains)}): human review required before finalising")
    return RouteDecision("standard", False, [], [], "no sensitive domain detected")


@dataclass
class AssignmentPlan:
    by_specialist: dict[str, list[SubQuestion]]
    notes: list[str]


def validate_assignments(plan: ResearchPlan, decision: OrchestrationDecision) -> AssignmentPlan:
    """Turn the Orchestrator's proposal into a valid assignment, recording every correction."""
    known = {sq.id: sq for sq in plan.subquestions}
    assigned: dict[str, str] = {}
    notes: list[str] = []
    for a in decision.assignments:
        for sq_id in a.subquestion_ids:
            if sq_id not in known:
                notes.append(f"dropped unknown subquestion id {sq_id!r} from {a.specialist}")
            elif sq_id in assigned:
                notes.append(f"{sq_id} assigned twice; kept {assigned[sq_id]}")
            else:
                assigned[sq_id] = a.specialist
    for sq in plan.subquestions:
        if sq.id not in assigned:
            assigned[sq.id] = sq.type
            notes.append(f"{sq.id} was unassigned; given to {sq.type} (its planned type)")

    by_specialist: dict[str, list[SubQuestion]] = {}
    for sq in plan.subquestions:  # keep plan order
        by_specialist.setdefault(assigned[sq.id], []).append(sq)
    return AssignmentPlan(by_specialist, notes)
