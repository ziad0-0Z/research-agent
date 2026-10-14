"""Memory policy: what may be remembered, for how long. Pure functions, decided in Python.

    write only when      the run's status is `completed` (released by the gate, or approved by a human)
    never write when     blocked · rejected · rejected_by_guardrail · paused · demo-fault run · MEMORY_MODE≠on
    admit evidence when  quote verified · not judged not_supported · claim and quote free of injection ·
                         source not flagged for injection
    TTL by claim type    pricing 30 d · market 90 d · risk 90 d · technical 180 d   (the shorter one wins:
                         a price found by the technical researcher still expires after 30 days)
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from urllib.parse import urlparse

from ..config import Settings
from ..guardrails.injection import detect_injection
from ..models import Evidence, Source, content_hash
from ..tools.quotes import normalize

COMMITTABLE_STATUS = "completed"

# Money, rates and price words. A hit makes the claim `pricing` whatever specialist found it.
PRICING = re.compile(
    r"[$€£]\s?\d|\b\d[\d,.]*\s?(usd|eur|dollars?|euros?|cents?)\b|\bper\s+(token|tokens|hour|month|year|seat|user|"
    r"request|gpu[- ]hour|million|1k|1m)\b|\b(price|prices|pricing|priced|cost per|subscription fee)\b",
    re.IGNORECASE)


def claim_type(specialist: str, claim: str, quote: str) -> str:
    if PRICING.search(claim) or PRICING.search(quote):
        return "pricing"
    return specialist if specialist in ("market", "risk", "technical") else "market"


def expires_at(settings: Settings, claim_type_: str, verified_at: datetime) -> datetime:
    return verified_at + timedelta(days=settings.memory_ttl(claim_type_))


def memory_hash(url: str, quote: str) -> str:
    """Identity of a memory item: the same quote from the same page, whatever run found it."""
    return content_hash(str(url).rstrip("/"), quote)


def page_hash(url: str, text: str) -> str:
    return content_hash(str(url).rstrip("/"), text)


def host(url: str) -> str:
    h = (urlparse(str(url)).hostname or "").lower()
    return h[4:] if h.startswith("www.") else h


def can_commit(settings: Settings, run_status: str | None, demo_fault: str | None) -> tuple[bool, str]:
    if settings.memory_mode != "on":
        return False, f"MEMORY_MODE={settings.memory_mode}"
    if demo_fault:
        return False, f"demo fault run ({demo_fault}) never writes to memory"
    if run_status != COMMITTABLE_STATUS:
        return False, f"run status is {run_status!r}; only completed runs are remembered"
    return True, "ok"


def admit_evidence(ev: Evidence, source: Source | None) -> tuple[bool, str]:
    if source is None:
        return False, "source missing"
    if not ev.verified:
        return False, "quote was never verified"
    if ev.status != "active":
        return False, f"evidence is {ev.status}"
    if ev.support == "not_supported":
        return False, "judged not_supported"
    if source.injection_detected:
        return False, "source flagged for prompt injection"
    if detect_injection(ev.claim).detected or detect_injection(ev.quote).detected:
        return False, "claim or quote contains injected instructions"
    return True, "ok"


_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
SAME_FACT_MIN_OVERLAP = 0.5


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUMBER.findall(text)}


def same_fact(a_url: str, a_quote: str, a_claim: str, b_url: str, b_quote: str, b_claim: str) -> bool:
    """Two remembered items are one fact when ALL hold:
      1. same page, and one quote is an excerpt of the other
      2. the claims' numbers agree (one claim's numbers are a subset of the other's)
      3. the claims share at least half their words (overlap coefficient)
    Rule 2 is what keeps apart "frontier models cost $2-$15 per million tokens" and "small models cost
    $0.10-$0.60 per million tokens": one quote holds both, the words overlap 0.86, the facts differ."""
    from .matching import content_terms

    if a_url.rstrip("/") != b_url.rstrip("/"):
        return False
    qa, qb = normalize(a_quote), normalize(b_quote)
    if qa not in qb and qb not in qa:
        return False
    na, nb = _numbers(a_claim), _numbers(b_claim)
    if not (na <= nb or nb <= na):
        return False
    ta, tb = content_terms(a_claim), content_terms(b_claim)
    return bool(ta and tb) and len(ta & tb) / min(len(ta), len(tb)) >= SAME_FACT_MIN_OVERLAP


def memory_support(run_support: str) -> str:
    """Run support levels → memory: anything not yet judged by the Critic is `unjudged`."""
    return run_support if run_support in ("supported", "partial") else "unjudged"
