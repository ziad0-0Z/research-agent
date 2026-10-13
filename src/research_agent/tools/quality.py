"""Deterministic source-quality scoring (code-orchestrated).

The score is explainable: every component is returned so the UI and the
presentation can show exactly why a source scored what it did.
"""
from __future__ import annotations

from datetime import date
from urllib.parse import urlsplit

DOMAIN_TYPE_BASE = {
    "official": 0.90,
    "academic": 0.85,
    "news": 0.70,
    "vendor": 0.50,
    "blog": 0.50,
    "forum": 0.35,
    "unknown": 0.40,
}

_ACADEMIC = ("arxiv.org", "acm.org", "ieee.org", "nature.com", "sciencedirect.com", "springer.com", "openreview.net")
_FORUM = ("reddit.com", "news.ycombinator.com", "stackoverflow.com", "quora.com", "stackexchange.com")
_BLOG = ("medium.com", "substack.com", "dev.to", "hashnode.dev", "wordpress.com", "blogspot.com")
_NEWS = ("reuters.com", "techcrunch.com", "theverge.com", "wired.com", "arstechnica.com", "venturebeat.com", "zdnet.com")
_VENDOR = ("openai.com", "anthropic.com", "google.com", "cloud.google.com", "aws.amazon.com", "azure.microsoft.com",
           "pinecone.io", "qdrant.tech", "huggingface.co", "together.ai", "groq.com", "nvidia.com")


def classify_domain(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    if host.endswith((".gov", ".gov.uk", ".europa.eu", ".int")) or ".gov." in host:
        return "official"
    if host.endswith((".edu", ".ac.uk")) or ".edu." in host or host.endswith(_ACADEMIC):
        return "academic"
    if host.endswith(_FORUM):
        return "forum"
    if host.endswith(_BLOG) or host.startswith("blog."):
        return "blog"
    if host.endswith(_NEWS):
        return "news"
    if host.endswith(_VENDOR):
        return "vendor"
    return "unknown"


def score_quality(*, domain_type: str, published: date | None, text_length: int,
                  injection_detected: bool, today: date | None = None) -> dict:
    today = today or date.today()
    base = DOMAIN_TYPE_BASE.get(domain_type, DOMAIN_TYPE_BASE["unknown"])
    components = {"domain_type": domain_type, "base": base}

    recency = 0.0
    if published is None:
        recency = -0.05
    else:
        age_days = (today - published).days
        if age_days <= 365:
            recency = 0.05
        elif age_days > 3 * 365:
            recency = -0.15
    components["recency"] = recency

    length = -0.10 if text_length < 800 else 0.0
    components["length"] = length

    score = max(0.0, min(1.0, base + recency + length))
    if injection_detected:
        score = min(score, 0.10)
        components["injection_cap"] = 0.10
    components["score"] = round(score, 2)
    return components
