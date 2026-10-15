"""Matching remembered evidence to a new plan: relevance gate, then filing. Pure code, no LLM.

    gate     IDF-weighted share of a subquestion's words found in the item (MEMORY_MIN_MATCH, at least two
             words). Rare words count, common ones ("cost", "million") barely do - measured in
             eval/memory_eval.py: 0/8 unrelated subquestions pass.
    filing   each item that passes goes to ONE subquestion: by meaning (embeddings) when available, else by
             the word match with a small same-specialist bonus.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..config import Settings
from ..models import ResearchPlan
from ..tools.corpus import tokenize
from .store import MemoryItem

MIN_MATCHED_TERMS = 2
TYPE_BONUS = 0.2


# Words that say who is asking or how, not what the question is about.
QUESTION_WORDS = {
    "our", "we", "us", "i", "my", "you", "your", "which", "between", "choose", "choosing", "decide", "deciding",
    "better", "best", "use", "using", "can", "could", "would", "will", "than", "there", "their", "them", "they",
    "when", "where", "why", "who", "whether", "about", "into", "need", "want", "go", "get", "ourselves",
    "team", "person", "people", "company", "small",
}


def _stem(t: str) -> str:
    """Crude suffix folding so "hosted", "hosting" and "hosts" all meet "host"."""
    for suffix in ("ing", "ed", "es", "s"):
        if len(t) > len(suffix) + 2 and t.endswith(suffix) and not t.endswith("ss"):
            return t[: -len(suffix)]
    return t


def content_terms(text: str) -> set[str]:
    return {_stem(t) for t in tokenize(text) if t not in QUESTION_WORDS and not t.isdigit()}


def question_similarity(a: str, b: str) -> float:
    ta, tb = content_terms(a), content_terms(b)
    return round(len(ta & tb) / len(ta | tb), 3) if ta and tb else 0.0


class TermIndex:
    """How rare each word is across what memory holds (IDF). Words found in many items ("cost", "time",
    "million") count little, so a subquestion cannot reuse memory on generic words alone."""

    def __init__(self, items: list[MemoryItem]):
        self.terms = {it.mem_id: _item_terms(it) for it in items}
        self.n = len(items)
        self.df: dict[str, int] = {}
        for ts in self.terms.values():
            for t in ts:
                self.df[t] = self.df.get(t, 0) + 1

    def idf(self, term: str) -> float:
        return math.log((self.n + 1) / (self.df.get(term, 0) + 1)) + 1.0


def _item_terms(item: MemoryItem) -> set[str]:
    return content_terms(f"{item.claim} {item.quote} {item.subquestion_text}")


def match(query: str, item: MemoryItem, index: TermIndex | None = None) -> tuple[float, int]:
    """IDF-weighted share of the query's content words found in the memory item."""
    q = content_terms(query)
    if not q:
        return 0.0, 0
    have = index.terms.get(item.mem_id) if index else None
    hit = q & (have if have is not None else _item_terms(item))
    weight = index.idf if index else (lambda _t: 1.0)
    return round(sum(weight(t) for t in hit) / sum(weight(t) for t in q), 3), len(hit)


def passes_gate(query: str, item: MemoryItem, min_match: float, index: TermIndex | None = None) -> tuple[bool, float]:
    """The reuse gate (also measured by eval/memory_eval.py): enough of the query's weighted words, at least two."""
    m, hits = match(query, item, index)
    return (m >= min_match and hits >= MIN_MATCHED_TERMS), m


# Decision facets every question shares ("compare cost, privacy and lock-in"). They say WHICH aspect is
# asked about, not WHAT the decision is about, so they can't make a remembered item relevant on their own:
# bank-regulation evidence about "customer data privacy" must not enter an e-commerce recommender question.
FACET_WORDS = {_stem(w) for w in (
    "cost", "costs", "price", "pricing", "total", "ownership", "tco", "quality", "latency", "performance",
    "scalability", "scale", "scaling", "security", "privacy", "compliance", "risk", "risks", "vendor", "lock",
    "lockin", "data", "time", "launch", "implementation", "implement", "build", "buy", "own", "managed",
    "operational", "complexity", "capabilities", "budget", "limited", "engineering", "resources", "compare",
    "trade", "offs", "provide", "evidence", "recommendation", "recommendations", "condition", "conditions",
    "alternative", "preferable", "month", "months", "year", "next", "growth", "grow", "volume", "startup",
    "should", "or", "adopt", "dedicated", "explain", "identify", "assume", "expects", "significantly",
    "its", "such", "require", "required", "requires", "based", "system", "systems", "using", "used", "make",
    "making", "whether", "option", "options", "versus", "vs", "api", "apis",
)}


def topic_terms(main_question: str) -> set[str]:
    """What the decision is about: the question's content words minus the generic decision facets."""
    return {t for t in content_terms(main_question) - FACET_WORDS if len(t) >= 2}


def shares_topic(main_question: str, item: MemoryItem) -> bool:
    """A remembered item must mention at least one topic word of the new question (not only a facet)."""
    topic = topic_terms(main_question)
    return not topic or bool(topic & _item_terms(item))


@dataclass
class Assignment:
    subquestion_id: str
    match: float              # best gate score (IDF word match) over the subquestions
    method: str               # dense (embeddings) | lexical
    scores: dict[str, float]  # per subquestion: cosine when dense, word match when lexical


def assign_items(settings: Settings, items: list[MemoryItem], plan: ResearchPlan) -> dict[str, Assignment]:
    """Which remembered items are relevant, and to which ONE subquestion each belongs.

    Gate: the IDF word match must pass for at least one subquestion (this keeps unrelated memory out).
    Assignment: by meaning when embeddings are available (cosine of claim+quote vs each subquestion), else
    by the word match with a small same-specialist bonus. Words alone mis-file items: "engineering labour
    costs outweigh compute savings" shares "cost" with a hardware-cost subquestion and loses to it, so its
    evidence would count for the wrong subquestion. A quote can be saved only once per run, so the choice
    matters for coverage and confidence.
    """
    from ..rag import get_embedder
    from ..rag.embeddings import cosine

    index = TermIndex(items)
    gate: dict[str, dict[str, float]] = {}
    for it in items:
        if not shares_topic(plan.main_question, it):
            continue  # matches a facet ("privacy", "cost") of a different decision
        for sq in plan.subquestions:
            ok, m = passes_gate(sq.question, it, settings.memory_min_match, index)
            if ok:
                gate.setdefault(it.mem_id, {})[sq.id] = m
    if not gate:
        return {}
    passed = [it for it in items if it.mem_id in gate]
    vectors = None
    if settings.embed_mode != "off":
        texts = [sq.question for sq in plan.subquestions] + [f"{it.claim} {it.quote}" for it in passed]
        vectors, _provider = get_embedder(settings).embed(texts)
    out: dict[str, Assignment] = {}
    n_sq = len(plan.subquestions)
    for k, it in enumerate(passed):
        best_m = max(gate[it.mem_id].values())
        if vectors is not None:
            sims = {sq.id: round(cosine(vectors[n_sq + k], vectors[j]), 4) for j, sq in enumerate(plan.subquestions)}
            out[it.mem_id] = Assignment(max(sims, key=lambda sid: sims[sid]), best_m, "dense", sims)
        else:
            ranks = {sq.id: gate[it.mem_id][sq.id] + (TYPE_BONUS if it.specialist == sq.type else 0.0)
                     for sq in plan.subquestions if sq.id in gate[it.mem_id]}
            out[it.mem_id] = Assignment(max(ranks, key=lambda sid: ranks[sid]), best_m, "lexical",
                                        dict(gate[it.mem_id]))
    return out
