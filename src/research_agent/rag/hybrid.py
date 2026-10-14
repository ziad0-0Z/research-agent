"""Hybrid retrieval: BM25 (exact terms, numbers) + embeddings (meaning), rank-fused, quality-weighted.

    BM25 rank ─┐
               ├─ Reciprocal Rank Fusion ─→ 0.8 × fused + 0.2 × source quality
    dense rank ┘

Why both: BM25 finds "$1.50 per hour" or "HNSW" exactly; embeddings find "labour" when the
query says "engineering time". RRF merges the two rankings without having to calibrate their
very different score scales.

When no embedding provider is available, retrieval falls back to BM25 alone and says so in
`method`. Deterministic for a fixed input (ties broken by document order).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from rank_bm25 import BM25Plus

from ..tools.corpus import tokenize
from .embeddings import EmbeddingChain, cosine

RRF_K = 10              # small k keeps a real gap between rank 1 and rank 5 in small collections
RELEVANCE_WEIGHT = 0.8  # the rest is source quality
MIN_DENSE_SIMILARITY = 0.35  # dense-only matches below this are noise, not candidates


@dataclass
class Doc:
    id: str
    text: str
    quality: float = 0.5
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Hit:
    doc: Doc
    score: float
    bm25_rank: int | None
    dense_rank: int | None
    similarity: float | None


@dataclass
class RetrievalResult:
    query: str
    hits: list[Hit]
    method: str          # "hybrid" | "bm25"
    embed_provider: str | None


def _ranks(scores: list[tuple[int, float]]) -> dict[int, int]:
    ordered = sorted(scores, key=lambda x: (-x[1], x[0]))
    return {i: r for r, (i, _) in enumerate(ordered, 1)}


def retrieve(query: str, docs: list[Doc], k: int = 5, embedder: EmbeddingChain | None = None) -> RetrievalResult:
    k = max(1, min(int(k), 10))
    if not docs or not query.strip():
        return RetrievalResult(query, [], "bm25", None)

    # Lexical ranking (only documents sharing at least one term).
    q_tokens = tokenize(query)
    tokenized = [tokenize(d.text) for d in docs]
    bm25 = BM25Plus(tokenized).get_scores(q_tokens) if q_tokens else [0.0] * len(docs)
    lexical = [(i, float(s)) for i, (s, toks) in enumerate(zip(bm25, tokenized)) if set(q_tokens) & set(toks)]
    bm25_rank = _ranks(lexical)

    # Dense ranking (all documents above a minimum similarity).
    sims: dict[int, float] = {}
    provider = None
    if embedder is not None:
        vectors, provider = embedder.embed([query] + [d.text for d in docs])
        if vectors is not None:
            qv = vectors[0]
            sims = {i: cosine(qv, v) for i, v in enumerate(vectors[1:])}
    dense = [(i, s) for i, s in sims.items() if s >= MIN_DENSE_SIMILARITY]
    dense_rank = _ranks(dense)

    if provider is None:  # BM25-only: score by normalized BM25 (RRF over one list flattens the ranking)
        top = max((s for _, s in lexical), default=0.0) or 1.0
        relevance = {i: s / top for i, s in lexical}
        method = "bm25"
    else:
        fused = {}
        for i in set(bm25_rank) | set(dense_rank):
            fused[i] = sum(1 / (RRF_K + r[i]) for r in (bm25_rank, dense_rank) if i in r)
        top = max(fused.values(), default=0.0) or 1.0
        relevance = {i: s / top for i, s in fused.items()}
        method = "hybrid"

    scored = sorted(
        ((round(RELEVANCE_WEIGHT * rel + (1 - RELEVANCE_WEIGHT) * docs[i].quality, 4), i)
         for i, rel in relevance.items()),
        key=lambda x: (-x[0], x[1]))
    hits = [Hit(doc=docs[i], score=score, bm25_rank=bm25_rank.get(i), dense_rank=dense_rank.get(i),
                similarity=round(sims[i], 4) if i in sims else None) for score, i in scored[:k]]
    return RetrievalResult(query, hits, method, provider)


_PARA = re.compile(r"\n\s*\n")
_SENT = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str, max_chars: int = 450) -> list[str]:
    """Paragraph chunks; long paragraphs are split on sentence boundaries."""
    chunks: list[str] = []
    for para in _PARA.split(text.strip()):
        para = " ".join(para.split())
        if not para:
            continue
        if len(para) <= max_chars:
            chunks.append(para)
            continue
        current = ""
        for sent in _SENT.split(para):
            if current and len(current) + len(sent) + 1 > max_chars:
                chunks.append(current)
                current = sent
            else:
                current = f"{current} {sent}".strip()
        if current:
            chunks.append(current)
    return chunks
