"""Offline corpus: synthetic pages in data/corpus/*.md with front matter."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path

from rank_bm25 import BM25Plus

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "is", "are", "vs", "with",
    "what", "how", "should", "do", "does", "be", "it", "at", "by", "as", "from", "that", "this",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


@dataclass(frozen=True)
class CorpusPage:
    file: str
    title: str
    url: str
    publisher: str
    domain_type: str
    published: date | None
    tags: tuple[str, ...]
    text: str = field(repr=False)


def _parse(path: Path) -> CorpusPage:
    raw = path.read_text(encoding="utf-8")
    if not raw.startswith("---"):
        raise ValueError(f"{path.name}: missing front matter")
    _, header, body = raw.split("---", 2)
    meta: dict[str, str] = {}
    for line in header.strip().splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip().strip('"')
    published = date.fromisoformat(meta["published"]) if meta.get("published") else None
    return CorpusPage(
        file=path.name,
        title=meta["title"],
        url=meta["url"],
        publisher=meta.get("publisher", ""),
        domain_type=meta.get("domain_type", "unknown"),
        published=published,
        tags=tuple(t.strip() for t in meta.get("tags", "").split(",") if t.strip()),
        text=body.strip(),
    )


class Corpus:
    def __init__(self, directory: Path):
        files = sorted(p for p in directory.glob("*.md") if p.name.lower() != "readme.md")
        self.pages = [_parse(p) for p in files]
        self._by_url = {p.url: p for p in self.pages}
        self._docs = [tokenize(f"{p.title} {' '.join(p.tags)} {p.text}") for p in self.pages]
        self._bm25 = BM25Plus(self._docs) if self._docs else None

    def get(self, url: str) -> CorpusPage | None:
        return self._by_url.get(url.strip())

    def search(self, query: str, k: int = 5) -> list[tuple[CorpusPage, float]]:
        if not self._bm25:
            return []
        q = tokenize(query)
        scores = self._bm25.get_scores(q)
        matching = [(p, float(s)) for p, s, d in zip(self.pages, scores, self._docs) if set(q) & set(d)]
        return sorted(matching, key=lambda x: x[1], reverse=True)[:k]


@lru_cache(maxsize=4)
def load_corpus(directory: str) -> Corpus:
    return Corpus(Path(directory))
