"""Embedding chain for hybrid retrieval: Ollama bge-m3 → Gemini embeddings → none (BM25-only).

Same idea as the LLM fallback chain in llm.py:
- Ollama (local) is primary: no quota, no network, safe for the demo.
- Gemini embeddings API is the fallback when Ollama is not running.
- If both fail, `embed()` returns None and retrieval runs on BM25 alone. Retrieval never fails
  because embeddings are missing; it only reports which method it used.

A provider that fails is skipped for a cooldown period, so a stopped Ollama costs one quick
connection error per minute, not one per query.

Vectors are cached on disk (data/cache/embeddings.db), keyed by provider/model and text hash.
That makes repeated runs fast, and it keeps record/replay runs stable: a replayed run reuses the
exact vectors of the recorded run, so retrieval order (and therefore the Critic prompt) matches.
"""
from __future__ import annotations

import hashlib
import math
import sqlite3
import threading
import time
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import httpx

from ..config import PROVIDER_BASE_URLS, Settings, load_settings

BATCH_SIZE = 32


class EmbeddingError(RuntimeError):
    pass


class EmbeddingProvider(Protocol):
    label: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


@dataclass
class OllamaEmbedder:
    base_url: str
    model: str
    timeout: float = 30.0

    @property
    def label(self) -> str:
        return f"ollama/{self.model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            r = httpx.post(f"{self.base_url.rstrip('/')}/api/embed",
                           json={"model": self.model, "input": texts}, timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"Ollama not reachable at {self.base_url}: {type(exc).__name__}") from exc
        if r.status_code == 404:
            raise EmbeddingError(f"Ollama model '{self.model}' not found: run `ollama pull {self.model}`")
        if r.status_code != 200:
            raise EmbeddingError(f"Ollama returned {r.status_code}: {r.text[:120]}")
        vectors = r.json().get("embeddings") or []
        if len(vectors) != len(texts):
            raise EmbeddingError(f"Ollama returned {len(vectors)} vectors for {len(texts)} texts")
        return vectors


@dataclass
class GeminiEmbedder:
    api_key: str
    model: str
    timeout: float = 30.0

    @property
    def label(self) -> str:
        return f"gemini/{self.model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        url = PROVIDER_BASE_URLS["gemini"] + "embeddings"
        try:
            r = httpx.post(url, headers={"Authorization": f"Bearer {self.api_key}"},
                           json={"model": self.model, "input": texts}, timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"Gemini embeddings not reachable: {type(exc).__name__}") from exc
        if r.status_code != 200:
            raise EmbeddingError(f"Gemini embeddings returned {r.status_code}: {r.text[:120]}")
        data = sorted(r.json().get("data") or [], key=lambda d: d.get("index", 0))
        vectors = [d["embedding"] for d in data]
        if len(vectors) != len(texts):
            raise EmbeddingError(f"Gemini returned {len(vectors)} vectors for {len(texts)} texts")
        return vectors


class VectorCache:
    """Tiny SQLite cache: (model label, text hash) → float32 vector."""

    def __init__(self, path: str | Path = ":memory:"):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("CREATE TABLE IF NOT EXISTS vectors (model TEXT NOT NULL, hash TEXT NOT NULL, "
                           "vector BLOB NOT NULL, PRIMARY KEY (model, hash))")
        self._conn.commit()

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]

    def get_many(self, model: str, texts: list[str]) -> dict[int, list[float]]:
        keys = [self.key(t) for t in texts]
        found: dict[int, list[float]] = {}
        with self._lock:
            for i, k in enumerate(keys):
                row = self._conn.execute("SELECT vector FROM vectors WHERE model=? AND hash=?", (model, k)).fetchone()
                if row:
                    found[i] = list(array("f", row[0]))
        return found

    def put_many(self, model: str, texts: list[str], vectors: list[list[float]]) -> None:
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO vectors VALUES (?,?,?)",
                [(model, self.key(t), array("f", v).tobytes()) for t, v in zip(texts, vectors)])
            self._conn.commit()


@dataclass
class EmbeddingChain:
    """Tries providers in order; returns (vectors, provider label) or (None, None)."""

    providers: list[EmbeddingProvider]
    cache: VectorCache
    cooldown_seconds: float = 60.0
    _cooling_until: dict[str, float] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)

    def embed(self, texts: list[str]) -> tuple[list[list[float]] | None, str | None]:
        if not texts or not self.providers:
            return None, None
        now = time.monotonic()
        for provider in self.providers:
            if self._cooling_until.get(provider.label, 0) > now:
                continue
            try:
                return self._embed_with(provider, texts), provider.label
            except EmbeddingError as exc:
                self._cooling_until[provider.label] = time.monotonic() + self.cooldown_seconds
                self.events.append({"time": time.time(), "provider": provider.label, "error": str(exc)[:160]})
        return None, None

    def _embed_with(self, provider: EmbeddingProvider, texts: list[str]) -> list[list[float]]:
        cached = self.cache.get_many(provider.label, texts)
        missing = [i for i in range(len(texts)) if i not in cached]
        for start in range(0, len(missing), BATCH_SIZE):
            idx = missing[start:start + BATCH_SIZE]
            batch = [texts[i] for i in idx]
            vectors = provider.embed(batch)
            self.cache.put_many(provider.label, batch, vectors)
            cached.update(zip(idx, vectors))
        return [cached[i] for i in range(len(texts))]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def build_chain(settings: Settings) -> EmbeddingChain:
    providers: list[EmbeddingProvider] = []
    mode = settings.embed_mode
    if mode in ("auto", "ollama"):
        providers.append(OllamaEmbedder(settings.ollama_base_url, settings.embed_model, settings.embed_timeout_seconds))
    if mode in ("auto", "gemini") and settings.gemini_api_key:
        providers.append(GeminiEmbedder(settings.gemini_api_key, settings.gemini_embed_model,
                                        settings.embed_timeout_seconds))
    cache = VectorCache(settings.embed_cache_path if providers else ":memory:")
    return EmbeddingChain(providers=providers, cache=cache)


_chain: EmbeddingChain | None = None


def get_embedder(settings: Settings | None = None) -> EmbeddingChain:
    """Shared embedding chain (like get_model): one cache and one cooldown state per process."""
    global _chain
    if _chain is None:
        _chain = build_chain(settings or load_settings())
    return _chain


def set_embedder(chain: EmbeddingChain | None) -> None:
    """Replace the shared chain (tests use a fake provider; None resets)."""
    global _chain
    _chain = chain
