"""Hybrid retrieval (RAG bonus): fusion, fallback chain, cache, cooldown. No real embedding calls."""
from research_agent.rag import Doc, chunk_text, retrieve, set_embedder
from research_agent.rag.embeddings import EmbeddingChain, EmbeddingError, VectorCache

# Toy "semantic" space: each concept is one dimension; words map to concepts.
CONCEPTS = {
    "labour": 0, "labor": 0, "engineer": 0, "engineers": 0, "engineering": 0, "staff": 0, "people": 0, "maintain": 0,
    "price": 1, "prices": 1, "cost": 1, "costs": 1, "spend": 1, "token": 1, "tokens": 1,
    "security": 2, "patch": 2, "patched": 2, "attack": 2, "breach": 2,
}


class FakeProvider:
    def __init__(self, label="fake/concepts", fail=False):
        self.label = label
        self.fail = fail
        self.calls = 0
        self.texts_embedded = 0

    def embed(self, texts):
        self.calls += 1
        if self.fail:
            raise EmbeddingError("provider down")
        self.texts_embedded += len(texts)
        out = []
        for t in texts:
            v = [0.0, 0.0, 0.0, 0.05]
            for w in t.lower().replace(".", " ").replace("?", " ").split():
                if w in CONCEPTS:
                    v[CONCEPTS[w]] += 1.0
            out.append(v)
        return out


def chain(*providers):
    return EmbeddingChain(providers=list(providers), cache=VectorCache(":memory:"))


DOCS = [
    Doc("ev1", "Keeping a self-hosted model reliable required half of a full-time engineer.", quality=0.9),
    Doc("ev2", "Output tokens typically cost three to five times more than input tokens.", quality=0.7),
    Doc("ev3", "Inference servers must be patched regularly against attack.", quality=0.8),
]
QUERY = "How much staff labour does running your own model need?"  # no word shared with ev1 except "model"


def test_hybrid_finds_meaning_that_bm25_misses():
    bm25 = retrieve("labour needed", DOCS, k=3, embedder=None)
    assert bm25.method == "bm25" and [h.doc.id for h in bm25.hits] == []   # "labour" never appears literally

    hybrid = retrieve("labour needed", DOCS, k=3, embedder=chain(FakeProvider()))
    assert hybrid.method == "hybrid" and hybrid.embed_provider == "fake/concepts"
    assert hybrid.hits[0].doc.id == "ev1" and hybrid.hits[0].dense_rank == 1 and hybrid.hits[0].bm25_rank is None


def test_fusion_rewards_items_found_by_both_rankings():
    hybrid = retrieve("token cost of output", DOCS, k=3, embedder=chain(FakeProvider()))
    top = hybrid.hits[0]
    assert top.doc.id == "ev2" and top.bm25_rank == 1 and top.dense_rank == 1


def test_chain_falls_back_to_next_provider_then_to_bm25():
    down, backup = FakeProvider("ollama/bge-m3", fail=True), FakeProvider("gemini/embedding")
    c = chain(down, backup)
    r = retrieve(QUERY, DOCS, embedder=c)
    assert r.method == "hybrid" and r.embed_provider == "gemini/embedding"
    assert c.events and c.events[0]["provider"] == "ollama/bge-m3"

    everything_down = chain(FakeProvider("a", fail=True), FakeProvider("b", fail=True))
    r = retrieve("output tokens cost", DOCS, embedder=everything_down)
    assert r.method == "bm25" and r.hits[0].doc.id == "ev2"   # still answers, lexically


def test_failed_provider_is_skipped_during_cooldown():
    down = FakeProvider("ollama/bge-m3", fail=True)
    c = chain(down, FakeProvider("gemini/embedding"))
    for _ in range(3):
        retrieve(QUERY, DOCS, embedder=c)
    assert down.calls == 1   # tried once, then skipped for the cooldown period


def test_vectors_are_cached():
    p = FakeProvider()
    c = chain(p)
    retrieve(QUERY, DOCS, embedder=c)
    first = p.texts_embedded
    retrieve(QUERY, DOCS, embedder=c)
    assert first == len(DOCS) + 1 and p.texts_embedded == first   # second query: all from cache


def test_quality_breaks_near_ties():
    docs = [Doc("low", "token cost", quality=0.1), Doc("high", "token cost", quality=0.9)]
    assert retrieve("token cost", docs, embedder=None).hits[0].doc.id == "high"


def test_retrieve_evidence_tool_reports_method(monkeypatch, tmp_path):
    import asyncio

    from research_agent import config as cfg
    from research_agent.context import AppContext
    from research_agent.store import Store
    from research_agent.tools import evidence, web
    monkeypatch.setenv("SEARCH_MODE", "offline")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("EMBED_MODE", "auto")
    ctx = AppContext(settings=cfg.load_settings(), store=Store(":memory:"), question="Cloud APIs or self-hosting?")
    asyncio.run(web.scrape_page(ctx, "https://applied-systems-institute.example/reports/llm-tco-small-teams"))
    evidence.save_evidence(ctx, source_id="src1", subquestion_id="sq1", claim="Self-hosting needs engineer time.",
                           quote="keeping a self-hosted model reliable in production required between 0.25 and 0.5 of "
                                 "a full-time engineer on an ongoing basis")
    set_embedder(chain(FakeProvider()))
    try:
        out = evidence.retrieve_evidence(ctx, "staff labour", k=3)
    finally:
        set_embedder(None)
    assert out["method"] == "hybrid" and out["results"][0]["evidence_id"] == "ev1"


def test_chunk_text_splits_long_paragraphs_on_sentences():
    text = "Short paragraph.\n\n" + " ".join(f"Sentence number {i} is here." for i in range(40))
    chunks = chunk_text(text, max_chars=200)
    assert chunks[0] == "Short paragraph." and all(len(c) <= 200 for c in chunks) and len(chunks) > 3
