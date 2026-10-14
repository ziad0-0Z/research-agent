"""Retrieval recall@5 over the offline corpus: BM25-only vs hybrid (BM25 + embeddings).

    python eval/retrieval_eval.py            # BM25, then hybrid if an embedding provider is reachable
    python eval/retrieval_eval.py --bm25     # BM25 only

Index: paragraph chunks of every corpus page (data/corpus). Test set: data/golden/retrieval.json.
recall@5 = relevant pages found in the top 5 chunks / all relevant pages (summed over queries).
Misses are listed, never hidden. Results are saved to eval/results/retrieval.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_agent.config import load_settings  # noqa: E402
from research_agent.rag import Doc, chunk_text, get_embedder, retrieve  # noqa: E402
from research_agent.tools.corpus import load_corpus  # noqa: E402
from research_agent.tools.quality import score_quality  # noqa: E402

K = 5


def build_docs(corpus_dir: Path) -> list[Doc]:
    docs = []
    for page in load_corpus(str(corpus_dir)).pages:
        quality = score_quality(domain_type=page.domain_type, published=page.published,
                                text_length=len(page.text), injection_detected=False)["score"]
        for i, chunk in enumerate(chunk_text(page.text)):
            docs.append(Doc(id=f"{page.file}#{i}", text=f"{page.title}. {chunk}", quality=quality,
                            meta={"file": page.file}))
    return docs


def evaluate(docs: list[Doc], items: list[dict], embedder) -> dict:
    found_total = relevant_total = 0
    rows = []
    method = provider = None
    for item in items:
        res = retrieve(item["query"], docs, k=K, embedder=embedder)
        method, provider = res.method, res.embed_provider or provider
        top_files = list(dict.fromkeys(h.doc.meta["file"] for h in res.hits))
        found = [f for f in item["relevant"] if f in top_files]
        found_total += len(found)
        relevant_total += len(item["relevant"])
        rows.append({"id": item["id"], "query": item["query"], "paraphrase": item["paraphrase"],
                     "found": found, "missed": [f for f in item["relevant"] if f not in found], "top": top_files})
    para = [r for r in rows if r["paraphrase"]]
    lit = [r for r in rows if not r["paraphrase"]]

    def recall(rs):
        rel = sum(len(r["found"]) + len(r["missed"]) for r in rs)
        return round(sum(len(r["found"]) for r in rs) / rel, 3) if rel else None

    return {"method": method, "embed_provider": provider, "k": K, "chunks": len(docs),
            "recall_at_5": round(found_total / relevant_total, 3), "found": found_total, "relevant": relevant_total,
            "recall_paraphrase": recall(para), "recall_literal": recall(lit), "rows": rows}


def report(r: dict) -> None:
    label = r["method"] + (f" ({r['embed_provider']})" if r["embed_provider"] else "")
    print(f"\n{label}: recall@{r['k']} = {r['found']}/{r['relevant']} ({r['recall_at_5']:.0%})"
          f" | literal {r['recall_literal']:.0%} | paraphrased {r['recall_paraphrase']:.0%}")
    for row in r["rows"]:
        if row["missed"]:
            print(f"  MISS {row['id']}{' (paraphrase)' if row['paraphrase'] else ''}: {row['query'][:70]!r}")
            print(f"       expected {', '.join(row['missed'])}; got {', '.join(row['top'][:3])}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bm25", action="store_true", help="skip the hybrid run")
    args = parser.parse_args()
    settings = load_settings()
    items = json.loads((ROOT / "data" / "golden" / "retrieval.json").read_text(encoding="utf-8"))["items"]
    docs = build_docs(settings.corpus_dir)
    print(f"{len(items)} queries over {len(docs)} chunks from {settings.corpus_dir.name}/")

    results = [evaluate(docs, items, None)]
    report(results[0])
    if not args.bm25 and settings.embed_mode != "off":
        hybrid = evaluate(docs, items, get_embedder(settings))
        if hybrid["method"] == "hybrid":
            results.append(hybrid)
            report(hybrid)
        else:
            events = get_embedder(settings).events
            print("\nHybrid skipped: no embedding provider reachable"
                  + (f" — {events[-1]['provider']}: {events[-1]['error']}" if events else ""))
            print("Start Ollama and run `ollama pull bge-m3`, or set GEMINI_API_KEY.")

    out = ROOT / "eval" / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "retrieval.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {(out / 'retrieval.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
