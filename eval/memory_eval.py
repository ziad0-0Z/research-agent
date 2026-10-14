"""Long-term memory evaluation (offline, no LLM): does memory show and reuse the right things?

    python eval/memory_eval.py

1. Related past reports: question similarity on golden pairs (data/golden/memory_pairs.json).
   precision / recall at MEMORY_SIMILARITY_THRESHOLD. These reports are shown only, never sent to an agent.
2. Reuse gate: memory is seeded with every paragraph of the LLM-hosting pages. For each golden
   subquestion we count which items pass the same gate the pipeline uses (memory.lookup.passes_gate).
   - false reuse: an unrelated subquestion (vector DBs, CRMs, office rent...) reuses anything   → target 0
   - precision:   share of reused items that come from a page allowed for that subquestion
   - recall:      related subquestions that reuse at least one allowed item
Misses are listed, never hidden. Results: eval/results/memory.json. The live time/LLM-call savings are
measured separately (HANDOFF live checklist: same question twice, memory on vs off).
"""
from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_agent.config import load_settings  # noqa: E402
from research_agent.memory.matching import TermIndex, passes_gate, question_similarity  # noqa: E402
from research_agent.memory.store import MemoryItem  # noqa: E402
from research_agent.models import utcnow  # noqa: E402
from research_agent.rag import chunk_text  # noqa: E402
from research_agent.tools.corpus import load_corpus  # noqa: E402

def seed(corpus_dir: Path, pages: list[str]) -> list[MemoryItem]:
    now = utcnow()
    items = []
    for page in load_corpus(str(corpus_dir)).pages:
        if page.file not in pages:
            continue
        for i, chunk in enumerate(chunk_text(page.text)):
            items.append(MemoryItem(
                mem_id=f"{page.file}#{i}", content_hash=f"{page.file}#{i}", claim=chunk[:400], quote=chunk[:600],
                claim_type="market", specialist="market", subquestion_text="", source_url=page.url,
                source_host="", source_title=page.title, source_domain_type=page.domain_type,
                source_published=None, source_quality=0.7, origin="offline", page_hash="", support="unjudged",
                first_run_id="seed", last_run_id="seed", created_at=now, last_verified_at=now,
                expires_at=now + timedelta(days=90)))
    return items


def main() -> int:
    settings = load_settings()
    golden = json.loads((ROOT / "data" / "golden" / "memory_pairs.json").read_text(encoding="utf-8"))
    out: dict = {}

    # 1. related past reports
    th = settings.memory_similarity_threshold
    rows = []
    for q in golden["questions"]:
        sim = question_similarity(q["a"], q["b"])
        rows.append({**q, "similarity": sim, "shown": sim >= th})
    tp = sum(r["shown"] and r["related"] for r in rows)
    fp = sum(r["shown"] and not r["related"] for r in rows)
    fn = sum(not r["shown"] and r["related"] for r in rows)
    out["related_reports"] = {"threshold": th, "precision": round(tp / (tp + fp), 3) if tp + fp else None,
                              "recall": round(tp / (tp + fn), 3) if tp + fn else None,
                              "false_shown": [r for r in rows if r["shown"] and not r["related"]],
                              "missed": [r for r in rows if not r["shown"] and r["related"]], "rows": rows}

    # 2. reuse gate
    items = seed(settings.corpus_dir, golden["seed_pages"])
    index = TermIndex(items)
    reuse_rows = []
    for sq in golden["reuse"]:
        passed = [it for it in items if passes_gate(sq["subquestion"], it, settings.memory_min_match, index)[0]]
        files = [it.mem_id.split("#")[0] for it in passed]
        good = [f for f in files if f in sq["allowed"]]
        reuse_rows.append({"id": sq["id"], "subquestion": sq["subquestion"], "related": sq["related"],
                           "reused": len(passed), "from_allowed": len(good), "pages": sorted(set(files))})
    related = [r for r in reuse_rows if r["related"]]
    unrelated = [r for r in reuse_rows if not r["related"]]
    total = sum(r["reused"] for r in related)
    out["reuse_gate"] = {
        "min_match": settings.memory_min_match, "seed_items": len(items),
        "false_reuse_subquestions": [r["id"] for r in unrelated if r["reused"]],
        "false_reuse_items": sum(r["reused"] for r in unrelated),
        "precision": round(sum(r["from_allowed"] for r in related) / total, 3) if total else None,
        "recall": round(sum(1 for r in related if r["from_allowed"]) / len(related), 3),
        "rows": reuse_rows}

    rr, rg = out["related_reports"], out["reuse_gate"]
    print(f"Related past reports (threshold {th}): precision {rr['precision']}, recall {rr['recall']}")
    for r in rr["false_shown"]:
        print(f"  shown but unrelated ({r['similarity']}): {r['b']}")
    for r in rr["missed"]:
        print(f"  related but not shown ({r['similarity']}): {r['b']}")
    print(f"\nReuse gate (min match {rg['min_match']}, {rg['seed_items']} seeded items):")
    print(f"  false reuse: {rg['false_reuse_items']} item(s) on {len(rg['false_reuse_subquestions'])}/"
          f"{len(unrelated)} unrelated subquestions {rg['false_reuse_subquestions'] or ''}")
    print(f"  precision {rg['precision']} (reused items from an allowed page), "
          f"recall {rg['recall']} (related subquestions with ≥1 allowed item)")
    for r in reuse_rows:
        flag = "" if (r["related"] and r["from_allowed"]) or (not r["related"] and not r["reused"]) else "  ← miss"
        print(f"  {r['id']} reused {r['reused']:2d} (allowed {r['from_allowed']:2d}) {r['subquestion'][:62]}{flag}")
    res = ROOT / "eval" / "results"
    res.mkdir(parents=True, exist_ok=True)
    (res / "memory.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\nSaved eval/results/memory.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
