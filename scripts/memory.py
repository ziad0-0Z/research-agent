"""Inspect and manage long-term memory (Phase 6).

    python scripts/memory.py stats
    python scripts/memory.py list [--status active|superseded|unverifiable|retired] [--type pricing|market|risk|technical]
    python scripts/memory.py show MEM_ID
    python scripts/memory.py delete MEM_ID            # hard delete + tombstone: an old run cannot bring it back
    python scripts/memory.py purge-expired            # unverifiable, or stale for more than 2× its TTL (no tombstone:
                                                      # new research may learn it again)
    python scripts/memory.py reports                  # released reports kept for "related past research"
    python scripts/memory.py context                  # facts the user stated in questions (shown only)
    python scripts/memory.py dedupe                   # mark excerpt-duplicates of one fact as superseded (kept: the first)
    python scripts/memory.py match RUN_ID             # dry run: which remembered items that run's plan would
                                                      # reuse, under which subquestion (no LLM, nothing written)
"""
from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_agent.config import load_settings  # noqa: E402
from research_agent.memory import MemoryStore  # noqa: E402
from research_agent.models import utcnow  # noqa: E402
from research_agent.store import Store  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["stats", "list", "show", "delete", "purge-expired", "reports", "context", "match", "dedupe"])
    parser.add_argument("mem_id", nargs="?")
    parser.add_argument("--status")
    parser.add_argument("--type", dest="claim_type")
    args = parser.parse_args()
    settings = load_settings()
    store = Store(settings.db_path)
    ms = MemoryStore(store)
    now = utcnow()

    if args.action == "stats":
        s = ms.stats()
        print(f"Evidence by status: {s['evidence_by_status'] or 'none'}")
        print(f"Active by claim type: {s['active_by_type'] or 'none'}")
        print(f"Reports: {s['reports']} | page snapshots: {s['pages']} | stated facts: {s['context_facts']} | "
              f"deleted (tombstones): {s['deleted']}")
        fresh = sum(1 for i in ms.list(status="active") if i.is_fresh(now))
        print(f"Active items fresh now: {fresh} (the rest are re-verified before reuse)")
        return 0
    if args.action == "list":
        items = ms.list(status=args.status, claim_type=args.claim_type)
        for i in items:
            age = "fresh" if i.is_fresh(now) else "stale"
            print(f"{i.mem_id}  {i.status:12s} {i.claim_type:9s} {i.support:9s} {age:5s} "
                  f"until {i.expires_at:%Y-%m-%d}  {i.source_host:32s} {i.claim[:70]}")
        print(f"{len(items)} item(s)")
        return 0
    if args.action == "show":
        item = ms.get(args.mem_id or "")
        if item is None:
            print(f"Unknown memory item {args.mem_id}")
            return 1
        for k, v in item.as_dict().items():
            print(f"{k:18s} {v}")
        return 0
    if args.action == "delete":
        if not args.mem_id:
            parser.error("mem_id is required")
        ok = ms.delete(args.mem_id, now)
        print(f"Deleted {args.mem_id} (tombstoned: committing an old run again will not restore it)" if ok
              else f"Unknown memory item {args.mem_id}")
        return 0 if ok else 1
    if args.action == "purge-expired":
        n = 0
        for i in ms.list():
            ttl = timedelta(days=settings.memory_ttl(i.claim_type))
            if i.status == "unverifiable" or now > i.expires_at + ttl:
                ms.purge(i.mem_id)
                n += 1
        print(f"Purged {n} item(s)")
        return 0
    if args.action == "reports":
        for r in ms.reports():
            print(f"{r['run_id']}  {r['released_at'][:16]}  conf {r['confidence']}  {r['review_status']:8s} {r['title'][:70]}")
        return 0
    if args.action == "context":
        rows = ms.context()
        for r in rows:
            print(f"{r['created_at'][:16]}  {r['key']:10s} {r['value']:28s} said as: \"{r['stated_span']}\"  ({r['run_id']})")
        print(f"{len(rows)} fact(s). Shown only: never sent to an agent.")
        return 0
    if args.action == "dedupe":
        from research_agent.memory.supersede import merge_duplicate_excerpts
        merged = merge_duplicate_excerpts(ms)
        for m in merged:
            print(f"{m['duplicate']} → superseded by {m['kept']} (same fact)")
        print(f"{len(merged)} duplicate(s) merged. Nothing deleted: `list --status superseded` shows them.")
        return 0
    if args.action == "match":
        import json

        from research_agent.memory.matching import assign_items
        from research_agent.models import ResearchPlan
        row = store.get_run(args.mem_id or "")
        if row is None:
            print(f"Unknown run {args.mem_id}")
            return 1
        plan = ResearchPlan.model_validate(json.loads(row["state_json"])["plan"])
        active = {it.mem_id: it for it in ms.list(status="active")}
        found = assign_items(settings, list(active.values()), plan)
        method = next(iter(found.values())).method if found else "none"
        print(f"Plan of {args.mem_id}: {len(found)} remembered item(s) pass the gate | filed by: {method}\n")
        for sq in plan.subquestions:
            mine = [(active[mid], a) for mid, a in found.items() if a.subquestion_id == sq.id]
            print(f"{sq.id} [{sq.type}] {sq.question[:95]}")
            for it, a in sorted(mine, key=lambda x: -x[1].scores.get(sq.id, 0)):
                others = ", ".join(f"{k} {v}" for k, v in a.scores.items())
                print(f"    {it.mem_id}  word match {a.match}  [{others}]  {it.claim[:70]}")
            if not mine:
                print("    (nothing from memory: researched fresh)")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
