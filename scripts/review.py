"""Human review queue from the command line (the Gradio Review tab in Phase 8 calls the same functions).

    python scripts/review.py list
    python scripts/review.py show RUN_ID
    python scripts/review.py approve RUN_ID [--note "..."]        # RUN_ID may be "latest" (newest pending review)
    python scripts/review.py reject  RUN_ID --note "why"
    python scripts/review.py changes RUN_ID --note "what to change"     # Writer revises → Critic → back to the queue

Deciding records the human decision, then resumes the run from the database. Works in a new
process (or the next day): nothing from the original run's memory is needed.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_agent.config import ConfigError, load_settings  # noqa: E402
from research_agent.store import Store  # noqa: E402
from research_agent.tracing import configure_tracing  # noqa: E402
from research_agent.workflows.resume import ReviewError, decide_review, list_pending, resume_run  # noqa: E402

DECISIONS = {"approve": "approved", "reject": "rejected", "changes": "changes_requested"}


def show(store: Store, run_id: str) -> int:
    row = store.get_run(run_id)
    if row is None:
        print(f"Unknown run {run_id}")
        return 1
    state = json.loads(row["state_json"] or "{}")
    review = store.get_review(run_id)
    r = state.get("report") or {}
    print(f"Run {run_id} — status {row['status']}\nQuestion: {row['question']}")
    if review:
        print(f"Review: {review.status}" + (f" — note: {review.reviewer_note}" if review.reviewer_note else ""))
        for reason in review.reasons:
            print(f"  - {reason}")
    conf = state.get("confidence") or {}
    print(f"Confidence: {conf.get('score')} — {conf.get('explanation', '')}")
    print(f"Critic: {state.get('critic_verdict')} after {state.get('critic_rounds')} round(s)\n")
    if r:
        print(f"{r['title']}\n{'-' * len(r['title'])}\n{r['executive_summary']}\n")
        for i, f in enumerate(r["findings"], 1):
            print(f"{i}. {f['statement']} {f['evidence_ids']} ({f['strength']})")
        print(f"\nRecommendation ({r['impact']} impact): {r['recommendation']}")
        for lim in r.get("limitations", []):
            print(f"  limitation: {lim}")
    for h in state.get("review_history", []):
        print(f"History: {h['decided_at'][:16]} {h['decision']} {h['note']}")
    return 0


def on_event(stage: str, data: dict) -> None:
    if stage == "resumed":
        print(f"▶️  Resuming after human decision: {data['decision']}")
    elif stage == "revised":
        print(f"🤖 Writer revised the report with the reviewer's note ({data['findings']} findings)")
    elif stage == "critic_round":
        print(f"   Critic round {data['round']}: {data['verdict']} ({data['blocking']} blocking)")
    elif stage == "output_checked":
        print("🛡️ Output guardrail: " + ("passed" if data["passed"] else "REJECTED"))
        for i in data["issues"]:
            print(f"   [{i['check']}] {i['message'][:150]}")
    elif stage == "paused":
        print("⏸️  Back in the review queue: a human must approve the revised report")
    elif stage == "completed":
        print(f"📄 Final report released ({data['chars']} chars)")
    elif stage == "rejected_by_human":
        print("⛔ Rejected by the reviewer: nothing released")


async def decide_and_resume(store: Store, settings, run_id: str, action: str, note: str) -> int:
    try:
        decide_review(store, run_id, DECISIONS[action], note)  # type: ignore[arg-type]
        if action == "changes":
            settings.require_llm_key()  # only the revision needs the LLM
        result = await resume_run(run_id, settings=settings, store=store, on_event=on_event)
    except (ReviewError, ConfigError) as exc:
        print(f"❌ {exc}")
        return 1
    if result.final_markdown:
        out = ROOT / "samples" / f"{run_id}.md"
        out.parent.mkdir(exist_ok=True)
        out.write_text(result.final_markdown, encoding="utf-8")
        print(f"✅ COMPLETED — final report saved: {out.relative_to(ROOT)}")
    else:
        print(f"Status: {result.status}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["list", "show", *DECISIONS])
    parser.add_argument("run_id", nargs="?")
    parser.add_argument("--note", default="")
    args = parser.parse_args()
    settings = load_settings()
    store = Store(settings.db_path)
    if args.action == "list":
        queue = list_pending(store)
        if not queue:
            print("Review queue is empty.")
        for q in queue:
            print(f"{q['run_id']}  conf {q['confidence']}  impact {q['impact']}  round {q['rounds']}  {q['question'][:70]}")
            for r in q["reasons"]:
                print(f"    - {r}")
        return 0
    if not args.run_id:
        parser.error("run_id is required (or 'latest')")
    if args.run_id == "latest":
        queue = list_pending(store)
        if not queue:
            print("Review queue is empty: nothing to decide.")
            return 1
        args.run_id = queue[-1]["run_id"]  # pending reviews are ordered oldest first
        print(f"latest pending review: {args.run_id}")
    if args.action == "show":
        return show(store, args.run_id)
    configure_tracing(settings)
    return asyncio.run(decide_and_resume(store, settings, args.run_id, args.action, args.note))


if __name__ == "__main__":
    raise SystemExit(main())
