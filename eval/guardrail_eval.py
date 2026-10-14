"""Input guardrail catch rate and false-positive rate on data/golden/guardrails.json.

    python eval/guardrail_eval.py           # regex layer only (offline, no LLM)
    python eval/guardrail_eval.py --live    # regex + Guard Agent (the real input guardrail)

Catch rate = attacks blocked / attacks. False-positive rate = normal questions blocked / normal questions.
Results are written to eval/results/guardrails.json.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents import Runner  # noqa: E402

from research_agent.guardrails.input import check_input, prefilter  # noqa: E402


async def evaluate(live: bool) -> dict:
    golden = json.loads((ROOT / "data" / "golden" / "guardrails.json").read_text(encoding="utf-8"))
    guard = None
    if live:
        from research_agent.agents.guard import guard as make_guard
        from research_agent.config import load_settings
        load_settings().require_llm_key()
        guard = make_guard()

    async def classify(text: str):
        if not live:
            pre = prefilter(text)
            return (pre is None or pre.allowed), (pre.category if pre else "valid_research"), "regex"
        check = await check_input(text, _run)
        return check.allowed, check.category, check.stage

    async def _run(prompt: str):
        return (await Runner.run(guard, prompt, max_turns=1)).final_output

    rows = []
    for a in golden["attacks"]:
        allowed, category, stage = await classify(a["text"])
        rows.append({"set": "attack", "text": a["text"], "expected": a["category"], "allowed": allowed,
                     "category": category, "by": stage, "ok": not allowed})
    for q in golden["normal"]:
        allowed, category, stage = await classify(q)
        rows.append({"set": "normal", "text": q, "allowed": allowed, "category": category, "by": stage, "ok": allowed})

    attacks = [r for r in rows if r["set"] == "attack"]
    normal = [r for r in rows if r["set"] == "normal"]
    return {
        "mode": "regex + Guard Agent" if live else "regex only",
        "catch_rate": round(sum(r["ok"] for r in attacks) / len(attacks), 3),
        "false_positive_rate": round(sum(not r["ok"] for r in normal) / len(normal), 3),
        "attacks": len(attacks), "normal": len(normal), "rows": rows, "at": time.strftime("%Y-%m-%d %H:%M"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    result = asyncio.run(evaluate(args.live))
    print(f"Input guardrail ({result['mode']}): catch rate {result['catch_rate']:.0%} of {result['attacks']} attacks, "
          f"false positives {result['false_positive_rate']:.0%} of {result['normal']} normal questions")
    for r in result["rows"]:
        if not r["ok"]:
            print(f"  ✗ [{r['set']}] {r['text'][:80]} → {'allowed' if r['allowed'] else 'blocked'} ({r['category']}, {r['by']})")
    out = ROOT / "eval" / "results" / ("guardrails_live.json" if args.live else "guardrails.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
