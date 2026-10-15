"""The whole evaluation in one command. Writes eval/results/*.json and eval/RESULTS.md.

    python eval\\run_eval.py                 # offline + recorded runs (no API calls, ~2 s)
    python eval\\run_eval.py --live          # + live routing (Guard, Planner, Orchestrator) and claim-support judge
    python eval\\run_eval.py --live --limit 5   # live parts on the first 5 golden items only (free-tier friendly)
    python eval\\run_eval.py --rerun         # also re-run the regex guardrail and memory evals first
    python eval\\run_eval.py --tests         # also run pytest and record the result
    python eval\\run_eval.py --samples       # also export every released report to samples/

What each number is made from is written next to it: offline (golden set + code), recorded (research.db),
or live (real model calls). A metric with no data is reported as "not measured", never as a pass.
Retrieval recall is measured by eval\\retrieval_eval.py (it needs Ollama for the hybrid run) and only read here.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_agent import evaluation as ev  # noqa: E402
from research_agent.config import load_settings  # noqa: E402
from research_agent.store import Store  # noqa: E402

TARGETS = {
    "citation_validity": (1.0, "100%"), "grounding_rate": (0.95, "≥ 95%"), "routing_accuracy": (0.90, "≥ 90%"),
    "guardrail_catch_rate": (1.0, "100%"), "false_positive_rate": (0.05, "≤ 5%"), "retrieval_recall_at_5": (0.85, "≥ 85%"),
    "output_guardrail_catch_rate": (1.0, "100%"), "claim_support_accuracy": (0.85, "≥ 85%"),
}


def pct(x) -> str:
    return "—" if x is None else f"{x * 100:.0f}%"


def verdict(name: str, value) -> str:
    if value is None:
        return "not measured"
    target, _ = TARGETS[name]
    ok = value <= target if name == "false_positive_rate" else value >= target
    return "meets target" if ok else "BELOW TARGET"


def run_script(args: list[str]) -> None:
    print(f"\n$ python {' '.join(args)}")
    subprocess.run([sys.executable, *args], cwd=str(ROOT), check=False)


def run_tests() -> None:
    t0 = time.time()
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=str(ROOT),
                          capture_output=True, text=True)
    from research_agent.ui.data import parse_pytest_summary
    lines = (proc.stdout or "").strip().splitlines()
    counts = parse_pytest_summary(lines[-1] if lines else "")
    counts["seconds"] = round(time.time() - t0, 1)
    ev.save_result("tests", counts)
    print(f"\nTests: {counts}")


def export_samples(store: Store) -> int:
    out = ROOT / "samples"
    out.mkdir(exist_ok=True)
    n = 0
    for row in store._query("SELECT * FROM runs WHERE status='completed' ORDER BY created_at"):
        state = json.loads(row["state_json"] or "{}")
        md = state.get("final_markdown")
        path = out / f"{row['id']}.md"
        if md and not path.exists():
            path.write_text(md, encoding="utf-8")
            n += 1
    return n


def build_summary(store: Store, settings) -> dict:
    golden = ev.load_golden("questions")["questions"]
    routing_off = ev.routing_offline(golden, settings.sensitive_domains)
    routing_rec = ev.routing_from_runs(store, golden)
    ev.save_result("routing", {"offline": routing_off, "recorded": routing_rec, "at": time.strftime("%Y-%m-%d %H:%M")})

    g = ev.grounding_metrics(store)
    faults = ev.fault_outcomes(store)
    par = ev.parallel_stats(store)
    critic = ev.critic_stats(store)
    scenarios = [s.__dict__ for s in ev.scenario_coverage(store, settings.traces_dir)]
    ev.save_result("recorded", {"grounding": g, "faults": faults, "parallel": par, "critic": critic,
                                "scenarios": scenarios, "at": time.strftime("%Y-%m-%d %H:%M")})

    gl = ev.load_result("guardrails_live") or ev.load_result("guardrails")
    retr = ev.load_result("retrieval") or []
    best_retr = next((r for r in retr if r.get("method") == "hybrid"), None) or (retr[0] if retr else None)
    rl = ev.load_result("routing_live")
    cl = ev.load_result("claims_live")
    tests = ev.load_result("tests")
    metrics = {
        "citation_validity": (g["citation_validity"], f"{g['valid']}/{g['cited']} cited ids, {g['runs']} released reports", "recorded"),
        "grounding_rate": (g["grounding_rate"], f"{g['grounded']}/{g['findings']} findings", "recorded"),
        "routing_accuracy": ((rl or {}).get("accuracy") if rl else routing_off["accuracy"],
                             (f"live: {rl['correct']}/{rl['n']} (route {pct(rl['route_accuracy'])}, specialists "
                              f"{pct(rl['specialist_accuracy'])})" if rl else
                              f"code router only: {routing_off['correct']}/{routing_off['n']}"),
                             "live" if rl else "offline"),
        "guardrail_catch_rate": ((gl or {}).get("catch_rate"), (gl or {}).get("mode", "not run"), "live" if ev.load_result("guardrails_live") else "offline"),
        "false_positive_rate": ((gl or {}).get("false_positive_rate"), f"of {(gl or {}).get('normal', 0)} normal questions", "same run"),
        "retrieval_recall_at_5": ((best_retr or {}).get("recall_at_5"),
                                  f"{(best_retr or {}).get('method', 'not run')}: {(best_retr or {}).get('found')}/{(best_retr or {}).get('relevant')}",
                                  "offline"),
        "output_guardrail_catch_rate": (faults["output_guardrail_catch_rate"],
                                        f"{faults['output_guardrail_caught']}/{faults['output_guardrail_runs']} planted faults stopped", "recorded"),
        "claim_support_accuracy": ((cl or {}).get("accuracy"), f"{sum(r.get('ok', False) for r in (cl or {}).get('rows', []))}/{(cl or {}).get('n', 0)} pairs" if cl else "run with --live", "live"),
    }
    summary = {"at": time.strftime("%Y-%m-%d %H:%M"),
               "metrics": {k: {"value": v, "detail": d, "how": h, "target": TARGETS[k][1], "verdict": verdict(k, v)}
                           for k, (v, d, h) in metrics.items()},
               "routing": {"offline": {k: routing_off[k] for k in ("n", "correct", "accuracy")},
                           "recorded": {k: routing_rec[k] for k in ("n", "correct", "accuracy")},
                           "live": {k: rl[k] for k in ("n", "correct", "accuracy", "route_accuracy", "specialist_accuracy", "errors")} if rl else None},
               "parallel": {k: par[k] for k in par if k != "rows"}, "critic": critic,
               "faults": {k: {"runs": v["runs"], "caught": v["caught"]} for k, v in faults["faults"].items()},
               "scenarios": scenarios, "tests": tests}
    ev.save_result("summary", summary)
    return summary


def write_markdown(summary: dict) -> Path:
    m = summary["metrics"]
    lines = [f"# Evaluation results (generated {summary['at']})", "",
             "Generated by `python eval\\run_eval.py`. Do not edit by hand; definitions are in METRICS.md.", "",
             "| Metric | Result | Target | Verdict | Measured | Detail |", "|---|---|---|---|---|---|"]
    for k, v in m.items():
        lines.append(f"| {k.replace('_', ' ')} | {pct(v['value'])} | {v['target']} | {v['verdict']} | {v['how']} | {v['detail']} |")
    r = summary["routing"]
    lines += ["", "## Routing", "",
              f"- Code router alone (offline, 20 golden questions incl. 4 keyword traps): {r['offline']['correct']}/{r['offline']['n']}",
              f"- Golden questions actually run (research.db): {r['recorded']['correct']}/{r['recorded']['n']}" if r["recorded"]["n"] else
              "- Golden questions actually run: none recorded yet",
              (f"- Live (Guard + router + Planner + Orchestrator): {r['live']['correct']}/{r['live']['n']}, "
               f"route {pct(r['live']['route_accuracy'])}, specialists {pct(r['live']['specialist_accuracy'])}, errors {r['live']['errors']}")
              if r["live"] else "- Live: not run (`--live`)"]
    p, c = summary["parallel"], summary["critic"]
    lines += ["", "## Reliability and loops", "",
              f"- Parallel specialists: median speedup {p.get('median_speedup')}× over {p['runs']} runs "
              f"(min {p.get('min_speedup')}×, max {p.get('max_speedup')}×), from recorded worker times",
              f"- Critic: {c['runs']} runs, verdicts {c['verdicts']}, mean rounds {c['mean_rounds']}, {c['revised_runs']} needed a revision",
              "- Planted faults: " + ", ".join(f"{k} {v['caught']}/{v['runs']}" for k, v in summary["faults"].items())]
    lines += ["", "## The 12 demo scenarios", "", "| # | Scenario | Shown by | Run |", "|---|---|---|---|"]
    for s in summary["scenarios"]:
        lines.append(f"| {s['number']} | {s['name']} | {s['proof']} | {s['run_id'] or s['how']} |")
    t = summary.get("tests")
    lines += ["", f"Tests: {t.get('passed', 0)} passed, {t.get('failed', 0)} failed" if t else "", ""]
    out = ROOT / "eval" / "RESULTS.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


async def live_parts(settings, limit: int | None) -> None:
    settings.require_llm_key()
    golden = ev.load_golden("questions")["questions"][:limit]
    print(f"\nLive routing on {len(golden)} golden questions (3 model calls each)…")
    rl = await ev.routing_live(golden, settings, on_row=lambda r: print(
        f"  {'ok' if r.get('ok') else 'x'} {r['id']} {r.get('path', '')} {r.get('specialists', '')} {'' if r.get('ok') else r.get('why', '')}"))
    rl["at"] = time.strftime("%Y-%m-%d %H:%M")
    ev.save_result("routing_live", rl)
    pairs = ev.load_golden("claim_pairs")["pairs"][:limit]
    print(f"\nClaim-support judge on {len(pairs)} pairs…")
    cl = await ev.claim_support_live(pairs)
    cl["at"] = time.strftime("%Y-%m-%d %H:%M")
    for r in cl["rows"]:
        print(f"  {'ok' if r.get('ok') else 'x'} {r['id']} expected {r['expected']}, got {r.get('got', r.get('error'))}")
    ev.save_result("claims_live", cl)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="call the real agents (needs the API key)")
    parser.add_argument("--limit", type=int, default=None, help="live parts: first N golden items only")
    parser.add_argument("--rerun", action="store_true", help="re-run the regex guardrail and memory evals first")
    parser.add_argument("--tests", action="store_true", help="run pytest and record the result")
    parser.add_argument("--samples", action="store_true", help="export released reports to samples/")
    args = parser.parse_args()

    settings = load_settings()
    store = Store(settings.db_path)
    if args.rerun:
        run_script(["eval/guardrail_eval.py"])
        run_script(["eval/memory_eval.py"])
        if args.live:
            run_script(["eval/guardrail_eval.py", "--live"])
    if args.tests:
        run_tests()
    if args.live:
        asyncio.run(live_parts(settings, args.limit))
    if args.samples:
        print(f"\nExported {export_samples(store)} new sample report(s) to samples/")

    summary = build_summary(store, settings)
    print("\n" + "=" * 78)
    for k, v in summary["metrics"].items():
        print(f"{k.replace('_', ' '):<30} {pct(v['value']):>6}  target {v['target']:<6} {v['verdict']:<13} [{v['how']}] {v['detail']}")
    covered = sum(1 for s in summary["scenarios"] if s["how"] != "missing")
    print(f"{'demo scenarios shown':<30} {covered}/12")
    for s in summary["scenarios"]:
        if s["how"] == "missing":
            print(f"  missing: {s['number']} {s['name']}")
    off = ev.load_result("routing")["offline"]
    for r in off["rows"]:
        if not r["ok"]:
            print(f"  x router {r['id']}: {r['why']} — {r['question'][:70]}")
    print(f"\nSaved eval/results/summary.json and {write_markdown(summary).relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
