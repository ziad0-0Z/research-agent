"""Evaluation: one place that turns golden sets and recorded runs into measured numbers.

Used by eval/run_eval.py (the command line) and by the UI's Evaluation page, so both show the same numbers.

Three kinds of measurement, always labelled with how they were made:
- offline:  deterministic code on a golden set (code router, regex guardrail). Repeatable, no API.
- recorded: computed from runs already stored in research.db (citations, grounding, planted faults,
            parallel timing, routing of questions that were actually run). No new model calls.
- live:     calls the real agents on a golden set (Guard + Planner + Orchestrator routing, claim-support
            judge). Needs the API key; costs free-tier requests.

Nothing here invents a number: a metric with no data returns None and the caller says "not measured".
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from .config import PROJECT_ROOT, Settings
from .store import Store
from .workflows.router import route_question, validate_assignments

GOLDEN_DIR = PROJECT_ROOT / "data" / "golden"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
SPECIALISTS = ("technical", "market", "risk")


def load_golden(name: str) -> dict:
    return json.loads((GOLDEN_DIR / f"{name}.json").read_text(encoding="utf-8"))


def _state(row) -> dict:
    try:
        return json.loads(row["state_json"] or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}


def rate(ok: int, n: int) -> float | None:
    return round(ok / n, 3) if n else None


# =========================================================================== routing

def score_specialists(expected: dict, actual: set[str]) -> tuple[bool, str]:
    """Specialists are right when every required one ran and no forbidden one did."""
    missing = [s for s in expected.get("required", []) if s not in actual]
    extra = [s for s in expected.get("forbidden", []) if s in actual]
    if not missing and not extra:
        return True, "ok"
    parts = ([f"missing {', '.join(missing)}"] if missing else []) + ([f"unexpected {', '.join(extra)}"] if extra else [])
    return False, "; ".join(parts)


def score_route(expected: dict, path: str, domains: list[str]) -> tuple[bool, str]:
    if path != expected["route"]:
        return False, f"routed {path}, expected {expected['route']}"
    if expected["route"] == "sensitive" and not set(expected.get("domains", [])) <= set(domains):
        return False, f"domains {domains}, expected {expected['domains']}"
    return True, "ok"


def routing_offline(questions: list[dict], sensitive_domains: tuple[str, ...] = ("financial", "medical", "legal")) -> dict:
    """The CODE router alone (keywords). Deterministic: the same input gives the same number every time."""
    rows = []
    for q in questions:
        r = route_question(q["question"], sensitive_domains=sensitive_domains)
        ok, why = score_route(q, r.path, r.domains)
        rows.append({"id": q["id"], "question": q["question"], "expected": q["route"], "path": r.path,
                     "domains": r.domains, "matched": r.matched_terms, "ok": ok, "why": why, "note": q.get("note", "")})
    n_ok = sum(r["ok"] for r in rows)
    return {"mode": "offline (code router, keywords only)", "n": len(rows), "correct": n_ok,
            "accuracy": rate(n_ok, len(rows)), "rows": rows}


def routing_from_runs(store: Store, questions: list[dict]) -> dict:
    """Routing of golden questions that were really run: route + specialists as recorded in research.db.
    Uses the newest run of each question that got as far as delegation."""
    by_text = {q["question"].strip().lower(): q for q in questions}
    seen: set[str] = set()
    rows = []
    for row in store.latest_runs(1000):
        q = by_text.get((row["question"] or "").strip().lower())
        if q is None or q["id"] in seen:
            continue
        s = _state(row)
        workers = {w.get("specialist") for w in s.get("workers") or [] if w.get("specialist")}
        if not workers or not s.get("route"):
            continue  # blocked or stopped before delegation: says nothing about routing
        seen.add(q["id"])
        route_ok, route_why = score_route(q, s["route"], q.get("domains", []) if s["route"] == "sensitive" else [])
        spec_ok, spec_why = score_specialists(q, workers)
        rows.append({"id": q["id"], "question": q["question"], "run_id": row["id"], "path": s["route"],
                     "specialists": sorted(workers), "route_ok": route_ok, "specialists_ok": spec_ok,
                     "ok": route_ok and spec_ok, "why": "; ".join(w for w in (route_why, spec_why) if w != "ok") or "ok"})
    n_ok = sum(r["ok"] for r in rows)
    return {"mode": "recorded (golden questions found in research.db)", "n": len(rows), "correct": n_ok,
            "accuracy": rate(n_ok, len(rows)), "rows": rows}


RunAgent = Callable[..., Awaitable[Any]]


async def routing_live(questions: list[dict], settings: Settings, run_agent: RunAgent | None = None,
                       on_row: Callable[[dict], None] | None = None) -> dict:
    """The real routing path for each golden question: input guardrail (regex + Guard Agent) → code router
    (keywords + the Guard's domain) → Planner types → Orchestrator → validate_assignments (code).
    No research is done: 3 model calls per question."""
    from .context import AppContext
    from .guardrails.input import check_input
    from .workflows import pipeline

    run_agent = run_agent or pipeline.default_run_agent
    store = Store(":memory:")
    rows = []
    for q in questions:
        ctx = AppContext(settings=settings, store=store, question=q["question"])
        row: dict[str, Any] = {"id": q["id"], "question": q["question"], "expected_route": q["route"]}
        try:
            check = await check_input(q["question"], lambda p, ctx=ctx: run_agent(pipeline._guard(), p, ctx, 1))
            ctx.input_check = check
            if not check.allowed:
                row.update(ok=False, route_ok=False, specialists_ok=False, why=f"blocked by the input guardrail ({check.category})")
            else:
                route = route_question(q["question"], guard_domain=check.domain, sensitive_domains=settings.sensitive_domains)
                plan = await run_agent(pipeline._planner(), q["question"], ctx, 2)
                decision = await run_agent(pipeline._orchestrator(), f"Research plan:\n{plan.model_dump_json(indent=1)}", ctx, 2)
                assignment = validate_assignments(plan, decision)
                specialists = {k for k, v in assignment.by_specialist.items() if v}
                route_ok, route_why = score_route(q, route.path, route.domains)
                spec_ok, spec_why = score_specialists(q, specialists)
                row.update(path=route.path, domains=route.domains, guard_domain=check.domain,
                           planner_types=[sq.type for sq in plan.subquestions], specialists=sorted(specialists),
                           corrections=assignment.notes, route_ok=route_ok, specialists_ok=spec_ok,
                           ok=route_ok and spec_ok,
                           why="; ".join(w for w in (route_why, spec_why) if w != "ok") or "ok")
        except Exception as exc:  # noqa: BLE001 - an outage is recorded as an error row, never as a pass
            row.update(ok=False, route_ok=False, specialists_ok=False, error=f"{type(exc).__name__}: {str(exc)[:200]}",
                       why="model call failed")
        rows.append(row)
        if on_row:
            on_row(row)
    scored = [r for r in rows if "error" not in r]
    return {
        "mode": "live (Guard + code router + Planner + Orchestrator)", "n": len(scored), "errors": len(rows) - len(scored),
        "correct": sum(r["ok"] for r in scored), "accuracy": rate(sum(r["ok"] for r in scored), len(scored)),
        "route_accuracy": rate(sum(r["route_ok"] for r in scored), len(scored)),
        "specialist_accuracy": rate(sum(r["specialists_ok"] for r in scored), len(scored)),
        "rows": rows,
    }


# =========================================================================== claim support (live)

async def claim_support_live(pairs: list[dict], run_agent: RunAgent | None = None) -> dict:
    """Claim Support Judge accuracy on fixed (quote, claim, expected) pairs. Prompt identical to validate_citation."""
    from agents import Runner

    from .agents.tool_agents import support_judge

    async def default(agent, prompt):
        return (await Runner.run(agent, prompt, max_turns=1)).final_output

    run_agent = run_agent or default
    rows = []
    for p in pairs:
        try:
            v = await run_agent(support_judge(), f"Claim: {p['claim']}\n\nQuote: {p['quote']}")
            rows.append({"id": p["id"], "claim": p["claim"], "expected": p["expected"], "got": v.support,
                         "reason": v.reason, "ok": v.support == p["expected"],
                         # the decision that matters downstream: may this citation back the claim at all?
                         "gate_ok": (v.support == "not_supported") == (p["expected"] == "not_supported")})
        except Exception as exc:  # noqa: BLE001
            rows.append({"id": p["id"], "claim": p["claim"], "expected": p["expected"], "error": f"{type(exc).__name__}: {exc}"[:200]})
    scored = [r for r in rows if "error" not in r]
    return {"mode": "live (Claim Support Judge)", "n": len(scored), "errors": len(rows) - len(scored),
            "accuracy": rate(sum(r["ok"] for r in scored), len(scored)),
            "gate_accuracy": rate(sum(r["gate_ok"] for r in scored), len(scored)), "rows": rows}


# =========================================================================== recorded runs (research.db)

def grounding_metrics(store: Store) -> dict[str, Any]:
    """Citation validity and grounding over every report the system released (status completed)."""
    cited = valid = findings = grounded = unchecked = 0
    runs = 0
    for row in store._query("SELECT * FROM runs WHERE status='completed'"):
        report = _state(row).get("report")
        if not report:
            continue
        runs += 1
        ev = {e.id: e for e in store.list_evidence(row["id"])}
        for f in report.get("findings", []):
            findings += 1
            ok = True
            for eid in f.get("evidence_ids", []):
                cited += 1
                e = ev.get(eid)
                if e and e.verified:
                    valid += 1
                if not (e and e.verified and e.support in ("supported", "partial")):
                    ok = False
                    if e and e.support == "unchecked":
                        unchecked += 1
            grounded += ok
    return {"runs": runs, "cited": cited, "valid": valid, "findings": findings, "grounded": grounded,
            "unchecked": unchecked,
            "citation_validity": valid / cited if cited else None,
            "grounding_rate": grounded / findings if findings else None}


def _fault(state: dict) -> str | None:
    from .ui.data import demo_fault_of  # one definition, shared with the UI
    return demo_fault_of(state)


def fault_outcomes(store: Store) -> dict[str, Any]:
    """Planted faults and what stopped them, from runs recorded with --fault / a demo scenario.

    - fake_citation, tampered_quote (planted AFTER the Critic): caught when the output guardrail rejected the run.
    - unsupported_claim (planted BEFORE the Critic): caught when the Critic needed more than one round
      (the planted finding made round 1 REVISE) and the run was not released with the planted text.
    - search_outage: handled when the run finished (any final status) with a degraded reliability record.
    """
    out: dict[str, dict] = {k: {"runs": 0, "caught": 0, "run_ids": []} for k in
                            ("fake_citation", "tampered_quote", "unsupported_claim", "search_outage")}
    for row in store.latest_runs(1000):
        s = _state(row)
        fault = _fault(s)
        if fault not in out:
            continue
        st = row["status"]
        if st == "running":
            continue
        o = out[fault]
        o["runs"] += 1
        o["run_ids"].append(row["id"])
        if fault in ("fake_citation", "tampered_quote"):
            caught = st == "rejected_by_guardrail"
        elif fault == "unsupported_claim":
            from .workflows.pipeline import PLANTED_STATEMENT
            findings = [f.get("statement", "") for f in (s.get("report") or {}).get("findings", [])]
            caught = (s.get("critic_rounds") or 0) >= 2 and PLANTED_STATEMENT not in findings
        else:
            rel = s.get("reliability") or {}
            caught = st in ("completed", "paused_for_review", "rejected") and bool(
                rel.get("partial") or (rel.get("search") or {}).get("fallback_calls"))
        o["caught"] += caught
    after = [out["fake_citation"], out["tampered_quote"]]
    n = sum(x["runs"] for x in after)
    return {"faults": out, "output_guardrail_runs": n, "output_guardrail_caught": sum(x["caught"] for x in after),
            "output_guardrail_catch_rate": rate(sum(x["caught"] for x in after), n)}


def parallel_stats(store: Store) -> dict[str, Any]:
    """Measured speedup of running specialists in parallel, from each run's recorded worker times."""
    rows = []
    for row in store.latest_runs(1000):
        tl = _state(row).get("timeline") or {}
        bars = tl.get("bars") or []
        if len(bars) >= 2 and tl.get("wall_seconds"):
            rows.append({"run_id": row["id"], "workers": len(bars), "wall": tl["wall_seconds"],
                         "sequential": tl["sequential_seconds"], "speedup": tl["speedup"]})
    if not rows:
        return {"runs": 0, "median_speedup": None, "rows": []}
    sp = sorted(r["speedup"] for r in rows)
    mid = len(sp) // 2
    median = sp[mid] if len(sp) % 2 else round((sp[mid - 1] + sp[mid]) / 2, 2)
    return {"runs": len(rows), "median_speedup": median, "min_speedup": sp[0], "max_speedup": sp[-1], "rows": rows}


def critic_stats(store: Store) -> dict[str, Any]:
    rounds, verdicts = [], {}
    for row in store.latest_runs(1000):
        s = _state(row)
        if s.get("critic_verdict"):
            rounds.append(s.get("critic_rounds") or 0)
            verdicts[s["critic_verdict"]] = verdicts.get(s["critic_verdict"], 0) + 1
    return {"runs": len(rounds), "verdicts": verdicts,
            "mean_rounds": round(sum(rounds) / len(rounds), 2) if rounds else None,
            "revised_runs": sum(1 for r in rounds if r >= 2)}


@dataclass
class Scenario:
    number: int
    name: str
    proof: str           # what in the stored run proves it
    run_id: str | None
    how: str             # "recorded" | "tests" | "missing"


SCENARIO_NAMES = {
    1: "Happy path", 2: "3+ sequential steps", 3: "Routing (sensitive → review path)", 4: "Parallelization",
    5: "Orchestrator-workers (subset of specialists)", 6: "Evaluator rejects, then passes", 7: "Input guardrail",
    8: "Tool blocked", 9: "Output guardrail", 10: "Failure recovery", 11: "Human escalation", 12: "Trace",
}


def scenario_coverage(store: Store, traces_dir: Path | None = None) -> list[Scenario]:
    """For each of the 12 required scenarios, the newest stored run that demonstrates it (or why none does)."""
    found: dict[int, tuple[str, str]] = {}
    weaker: dict[int, tuple[str, str]] = {}  # used only when no clean example exists

    def take(n: int, run_id: str, proof: str, clean: bool = True) -> None:
        (found if clean else weaker).setdefault(n, (run_id, proof))

    for row in store.latest_runs(1000):  # newest first
        s, st, rid = _state(row), row["status"], row["id"]
        fault = _fault(s)
        workers = {w.get("specialist") for w in s.get("workers") or []}
        if st == "completed" and not fault:
            take(1, rid, "released with every check passed")
        if s.get("plan") and s.get("analysis") and s.get("report"):
            take(2, rid, "plan → evidence → analysis → report stored")
        if s.get("route") == "sensitive":
            take(3, rid, "code router chose the sensitive path")
        bars = (s.get("timeline") or {}).get("bars") or []
        if len(bars) >= 2:
            take(4, rid, f"{len(bars)} specialists overlapped, speedup {(s.get('timeline') or {}).get('speedup')}×")
        if workers and len(workers - {None}) < 3 and s.get("plan"):
            take(5, rid, f"orchestrator used {', '.join(sorted(w for w in workers if w))} only", clean=not fault)
        if (s.get("critic_rounds") or 0) >= 2:
            take(6, rid, f"Critic: {s.get('critic_rounds')} rounds, final {s.get('critic_verdict')}")
        if st == "blocked":
            take(7, rid, f"blocked at input ({(s.get('input_check') or {}).get('category', 'guardrail')})")
        if st == "rejected_by_guardrail":
            take(9, rid, f"output guardrail rejected ({fault or 'real issue'})")
        rel = s.get("reliability") or {}
        if fault == "search_outage" or rel.get("partial") or (rel.get("search") or {}).get("fallback_calls"):
            take(10, rid, "search failed; run finished with a labelled partial report")
        elif rel.get("llm_fallbacks"):
            take(10, rid, f"{len(rel['llm_fallbacks'])} model call(s) recovered on a fallback model", clean=False)
        if s.get("review_history") or st == "paused_for_review":
            take(11, rid, "paused for a human decision" + (" and decided" if s.get("review_history") else ""))
        if s.get("trace_ids") and traces_dir is not None and any(
                (traces_dir / f"{t}.jsonl").exists() for t in s["trace_ids"]):
            take(12, rid, "trace file linked to the run")
    out = []
    for n, name in SCENARIO_NAMES.items():
        if n == 8:
            out.append(Scenario(8, name, "protected tools refuse unsafe calls (tests + live check button)", None, "tests"))
        elif n in found or n in weaker:
            run_id, proof = found.get(n) or weaker[n]
            out.append(Scenario(n, name, proof, run_id, "recorded"))
        else:
            out.append(Scenario(n, name, "no stored run shows this yet", None, "missing"))
    return out


# =========================================================================== results files

def save_result(name: str, data: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def load_result(name: str) -> dict | None:
    try:
        return json.loads((RESULTS_DIR / f"{name}.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
