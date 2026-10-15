"""Evaluator-optimizer: Writer ⇄ Critic, bounded, with the verdict decided in Python.

Each round:
    1. CODE checks       citations exist, contradictions acknowledged, research gaps disclosed,
                         subquestion coverage
    2. RAG + LLM review  per finding: hybrid retrieval finds the most relevant saved evidence;
                         the Critic judges whether the cited quotes support the statement and
                         points to better evidence from what retrieval found
    3. CODE auto-fixes   strength rules (partial support is never "strong", low-quality-only
                         evidence is "weak") — applied directly, no LLM round needed
    4. CODE verdict      PASS   no blocking issues
                         REVISE blocking issues and rounds left → Writer gets the instructions
                         FAIL   loop ended with blocking issues (max rounds, no progress, or
                                Writer error); the best draft is kept and flagged for review

Stop conditions: PASS · max rounds · no progress (same blocking issues twice) · Writer error.
Findings the Critic already judged are not re-judged if the Writer leaves them unchanged.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from agents import Agent, custom_span

from ..context import AppContext
from ..models import (
    Critique, CriticReview, EvidenceAnalysis, FindingCheck, ResearchPlan, ResearchReport, WorkerFindings,
    normalize_evidence_id,
)
from ..tools.evidence import check_citation, retrieve_evidence
from .quality import normalize_quote

MAX_ROUNDS_CAP = 5           # hard ceiling, whatever MAX_CRITIC_ROUNDS says
RETRIEVED_PER_FINDING = 3    # "other relevant evidence" shown to the Critic per finding
LOW_QUALITY = 0.5            # sources below this alone can only support a "weak" finding
HIGH_QUALITY = 0.8           # one source at or above this can support a "strong" finding
_SUPPORT_RANK = {"not_supported": 0, "partial": 1, "supported": 2}

Severity = Literal["blocking", "advisory"]
RunStep = Callable[[Agent, str], Awaitable[Any]]
EventCallback = Callable[[str, dict], None]


@dataclass
class Issue:
    severity: Severity
    kind: str          # invalid_citation | unsupported | contradiction_ignored | gaps_hidden | coverage | critic
    message: str
    instruction: str = ""
    finding: int | None = None


@dataclass
class CriticRound:
    round: int
    verdict: Literal["PASS", "REVISE", "FAIL"]
    report: ResearchReport
    critique: Critique
    issues: list[Issue]
    autofixes: list[str]
    support: dict[int, str]          # finding number → support level
    citations: dict[int, dict[str, str]]  # finding number → {evidence id → support level}
    retrieval_method: str
    judged: int
    reused: int
    seconds: float

    @property
    def blocking(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "blocking"]

    def summary(self) -> dict:
        return {
            "round": self.round, "verdict": self.verdict, "blocking": len(self.blocking),
            "advisory": len(self.issues) - len(self.blocking),
            "issues": [f"[{i.severity}] {i.message}" for i in self.issues],
            "autofixes": self.autofixes, "support": self.support, "retrieval": self.retrieval_method,
            "judged": self.judged, "reused": self.reused, "seconds": self.seconds,
            "revision_instructions": self.critique.revision_instructions,
        }


@dataclass
class LoopResult:
    report: ResearchReport
    verdict: Literal["PASS", "FAIL"]
    stop_reason: str                 # passed | max_rounds | no_progress | writer_error
    rounds: list[CriticRound] = field(default_factory=list)
    best_round: int = 1

    @property
    def history(self) -> list[dict]:
        return [r.summary() for r in self.rounds]


# --------------------------------------------------------------------------- deterministic checks

_ID_IN_TEXT = re.compile(r"\bev\s*[-_]?\s*\d+\b", re.IGNORECASE)


def _ids_in(text: str) -> set[str]:
    return {normalize_evidence_id(m) for m in _ID_IN_TEXT.findall(text or "")}


def code_checks(ctx: AppContext, plan: ResearchPlan, analysis: EvidenceAnalysis,
                workers: list[WorkerFindings], report: ResearchReport) -> list[Issue]:
    issues: list[Issue] = []

    # 1. Every cited id must exist and its quote must still match the source.
    for n, f in enumerate(report.findings, 1):
        for eid in f.evidence_ids:
            det = check_citation(ctx, eid)
            if not det["valid"]:
                issues.append(Issue("blocking", "invalid_citation", f"Finding {n} cites {eid}: {det['reason']}",
                                    f"Finding {n} cites {eid}, which {det['reason']}. Replace it with a valid id from "
                                    "\"Valid evidence ids\" or remove the citation (and the finding, if nothing "
                                    "supports it).", n))

    # 2. Contradictions found by the Evidence Analyst must be acknowledged.
    if analysis.contradictions and not report.contradictions:
        topics = "; ".join(f"{c.topic} ({', '.join(c.evidence_ids_a)} vs {', '.join(c.evidence_ids_b)})"
                           for c in analysis.contradictions)
        issues.append(Issue("blocking", "contradiction_ignored",
                            f"The analysis found {len(analysis.contradictions)} contradiction(s) but the report lists none",
                            f"Add these contradictions to 'contradictions', citing the evidence ids in brackets and "
                            f"saying why the sources differ: {topics}."))
    else:
        mentioned = _ids_in(" ".join(report.contradictions))
        for c in analysis.contradictions:
            if not (set(c.evidence_ids_a) & mentioned and set(c.evidence_ids_b) & mentioned):
                issues.append(Issue("advisory", "contradiction_ignored",
                                    f"Contradiction '{c.topic}' ({', '.join(c.evidence_ids_a)} vs "
                                    f"{', '.join(c.evidence_ids_b)}) is not clearly addressed",
                                    f"Address the contradiction '{c.topic}' [{', '.join(c.evidence_ids_a)} vs "
                                    f"{', '.join(c.evidence_ids_b)}] in 'contradictions'."))

    # 3. Failed research must be visible: failed workers or gaps need limitations.
    incomplete = [w.specialist for w in workers if w.status != "success"]
    if (incomplete or analysis.gaps) and not report.limitations:
        what = ", ".join([f"{s} research {next(w.status for w in workers if w.specialist == s)}"
                          for s in incomplete] + [f"weak evidence for {g}" for g in analysis.gaps])
        issues.append(Issue("blocking", "gaps_hidden", f"Research was incomplete ({what}) but limitations is empty",
                            f"List these in 'limitations': {what}."))

    # 4. Coverage: a subquestion with saved evidence should be reflected in the findings.
    cited = report.cited_evidence_ids()
    by_sq: dict[str, list[str]] = {}
    for ev in ctx.store.list_evidence(ctx.run_id):
        by_sq.setdefault(ev.subquestion_id, []).append(ev.id)
    for sq in plan.subquestions:
        ids = by_sq.get(sq.id, [])
        if ids and not (set(ids) & cited):
            issues.append(Issue("advisory", "coverage", f"{sq.id} has evidence ({', '.join(ids[:4])}) but no finding uses it",
                                f"Add a finding for {sq.id} ('{sq.question[:90]}') using {', '.join(ids[:4])}, "
                                "or explain in limitations why it is left out."))
    return issues


def apply_strength_rules(ctx: AppContext, report: ResearchReport, support: dict[int, str]) -> tuple[ResearchReport, list[str]]:
    """Python decides what strength the evidence allows; the Writer's label can only go down."""
    fixes: list[str] = []
    findings = []
    for n, f in enumerate(report.findings, 1):
        strength = f.strength
        if support.get(n) == "not_supported":  # already a blocking issue; the Writer must rewrite it
            findings.append(f)
            continue
        sources = {}
        quotes: set[str] = set()
        for eid in f.evidence_ids:
            ev = ctx.store.get_evidence(ctx.run_id, eid)
            src = ctx.store.get_source(ctx.run_id, ev.source_id) if ev else None
            if ev and normalize_quote(ev.quote) in quotes:
                continue  # the same sentence cited twice (e.g. fresh + from memory) is one piece of evidence
            if ev:
                quotes.add(normalize_quote(ev.quote))
            if src:
                sources[str(src.url).rstrip("/")] = src.quality_score  # distinct pages, not source rows
        qualities = list(sources.values())
        if strength == "strong" and support.get(n) == "partial":
            strength = "moderate"
            fixes.append(f"Finding {n}: strong → moderate (support is only partial)")
        if strength == "strong" and not (len(qualities) >= 2 or any(q >= HIGH_QUALITY for q in qualities)):
            strength = "moderate"
            fixes.append(f"Finding {n}: strong → moderate (needs 2+ sources or one with quality ≥ {HIGH_QUALITY})")
        if strength != "weak" and qualities and all(q < LOW_QUALITY for q in qualities):
            fixes.append(f"Finding {n}: {strength} → weak (only low-quality sources, all < {LOW_QUALITY})")
            strength = "weak"
        findings.append(f if strength == f.strength else f.model_copy(update={"strength": strength}))
    return (report.model_copy(update={"findings": findings}) if fixes else report), fixes


# --------------------------------------------------------------------------- RAG + LLM review

def _finding_key(statement: str, evidence_ids: list[str]) -> str:
    return json.dumps([" ".join(statement.split()).lower(), sorted(evidence_ids)])


def _evidence_view(ctx: AppContext, eid: str, quote_chars: int) -> dict:
    ev = ctx.store.get_evidence(ctx.run_id, eid)
    src = ctx.store.get_source(ctx.run_id, ev.source_id) if ev else None
    return {"id": eid, "claim": ev.claim if ev else "", "quote": (ev.quote if ev else "")[:quote_chars],
            "source_type": src.domain_type if src else "unknown",
            "source_quality": src.quality_score if src else 0.0}


async def build_critic_prompt(ctx: AppContext, plan: ResearchPlan, analysis: EvidenceAnalysis,
                              report: ResearchReport, to_judge: list[int]) -> tuple[str, dict[int, set[str]], str]:
    """Returns (prompt, retrieved ids per finding, retrieval method). No run ids or timestamps go in,
    so the same report and evidence give the same prompt (record/replay stays valid)."""
    items, retrieved, method = [], {}, "none"
    for n in to_judge:
        f = report.findings[n - 1]
        valid = [e for e in f.evidence_ids if check_citation(ctx, e)["valid"]]
        found = await asyncio.to_thread(retrieve_evidence, ctx, f.statement, RETRIEVED_PER_FINDING, set(valid))
        method = found["method"] if found["method"] != "none" else method
        retrieved[n] = {r["evidence_id"] for r in found["results"]}
        items.append({
            "finding": n, "statement": f.statement, "strength": f.strength,
            "cited_evidence": [_evidence_view(ctx, e, 400) for e in valid],
            "other_relevant_evidence": [
                {"id": r["evidence_id"], "claim": r["claim"], "quote": r["quote"][:250],
                 "source_quality": r["source_quality"]} for r in found["results"]],
        })
    contradictions = [c.model_dump() for c in analysis.contradictions]
    prompt = (
        f"Decision question: {plan.main_question}\n\n"
        f"Report summary: {report.executive_summary}\n"
        f"Recommendation: {report.recommendation}\n"
        f"Report contradictions section: {json.dumps(report.contradictions, ensure_ascii=False)}\n\n"
        f"Contradictions found by the evidence analysis:\n{json.dumps(contradictions, indent=1, ensure_ascii=False)}\n\n"
        f"Findings to check (JSON):\n{json.dumps(items, indent=1, ensure_ascii=False)}"
    )
    return prompt, retrieved, method


def _clean_review(review: CriticReview, report: ResearchReport, to_judge: list[int],
                  retrieved: dict[int, set[str]]) -> dict[int, FindingCheck]:
    """Keep only what the Critic was asked about; drop ids it was never shown."""
    out: dict[int, FindingCheck] = {}
    for fc in review.findings:
        if fc.finding not in to_judge or fc.finding in out:
            continue
        cited = set(report.findings[fc.finding - 1].evidence_ids)
        citations = [c for c in fc.citations if normalize_evidence_id(c.evidence_id) in cited]
        citations = [c.model_copy(update={"evidence_id": normalize_evidence_id(c.evidence_id)}) for c in citations]
        better = [normalize_evidence_id(i) for i in fc.better_evidence_ids]
        better = [i for i in dict.fromkeys(better) if i in retrieved.get(fc.finding, set())]
        support = fc.support
        # The finding can't be better supported than its best citation.
        if citations and _SUPPORT_RANK[support] > max(_SUPPORT_RANK[c.support] for c in citations):
            support = max((c.support for c in citations), key=_SUPPORT_RANK.__getitem__)
        out[fc.finding] = fc.model_copy(update={"citations": citations, "better_evidence_ids": better,
                                                "support": support})
    return out


def issues_from_checks(report: ResearchReport, checks: dict[int, dict]) -> list[Issue]:
    issues = []
    for n, chk in sorted(checks.items()):
        f = report.findings[n - 1]
        better = f" Candidates from retrieval: {', '.join(chk['better'])}." if chk["better"] else ""
        problem = chk["problem"] or "the cited quotes do not back it"
        if chk["support"] == "not_supported":
            issues.append(Issue("blocking", "unsupported",
                                f"Finding {n} (\"{f.statement[:80]}\") is not supported: {problem}",
                                f"Finding {n} (\"{f.statement[:140]}\") is not supported by "
                                f"{', '.join(f.evidence_ids)}: {problem}. Rewrite it to say only what the quotes "
                                f"say, cite better evidence, or remove it.{better}", n))
        elif chk["support"] == "partial":
            issues.append(Issue("advisory", "unsupported",
                                f"Finding {n} (\"{f.statement[:80]}\") is only partly supported: {problem}",
                                f"Finding {n}: narrow the statement to what the quotes support ({problem}).{better}", n))
    return issues


# --------------------------------------------------------------------------- the loop

def _critique(issues: list[Issue], verdict: str) -> Critique:
    return Critique(
        verdict=verdict,  # type: ignore[arg-type]
        unsupported_claims=[i.message for i in issues if i.kind == "unsupported"],
        missing_sources=[i.message for i in issues if i.kind in ("invalid_citation", "coverage")],
        contradictions_ignored=[i.message for i in issues if i.kind == "contradiction_ignored"],
        revision_instructions=[i.instruction for i in issues if i.instruction] if verdict == "REVISE" else [],
    )


def revision_prompt(writer_prompt: str, report: ResearchReport, instructions: list[str],
                    source: str = "the Critic") -> str:
    return (writer_prompt + "\n\nPrevious draft (JSON):\n" + report.model_dump_json(indent=1)
            + f"\n\nRevision instructions (from {source}):\n" + "\n".join(f"- {i}" for i in instructions)
            + "\nKeep findings that are not mentioned above exactly as they are.")


async def writer_critic_loop(ctx: AppContext, plan: ResearchPlan, analysis: EvidenceAnalysis,
                             workers: list[WorkerFindings], writer_prompt: str, draft: ResearchReport, *,
                             run_writer: RunStep, run_critic: RunStep, writer_agent: Callable[[], Agent],
                             critic_agent: Callable[[], Agent], max_rounds: int,
                             on_event: EventCallback | None = None) -> LoopResult:
    max_rounds = max(1, min(int(max_rounds), MAX_ROUNDS_CAP))
    judged: dict[str, dict] = {}       # finding key → {"support", "problem", "better", "citations"}
    rounds: list[CriticRound] = []
    report = draft
    stop_reason = "max_rounds"

    for rnd in range(1, max_rounds + 1):
        t0 = time.time()
        with custom_span(f"Critic round {rnd}", data={"round": rnd, "findings": len(report.findings)}):
            with custom_span("Critic checks (code)"):
                issues = code_checks(ctx, plan, analysis, workers, report)

            keys = {n: _finding_key(f.statement, f.evidence_ids) for n, f in enumerate(report.findings, 1)}
            invalid_only = {i.finding for i in issues if i.kind == "invalid_citation"}
            to_judge = [n for n, f in enumerate(report.findings, 1)
                        if keys[n] not in judged
                        and any(check_citation(ctx, e)["valid"] for e in f.evidence_ids)]
            reused = sum(1 for n in keys if keys[n] in judged)
            method = "cached" if not to_judge else "none"
            if to_judge:
                prompt, retrieved, method = await build_critic_prompt(ctx, plan, analysis, report, to_judge)
                try:
                    with custom_span("Critic review (LLM + RAG)", data={"findings": to_judge, "retrieval": method}):
                        review: CriticReview = await run_critic(critic_agent(), prompt)
                    cleaned = _clean_review(review, report, to_judge, retrieved)
                    for n, fc in cleaned.items():
                        judged[keys[n]] = {"support": fc.support, "problem": fc.problem,
                                           "better": fc.better_evidence_ids,
                                           "citations": {c.evidence_id: c.support for c in fc.citations}}
                    missing = [n for n in to_judge if n not in cleaned]
                    if missing:
                        issues.append(Issue("advisory", "critic", f"Critic skipped finding(s) {missing}; "
                                                                  "they will be reviewed next round"))
                    for msg in review.contradictions_ignored[:3]:
                        issues.append(Issue("advisory", "contradiction_ignored", f"Critic: {msg[:200]}",
                                            f"Address this contradiction: {msg[:200]}"))
                    for msg in review.other_issues[:3]:
                        issues.append(Issue("advisory", "critic", f"Critic: {msg[:200]}", msg[:200]))
                except Exception as exc:  # a failed review must not lose the report
                    issues.append(Issue("advisory", "critic",
                                        f"Critic review failed ({type(exc).__name__}); support not verified this round"))

            checks = {n: judged[keys[n]] for n in keys if keys[n] in judged and n not in invalid_only}
            issues += issues_from_checks(report, checks)
            support = {n: c["support"] for n, c in checks.items()}
            with custom_span("Strength rules (code)"):
                report, fixes = apply_strength_rules(ctx, report, support)

            blocking = [i for i in issues if i.severity == "blocking"]
            no_progress = bool(rounds) and bool(blocking) and \
                {i.message for i in rounds[-1].blocking} == {i.message for i in blocking}
            if not blocking:
                verdict, stop_reason = "PASS", "passed"
            elif rnd >= max_rounds:
                verdict, stop_reason = "FAIL", "max_rounds"
            elif no_progress:
                verdict, stop_reason = "FAIL", "no_progress"
            else:
                verdict = "REVISE"
            critic_round = CriticRound(rnd, verdict, report, _critique(issues, verdict), issues, fixes, support,
                                       {n: c["citations"] for n, c in checks.items()},
                                       method, len(to_judge), reused, round(time.time() - t0, 2))
            rounds.append(critic_round)
        if on_event:
            on_event("critic_round", critic_round.summary())
        if verdict != "REVISE":
            break

        # REVISE: the Writer gets every instruction (blocking first, then advisory).
        instructions = [i.instruction for i in blocking if i.instruction] + \
                       [i.instruction for i in issues if i.severity == "advisory" and i.instruction]
        try:
            with custom_span(f"Writer revision {rnd}"):
                report = await run_writer(writer_agent(), revision_prompt(writer_prompt, report, instructions))
        except Exception as exc:
            critic_round.verdict = "FAIL"
            critic_round.critique = _critique(issues, "FAIL")
            stop_reason = "writer_error"
            if on_event:
                on_event("revision_failed", {"round": rnd, "error": f"{type(exc).__name__}: {exc}"[:200]})
            break
        if on_event:
            on_event("revised", {"round": rnd, "findings": len(report.findings)})

    # Best draft: fewest blocking issues, latest on a tie.
    best = min(rounds, key=lambda r: (len(r.blocking), -r.round))
    final_verdict: Literal["PASS", "FAIL"] = "PASS" if not best.blocking else "FAIL"
    _store_support(ctx, best)
    return LoopResult(report=best.report, verdict=final_verdict, stop_reason=stop_reason,
                      rounds=rounds, best_round=best.round)


def _store_support(ctx: AppContext, best: CriticRound) -> None:
    """Record claim support on the cited evidence of the kept report (per citation where the Critic
    judged it, else the finding's level). Confidence ignores evidence judged not_supported."""
    levels: dict[str, str] = {}
    for n, f in enumerate(best.report.findings, 1):
        if n not in best.support:
            continue
        per_citation = best.citations.get(n, {})
        for eid in f.evidence_ids:
            level = per_citation.get(eid, best.support[n])
            prev = levels.get(eid)
            if prev is None or _SUPPORT_RANK[level] > _SUPPORT_RANK[prev]:
                levels[eid] = level
    for eid, level in levels.items():
        if ctx.store.get_evidence(ctx.run_id, eid) is not None:
            ctx.store.set_support(ctx.run_id, eid, level)
