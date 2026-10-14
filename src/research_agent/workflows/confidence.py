"""Confidence is computed in Python, never self-reported by the LLM.

score = coverage × source_quality × (1 − contradiction_penalty) × worker_factor

- coverage: share of planned subquestions answered by at least one cited,
  usable piece of evidence (support not "not_supported")
- source_quality: mean quality of the distinct sources behind cited evidence
- contradiction_penalty: 0.1 per contradiction (max 0.3), halved when the
  report acknowledges contradictions
- worker_factor: −0.15 per failed or timed-out worker
"""
from __future__ import annotations

from ..context import AppContext
from ..models import ResearchPlan, ResearchReport, WorkerFindings


def compute_confidence(ctx: AppContext, plan: ResearchPlan, report: ResearchReport,
                       workers: list[WorkerFindings]) -> dict:
    cited = report.cited_evidence_ids()
    usable_by_sq: dict[str, int] = {}
    source_ids: set[str] = set()
    for eid in cited:
        ev = ctx.store.get_evidence(ctx.run_id, eid)
        if ev is None or not ev.verified or ev.support == "not_supported":
            continue
        usable_by_sq[ev.subquestion_id] = usable_by_sq.get(ev.subquestion_id, 0) + 1
        source_ids.add(ev.source_id)

    total_sq = len(plan.subquestions)
    covered = sum(1 for sq in plan.subquestions if usable_by_sq.get(sq.id))
    coverage = covered / total_sq if total_sq else 0.0

    qualities = []
    for sid in source_ids:
        src = ctx.store.get_source(ctx.run_id, sid)
        if src:
            qualities.append(src.quality_score)
    quality = sum(qualities) / len(qualities) if qualities else 0.0

    n_contra = len(ctx.analysis.contradictions) if ctx.analysis else 0
    penalty = min(0.3, 0.1 * n_contra)
    if penalty and report.contradictions:
        penalty /= 2

    bad_workers = sum(1 for w in workers if w.status in ("failed", "timeout"))
    worker_factor = max(0.0, 1 - 0.15 * bad_workers)

    score = coverage * quality * (1 - penalty) * worker_factor
    return {
        "score": round(score, 2),
        "coverage": round(coverage, 2),
        "covered_subquestions": f"{covered}/{total_sq}",
        "source_quality": round(quality, 2),
        "contradiction_penalty": round(penalty, 2),
        "worker_factor": round(worker_factor, 2),
        "explanation": (
            f"{covered}/{total_sq} subquestions covered by usable cited evidence; "
            f"mean source quality {quality:.2f}; {n_contra} contradiction(s)"
            f"{' (acknowledged)' if n_contra and report.contradictions else ''}; "
            f"{bad_workers} failed/timed-out worker(s)."
        ),
    }
