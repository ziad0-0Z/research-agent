"""Release: the code-only path from a checked draft to a final report.

    report
      → [CODE] output guardrail        fails → rejected_by_guardrail (no token, nothing released)
      → [CODE] approval token          bound to this report's content hash
      → [CODE] impact gate             sensitive domain · Critic FAIL · guard unavailable ·
                                       high impact · confidence < threshold   → human review
      → paused_for_review              persisted; resume after a human decides
        or generate_report(token)      → completed
      → [CODE] memory commit           completed runs only: verified evidence + the released report

Used by the pipeline and by resume, so a resumed run passes exactly the same checks again.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timezone
from typing import Literal

from agents import custom_span, guardrail_span

from ..context import AppContext
from ..guardrails.output import OutputCheck, check_output
from ..memory import commit_completed_run
from ..models import HumanReview, ResearchPlan, ResearchReport, WorkerFindings
from ..tools import review as review_tools

ReleaseStatus = Literal["completed", "paused_for_review", "rejected_by_guardrail"]
EventCallback = Callable[[str, dict], None]


@dataclass
class Release:
    status: ReleaseStatus
    output_check: OutputCheck
    gate_reasons: list[str] = field(default_factory=list)
    final_markdown: str | None = None
    review: HumanReview | None = None
    memory_commit: dict | None = None


def impact_gate(ctx: AppContext, report: ResearchReport, confidence: dict) -> list[str]:
    """Why this report needs a human before release. Empty list → it can be released directly."""
    reasons = list(ctx.review_reasons)  # sensitive route, Critic FAIL, guard unavailable (flagged earlier)
    if report.impact == "high":
        reasons.append("high-impact recommendation (hard to reverse)")
    threshold = ctx.settings.review_confidence_threshold
    if confidence["score"] < threshold:
        reasons.append(f"confidence {confidence['score']} is below {threshold}")
    return list(dict.fromkeys(reasons))


def final_header(ctx: AppContext, confidence: dict, review: HumanReview | None) -> str:
    lines = [f"> **Confidence {confidence['score']}** (computed in code): {confidence['explanation']}"]
    if review is not None and review.status == "approved":
        when = review.decided_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if review.decided_at else ""
        note = f" Note: {review.reviewer_note}" if review.reviewer_note else ""
        lines.append(f"> **Approved by a human reviewer** {when}. Review reasons: "
                     f"{'; '.join(review.reasons)}.{note}")
    return "\n".join(lines) + "\n\n"


def release_report(ctx: AppContext, workers: list[WorkerFindings], confidence: dict,
                   on_event: EventCallback | None = None, plan: ResearchPlan | None = None) -> Release:
    emit = on_event or (lambda stage, data: None)
    report = ctx.final_report

    with guardrail_span("Output guardrail (code)") as span:
        check = check_output(ctx, report, workers)
        span.span_data.triggered = not check.passed
    emit("output_checked", check.summary())
    if not check.passed:
        ctx.revoke_approval_token()
        ctx.store.update_run(ctx.run_id, status="rejected_by_guardrail")
        emit("rejected", {"reasons": [i.message for i in check.issues]})
        return Release("rejected_by_guardrail", check)

    token = ctx.issue_approval_token()
    assert report is not None
    with custom_span("Impact gate (code)"):
        reasons = impact_gate(ctx, report, confidence)
    for r in reasons:
        ctx.flag_review(r)
    review = ctx.store.get_review(ctx.run_id)
    approved = review is not None and review.status == "approved"
    emit("gate", {"needs_review": bool(reasons) and not approved, "reasons": reasons, "approved": approved})

    if reasons and not approved:
        review_tools.request_human_approval(ctx, reasons)
        review = ctx.store.get_review(ctx.run_id)
        emit("paused", {"reasons": review.reasons if review else reasons})
        return Release("paused_for_review", check, reasons, review=review)

    out = review_tools.generate_report(ctx, token)  # protected action: token + approval checked again inside
    markdown = final_header(ctx, confidence, review) + out["markdown"]
    emit("completed", {"chars": len(markdown)})
    # The one place long-term memory is written: only here is the run `completed`.
    # Best effort: the report is already released, so a memory error is reported, never raised.
    with custom_span("Memory commit (code)"):
        try:
            committed = commit_completed_run(ctx, plan=plan, workers=workers, confidence=confidence,
                                             final_markdown=markdown)
        except Exception as exc:
            committed = {"committed": False, "reason": f"memory error ({type(exc).__name__}: {exc})"[:300]}
    emit("memory_committed", committed)
    return Release("completed", check, reasons, markdown, review, committed)
