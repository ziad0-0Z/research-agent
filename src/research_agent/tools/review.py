"""request_human_approval and generate_report.

generate_report is the protected action: it renders only the report stored in
the run context (set by the pipeline after the output guardrail), and only with
an approval token that Python issued. The LLM can neither supply report content
nor forge the token.
"""
from __future__ import annotations

from ..context import AppContext
from ..models import HumanReview
from .evidence import check_citation
from .web import ToolError


def request_human_approval(ctx: AppContext, reasons: list[str]) -> dict:
    reasons = [r.strip()[:200] for r in (reasons or []) if r and r.strip()]
    reasons = list(dict.fromkeys(reasons + ctx.review_reasons))
    if not reasons:
        raise ToolError("at least one reason is required")
    existing = ctx.store.get_review(ctx.run_id)
    if existing and existing.status == "pending":
        ctx.log_tool("request_human_approval", True, "already pending (idempotent)")
        return {"status": "pending", "already_requested": True, "reasons": existing.reasons,
                "message": "Review already requested. Execution stays paused until a human decides."}
    review = HumanReview(run_id=ctx.run_id, reasons=reasons)
    ctx.store.upsert_review(review)
    ctx.store.update_run(ctx.run_id, status="paused_for_review")
    ctx.review_requested = True
    ctx.log_tool("request_human_approval", True, f"pending: {'; '.join(reasons)}")
    return {"status": "pending", "already_requested": False, "reasons": reasons,
            "message": "Human review requested. The report will not be finalised until it is approved."}


def generate_report(ctx: AppContext, approval_token: str) -> dict:
    if not ctx.approval_token_valid(approval_token):
        ctx.log_tool("generate_report", False, "BLOCKED: missing or invalid approval token")
        raise ToolError("blocked: generate_report requires a valid approval token issued after the "
                        "output guardrail (and human approval when required)")
    report = ctx.final_report
    if report is None:
        ctx.log_tool("generate_report", False, "no validated report in context")
        raise ToolError("no validated report available")
    review = ctx.store.get_review(ctx.run_id)
    if ctx.requires_review and (review is None or review.status != "approved"):
        ctx.log_tool("generate_report", False, "BLOCKED: human approval required but not granted")
        raise ToolError("blocked: this report requires human approval first")

    markdown = render_markdown(ctx)
    ctx.store.update_run(ctx.run_id, status="completed")
    ctx.log_tool("generate_report", True, f"rendered {len(markdown)} chars")
    return {"status": "completed", "markdown": markdown}


def render_markdown(ctx: AppContext) -> str:
    r = ctx.final_report
    assert r is not None
    lines = [f"# {r.title}", "", "## Executive summary", r.executive_summary, "", "## Findings"]
    for i, f in enumerate(r.findings, 1):
        refs = ", ".join(f"[{eid}]" for eid in f.evidence_ids)
        lines.append(f"{i}. {f.statement} {refs} _(evidence: {f.strength})_")
    if r.contradictions:
        lines += ["", "## Where sources disagree"] + [f"- {c}" for c in r.contradictions]
    lines += ["", "## Recommendation", r.recommendation, "", f"**Impact:** {r.impact}"]
    if r.limitations:
        lines += ["", "## Limitations"] + [f"- {x}" for x in r.limitations]

    lines += ["", "## Evidence appendix", "", "| Id | Quote | Source |", "|---|---|---|"]
    for eid in sorted(r.cited_evidence_ids(), key=lambda x: int(x[2:]) if x[2:].isdigit() else 0):
        det = check_citation(ctx, eid)
        if not det["valid"]:
            lines.append(f"| {eid} | _invalid: {det['reason']}_ | — |")
            continue
        ev, src = det["evidence"], det["source"]
        quote = ev.quote.replace("|", "\\|")
        lines.append(f"| {eid} | “{quote}” | [{src.title}]({src.url}) · {src.domain_type} · quality {src.quality_score} |")
    return "\n".join(lines)
