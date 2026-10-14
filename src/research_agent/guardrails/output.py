"""Output guardrail (pure Python): the last gate before a report can be released.

The Writer ⇄ Critic loop *improves* the report; this guardrail only *decides* whether the exact
report in hand may be released. It never edits the report and never calls an LLM. If every check
passes, Python issues an approval token bound to this report's content hash; generate_report
refuses to run without it.

Checks (all blocking):
  schema          re-validates ResearchReport (strict, extra="forbid")
  structure       findings present; failed/timed-out research disclosed in limitations
  citations       every cited id exists, its quote was verified and still matches the stored page
  support         no finding cites evidence the Critic judged not_supported;
                  partial-only support is never "strong"
  unconfirmed     the report doesn't claim actions no tool confirmed (human approval, tests run...)
  links           every URL in the report text belongs to a saved source
  injection       no injected instructions leaked from web pages into the report text
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import ValidationError

from ..context import AppContext
from ..models import ResearchReport, WorkerFindings
from ..tools.evidence import check_citation
from .injection import detect_injection


@dataclass
class OutputIssue:
    check: str
    message: str
    finding: int | None = None


@dataclass
class OutputCheck:
    passed: bool
    issues: list[OutputIssue] = field(default_factory=list)
    checks_run: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {"passed": self.passed, "checks_run": self.checks_run,
                "issues": [{"check": i.check, "finding": i.finding, "message": i.message} for i in self.issues]}


# Claims of actions this system never performs, or only performs through a confirming tool.
UNCONFIRMED_ACTIONS: list[tuple[str, re.Pattern[str], str | None]] = [
    ("human_approval", re.compile(
        r"\b(approved|reviewed|signed off|validated)\s+by\s+(a\s+|the\s+|our\s+)?"
        r"(human|reviewer|expert|compliance|legal|lawyer|auditor|team)\b", re.I), "approved_review"),
    ("own_experiment", re.compile(
        r"\bwe\s+(have\s+)?(tested|benchmarked|deployed|measured|ran|interviewed|surveyed|audited)\b", re.I), None),
]
_URL = re.compile(r"https?://[^\s)\]>\"']+", re.I)


def _report_texts(report: ResearchReport) -> list[str]:
    return ([report.title, report.executive_summary, report.recommendation] + [f.statement for f in report.findings]
            + list(report.contradictions) + list(report.limitations))


def check_output(ctx: AppContext, report: ResearchReport | None,
                 workers: list[WorkerFindings] | None = None) -> OutputCheck:
    out = OutputCheck(passed=False)
    issues = out.issues

    # 1. schema
    out.checks_run.append("schema")
    if report is None:
        issues.append(OutputIssue("schema", "no report to check"))
        return out
    try:
        report = ResearchReport.model_validate(report.model_dump())
    except ValidationError as exc:
        issues.append(OutputIssue("schema", f"report does not match the schema: {exc.errors()[0]['msg']}"))
        return out

    # 2. structure
    out.checks_run.append("structure")
    failed = [w for w in workers or [] if w.status in ("failed", "timeout")]
    if failed and not report.limitations:
        names = ", ".join(w.specialist for w in failed)
        issues.append(OutputIssue("structure", f"research by {names} failed but the report lists no limitations"))

    # 3. citations + 4. support
    out.checks_run += ["citations", "support"]
    for n, f in enumerate(report.findings, 1):
        levels = []
        for eid in f.evidence_ids:
            det = check_citation(ctx, eid)
            if not det["valid"]:
                issues.append(OutputIssue("citations", f"finding {n} cites {eid}: {det['reason']}", n))
                continue
            levels.append(det["evidence"].support)
            if det["evidence"].support == "not_supported":
                issues.append(OutputIssue("support", f"finding {n} cites {eid}, judged not to support the claim", n))
        if levels and all(lv == "partial" for lv in levels) and f.strength == "strong":
            issues.append(OutputIssue("support", f"finding {n} is 'strong' but its evidence only partly supports it", n))

    texts = _report_texts(report)
    joined = "\n".join(texts)

    # 5. claims of unconfirmed actions
    out.checks_run.append("unconfirmed_actions")
    review = ctx.store.get_review(ctx.run_id)
    approved = review is not None and review.status == "approved"
    for name, rx, requires in UNCONFIRMED_ACTIONS:
        m = rx.search(joined)
        if m and not (requires == "approved_review" and approved):
            issues.append(OutputIssue("unconfirmed_actions",
                                      f"report claims {m.group(0)!r}, but no tool confirmed it ({name})"))

    # 6. links must come from saved sources
    out.checks_run.append("links")
    known = {str(s.url).rstrip("/") for s in ctx.store.list_sources(ctx.run_id)}
    for url in sorted(set(_URL.findall(joined))):
        if url.rstrip("/.,;") not in known:
            issues.append(OutputIssue("links", f"URL {url!r} is not one of the saved sources"))

    # 7. no injected instructions leaked into the report
    out.checks_run.append("injection")
    leak = detect_injection(joined)
    if leak.detected:
        issues.append(OutputIssue("injection", f"report contains instruction-like text ({', '.join(leak.patterns)})"))

    out.passed = not issues
    return out
