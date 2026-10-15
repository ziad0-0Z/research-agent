"""Phase 11 — evidence graph: how one report's findings rest on evidence, and that evidence on sources.

Three columns, read left to right as the chain the system enforces:

    sources ──(quoted from)── evidence ──(cited by)── findings
                                 └── contradiction arcs between evidence the Evidence Analyst paired ──┘

Built only from what the run stored (report, evidence rows with their support level, sources, the analysis'
contradiction pairs). Pure Python → inline SVG: no graph library, no JavaScript, the same picture every time.
Line style carries the support level (solid / dashed / dotted) so the meaning never depends on colour alone.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, field

SUPPORT_STYLE = {  # support level → (css colour var, dash pattern, label)
    "supported": ("var(--ok)", "", "Supported"),
    "partial": ("var(--warn)", "6 4", "Partly supported"),
    "not_supported": ("var(--bad)", "2 4", "Not supported"),
    "unchecked": ("var(--slate-2)", "1 5", "Not judged"),
}


@dataclass
class Graph:
    findings: list[dict] = field(default_factory=list)   # {key, label, statement, strength, evidence: [ids]}
    evidence: list[dict] = field(default_factory=list)   # {id, claim, quote, support, source_id, memory}
    sources: list[dict] = field(default_factory=list)    # {id, title, domain_type, quality, host}
    contradictions: list[tuple[str, str, str]] = field(default_factory=list)  # (ev a, ev b, topic)
    uncited: int = 0
    contradictions_total: int = 0  # every pair the analyst reported, incl. ones between uncited evidence

    def stats(self) -> dict:
        src_of = {e["id"]: e["source_id"] for e in self.evidence}
        multi = sum(1 for f in self.findings if len({src_of.get(i) for i in f["evidence"] if i in src_of}) >= 2)
        return {"findings": len(self.findings), "evidence": len(self.evidence), "sources": len(self.sources),
                "contradictions": len(self.contradictions), "contradictions_total": self.contradictions_total,
                "multi_source_findings": multi, "uncited": self.uncited}


def build_graph(report: dict | None, evidence: dict[str, dict], sources: dict[str, dict],
                analysis: dict | None = None) -> Graph:
    g = Graph()
    if not report:
        return g
    order: dict[str, int] = {}
    for i, f in enumerate(report.get("findings", []), 1):
        ids = [e for e in f.get("evidence_ids", []) if e in evidence]
        g.findings.append({"key": f"F{i}", "statement": f.get("statement", ""), "strength": f.get("strength", ""),
                           "evidence": ids})
        for e in ids:
            order.setdefault(e, i)
    # evidence in the order findings first cite it: fewer crossing lines
    for eid in sorted(order, key=lambda x: (order[x], _num(x))):
        e = evidence[eid]
        g.evidence.append({"id": eid, "claim": e.get("claim", ""), "quote": e.get("quote", ""),
                           "support": e.get("support") or "unchecked", "source_id": e.get("source_id"),
                           "memory": bool(e.get("memory"))})
    g.uncited = len(set(evidence) - set(order))
    pos = {e["id"]: i for i, e in enumerate(g.evidence)}
    by_source: dict[str, list[int]] = {}
    for e in g.evidence:
        by_source.setdefault(e["source_id"], []).append(pos[e["id"]])
    for sid in sorted(by_source, key=lambda s: sum(by_source[s]) / len(by_source[s])):  # barycentre order
        s = sources.get(sid) or {}
        g.sources.append({"id": sid, "title": s.get("title") or sid, "domain_type": s.get("domain_type") or "",
                          "quality": s.get("quality_score"), "url": s.get("url") or "", "origin": s.get("origin") or ""})
    seen = set()
    g.contradictions_total = len((analysis or {}).get("contradictions", []))
    for c in (analysis or {}).get("contradictions", []):
        for a in c.get("evidence_ids_a", []):
            for b in c.get("evidence_ids_b", []):
                key = tuple(sorted((a, b)))
                if a in pos and b in pos and a != b and key not in seen:
                    seen.add(key)
                    g.contradictions.append((a, b, c.get("topic", "")))
    return g


def _num(eid: str) -> int:
    m = re.search(r"\d+", eid)
    return int(m.group()) if m else 0


def _e(x) -> str:
    return html.escape(str(x), quote=True)


def _cut(text: str, n: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _wrap(text: str, width: int, lines: int) -> list[str]:
    words, out, cur = " ".join((text or "").split()).split(" "), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            out.append(cur)
            cur = w
            if len(out) == lines:
                break
        else:
            cur = f"{cur} {w}".strip()
    else:
        if cur:
            out.append(cur)
    if len(out) == lines and " ".join(out) != " ".join(words):
        out[-1] = _cut(out[-1] + " …", width)
    return out[:lines]


# column geometry (viewBox units; the SVG scales to the container width)
W = 1000
SX, SW = 8, 250        # sources
EX, EW = 372, 112      # evidence
FX, FW = 612, 380      # findings
ROW_S, ROW_E, ROW_F = 52, 34, 64
PAD = 34


def render_svg(g: Graph) -> str:
    if not g.findings:
        return ""
    n_s, n_e, n_f = len(g.sources), len(g.evidence), len(g.findings)
    height = max(n_s * ROW_S, n_e * ROW_E, n_f * ROW_F) + 2 * PAD

    def ys(n: int, row: int) -> list[float]:
        span = height - 2 * PAD
        step = span / n if n else 0
        return [PAD + step * (i + .5) for i in range(n)]

    sy = dict(zip((s["id"] for s in g.sources), ys(n_s, ROW_S)))
    ey = dict(zip((e["id"] for e in g.evidence), ys(n_e, ROW_E)))
    fy = dict(zip((f["key"] for f in g.findings), ys(n_f, ROW_F)))
    sup = {e["id"]: e["support"] for e in g.evidence}
    parts = [f'<svg class="rp-graph" viewBox="0 0 {W} {height:.0f}" role="img" '
             f'aria-label="Evidence graph: {n_f} findings cite {n_e} evidence items quoted from {n_s} sources'
             f'{f", {len(g.contradictions)} contradictions" if g.contradictions else ""}.">']
    # column titles
    for x, t in ((SX, "Sources"), (EX, "Evidence"), (FX, "Findings")):
        parts.append(f'<text x="{x}" y="16" class="rp-g-col">{t}</text>')
    # edges: source → evidence
    for e in g.evidence:
        y1, y2 = sy.get(e["source_id"]), ey[e["id"]]
        if y1 is None:
            continue
        x1, x2 = SX + SW, EX
        parts.append(f'<path class="rp-g-src" data-ev="{_e(e["id"])}" d="M{x1},{y1:.1f} C{x1 + 60},{y1:.1f} {x2 - 60},{y2:.1f} {x2},{y2:.1f}"/>')
    # edges: evidence → finding (styled by the evidence's judged support)
    for f in g.findings:
        for eid in f["evidence"]:
            color, dash, label = SUPPORT_STYLE.get(sup.get(eid, "unchecked"), SUPPORT_STYLE["unchecked"])
            y1, y2 = ey[eid], fy[f["key"]]
            x1, x2 = EX + EW, FX
            d = f' stroke-dasharray="{dash}"' if dash else ""
            parts.append(f'<path class="rp-g-cite" style="stroke:{color}"{d} d="M{x1},{y1:.1f} C{x1 + 55},{y1:.1f} '
                         f'{x2 - 55},{y2:.1f} {x2},{y2:.1f}"><title>{_e(eid)} → {_e(f["key"])}: {label}</title></path>')
    # contradiction arcs, on the left edge of the evidence column
    for a, b, topic in g.contradictions:
        y1, y2 = ey[a], ey[b]
        bulge = min(70, 18 + abs(y2 - y1) * .25)
        parts.append(f'<path class="rp-g-contra" d="M{EX},{y1:.1f} C{EX - bulge},{y1:.1f} {EX - bulge},{y2:.1f} {EX},{y2:.1f}">'
                     f'<title>Contradiction ({_e(topic)}): {_e(a)} vs {_e(b)}</title></path>')
    # nodes: sources
    for s in g.sources:
        y = sy[s["id"]]
        q = s["quality"]
        qw = (SW - 24) * (q or 0)
        meta = " · ".join(x for x in (s["domain_type"], f"quality {q:.2f}" if q is not None else "") if x)
        parts.append(f'<g class="rp-g-node rp-g-s"><title>{_e(s["title"])}\n{_e(s["url"])}</title>'
                     f'<rect x="{SX}" y="{y - 21:.1f}" width="{SW}" height="42" rx="7"/>'
                     f'<text x="{SX + 12}" y="{y - 4:.1f}" class="rp-g-t">{_e(_cut(s["title"], 36))}</text>'
                     f'<text x="{SX + 12}" y="{y + 11:.1f}" class="rp-g-m">{_e(meta)}</text>'
                     f'<rect x="{SX + 12}" y="{y + 15:.1f}" width="{qw:.1f}" height="2" class="rp-g-q"/></g>')
    # nodes: evidence (square: a quote Python verified against the page)
    for e in g.evidence:
        y = ey[e["id"]]
        color, _, label = SUPPORT_STYLE.get(e["support"], SUPPORT_STYLE["unchecked"])
        mem = " · memory" if e["memory"] else ""
        parts.append(f'<g class="rp-g-node rp-g-e"><title>{_e(e["id"])} — {label}{mem}\nClaim: {_e(e["claim"])}\n'
                     f'Quote: “{_e(e["quote"])}”</title>'
                     f'<rect x="{EX}" y="{y - 13:.1f}" width="{EW}" height="26" rx="5"/>'
                     f'<rect x="{EX}" y="{y - 13:.1f}" width="5" height="26" rx="2" style="fill:{color}"/>'
                     f'<text x="{EX + 13}" y="{y + 4:.1f}" class="rp-g-id">{_e(e["id"])}</text>'
                     f'<text x="{EX + EW - 8}" y="{y + 4:.1f}" class="rp-g-m" text-anchor="end">'
                     f'{_e({"supported": "sup", "partial": "part", "not_supported": "no", "unchecked": "—"}.get(e["support"], ""))}'
                     f'{" ◆" if e["memory"] else ""}</text></g>')
    # nodes: findings
    for f in g.findings:
        y = fy[f["key"]]
        lines = _wrap(f["statement"], 54, 2)
        txt = "".join(f'<text x="{FX + 46}" y="{y - 6 + i * 15 - (len(lines) - 1) * 7.5 + 4:.1f}" class="rp-g-t">{_e(l)}</text>'
                      for i, l in enumerate(lines))
        parts.append(f'<g class="rp-g-node rp-g-f"><title>{_e(f["key"])} ({_e(f["strength"])} evidence): {_e(f["statement"])}</title>'
                     f'<rect x="{FX}" y="{y - 26:.1f}" width="{FW}" height="52" rx="8"/>'
                     f'<text x="{FX + 12}" y="{y + 4:.1f}" class="rp-g-key">{_e(f["key"])}</text>{txt}</g>')
    parts.append("</svg>")
    return "".join(parts)


def legend() -> str:
    items = "".join(
        f'<span><svg width="28" height="8" aria-hidden="true"><line x1="0" y1="4" x2="28" y2="4" style="stroke:{c}" '
        f'stroke-width="2.4"{f" stroke-dasharray={chr(34)}{d}{chr(34)}" if d else ""}/></svg>{label}</span>'
        for c, d, label in SUPPORT_STYLE.values())
    return (f'<div class="rp-g-legend">{items}<span><svg width="28" height="8" aria-hidden="true"><line x1="0" y1="4" '
            f'x2="28" y2="4" class="rp-g-contra"/></svg>Contradiction</span><span>◆ reused from memory</span></div>')


def graph_html(report: dict | None, evidence: dict, sources: dict, analysis: dict | None) -> str:
    """The whole tab body: summary numbers, the graph, a legend, and a plain list for screen readers."""
    from . import components as c
    g = build_graph(report, evidence, sources, analysis)
    if not g.findings:
        return c.empty("No graph for this run", "The run produced no report findings to connect to evidence.")
    st = g.stats()
    summary = c.stats([
        c.stat("Findings", st["findings"]),
        c.stat("Cited evidence", st["evidence"], f"{st['uncited']} saved but not cited" if st["uncited"] else "all saved evidence cited"),
        c.stat("Sources", st["sources"]),
        c.stat("Backed by 2+ sources", f"{st['multi_source_findings']}/{st['findings']}", "findings citing quotes from different pages"),
        c.stat("Contradictions", st["contradictions_total"],
               (f"{st['contradictions']} drawn between cited evidence" if st["contradictions_total"] else
                "none found by the Evidence Analyst"), "warn" if st["contradictions_total"] else ""),
    ], "rp-stats-graph")
    text = "".join(f"<li><b>{_e(f['key'])}</b> {_e(_cut(f['statement'], 140))} — cites "
                   f"{_e(', '.join(f['evidence']) or 'nothing')}</li>" for f in g.findings)
    return (summary + f'<div class="rp-graph-wrap">{render_svg(g)}</div>' + legend() +
            '<p class="rp-caption">Hover a node or line for the full claim, quote or source. Lines into a finding are '
            'styled by how the Critic judged that quote against the finding\'s claim.</p>'
            f'<details class="rp-details"><summary>The same graph as a list</summary><ul class="rp-list">{text}</ul></details>')
