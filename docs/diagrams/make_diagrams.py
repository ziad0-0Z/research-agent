"""Draw the workflow diagrams in docs/diagrams/*.svg.

Plain SVG, no dependencies: run `python docs/diagrams/make_diagrams.py` after changing a layout.
Colour key: pink pill = input/output, green = model call (agent), grey = Python decision,
blue = tool or store, purple = human.
"""
from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).parent
FONT = "Inter, 'Segoe UI', -apple-system, Helvetica, Arial, sans-serif"

STYLE = {
    "io":    ("#FCEDE9", "#EDBFB3", "#C2593F"),
    "llm":   ("#EFF6EA", "#C9DFBC", "#5B9440"),
    "code":  ("#F4F3EE", "#D3CFC4", "#55514A"),
    "tool":  ("#EBF2FA", "#C3D6EC", "#3F6E9C"),
    "human": ("#F1EEFB", "#CFC6EF", "#6650B8"),
    "stop":  ("#FCEDE9", "#EDBFB3", "#C2593F"),
}
LINE = "#A8A39B"
MUTED = "#8A857C"


class Svg:
    def __init__(self, w: int, h: int, title: str):
        self.w, self.h, self.title = w, h, title
        self.parts: list[str] = []
        self.nodes: dict[str, tuple[float, float, float, float]] = {}

    # ---- shapes -------------------------------------------------------------------------------------
    def node(self, key: str, cx: float, cy: float, label: str, kind: str = "llm", w: float | None = None,
             h: float = 40, sub: str | None = None) -> None:
        fill, stroke, text = STYLE[kind]
        lines = label.split("\n")
        width = w or max(96, 9.0 * max(len(s) for s in lines) + 40, 6.6 * len(sub or '') + 32)
        height = h if not sub and len(lines) == 1 else h + 16 * (len(lines) - 1) + (14 if sub else 0)
        x, y = cx - width / 2, cy - height / 2
        rx = height / 2 if kind in ("io", "stop") else 3
        dash = ' stroke-dasharray="4 3"' if kind == "code" else ""
        self.parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" rx="{rx:.1f}" '
                          f'fill="{fill}" stroke="{stroke}" stroke-width="1.2"{dash}/>')
        n = len(lines) + (1 if sub else 0)
        top = cy - (n - 1) * 8
        for i, s in enumerate(lines):
            self.parts.append(f'<text x="{cx:.1f}" y="{top + 16 * i + 5:.1f}" text-anchor="middle" font-size="14.5" '
                              f'font-weight="600" fill="{text}">{esc(s)}</text>')
        if sub:
            self.parts.append(f'<text x="{cx:.1f}" y="{top + 16 * len(lines) + 4:.1f}" text-anchor="middle" '
                              f'font-size="11.5" fill="{text}" opacity="0.8">{esc(sub)}</text>')
        self.nodes[key] = (x, y, width, height)

    def text(self, x: float, y: float, s: str, size: float = 12.5, color: str = MUTED, anchor: str = "middle",
             weight: str = "400", italic: bool = False) -> None:
        style = ' font-style="italic"' if italic else ""
        self.parts.append(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="{size}" '
                          f'font-weight="{weight}" fill="{color}"{style}>{esc(s)}</text>')

    def box(self, x: float, y: float, w: float, h: float, label: str | None = None) -> None:
        """A light frame around a group of nodes."""
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="none" stroke="#E4E1D9" '
                          f'stroke-width="1.2"/>')
        if label:
            self.text(x + 12, y + 18, label, 11.5, MUTED, "start", "600")

    # ---- edges --------------------------------------------------------------------------------------
    def anchor(self, key: str, side: str) -> tuple[float, float]:
        x, y, w, h = self.nodes[key]
        return {"l": (x, y + h / 2), "r": (x + w, y + h / 2), "t": (x + w / 2, y),
                "b": (x + w / 2, y + h)}[side]

    def arrow(self, a: str, b: str, sa: str = "r", sb: str = "l", dashed: bool = False, label: str | None = None,
              curve: float = 0.0, gap: float = 12, label_dy: float = -8, color: str = LINE,
              label_color: str = MUTED, label_dx: float = 0) -> None:
        x1, y1 = self.anchor(a, sa)
        x2, y2 = self.anchor(b, sb)
        self.path(x1, y1, x2, y2, dashed, label, curve, gap, label_dy, color, label_color, label_dx)

    def path(self, x1, y1, x2, y2, dashed=False, label=None, curve=0.0, gap=12, label_dy=-8, color=LINE,
             label_color=MUTED, label_dx=0.0):
        dx, dy = x2 - x1, y2 - y1
        length = max((dx * dx + dy * dy) ** 0.5, 1)
        ux, uy = dx / length, dy / length
        x1, y1, x2, y2 = x1 + ux * gap, y1 + uy * gap, x2 - ux * gap, y2 - uy * gap
        dash = ' stroke-dasharray="3 4"' if dashed else ""
        if curve:
            mx, my = (x1 + x2) / 2 - uy * curve, (y1 + y2) / 2 + ux * curve
            d = f"M{x1:.1f},{y1:.1f} Q{mx:.1f},{my:.1f} {x2:.1f},{y2:.1f}"
            lx, ly = (x1 + 2 * mx + x2) / 4, (y1 + 2 * my + y2) / 4
        else:
            d = f"M{x1:.1f},{y1:.1f} L{x2:.1f},{y2:.1f}"
            lx, ly = (x1 + x2) / 2, (y1 + y2) / 2
        self.parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.4"{dash} '
                          f'marker-end="url(#head)"/>')
        if label:
            self.text(lx + label_dx, ly + label_dy, label, 12, label_color)

    def poly(self, points: list[tuple[float, float]], dashed=False, label=None, label_at=None, color=LINE):
        dash = ' stroke-dasharray="3 4"' if dashed else ""
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in points)
        self.parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.4"{dash} '
                          f'stroke-linejoin="round" marker-end="url(#head)"/>')
        if label and label_at:
            self.text(label_at[0], label_at[1], label, 12)

    def legend(self, kinds: list[str], y: float | None = None) -> None:
        names = {"io": "input / output", "llm": "agent (model call)", "code": "Python decision",
                 "tool": "tool / store", "human": "human"}
        y = y or self.h - 22
        x = 24
        for k in kinds:
            fill, stroke, _ = STYLE[k]
            dash = ' stroke-dasharray="4 3"' if k == "code" else ""
            rx = 7 if k == "io" else 2
            self.parts.append(f'<rect x="{x}" y="{y - 10}" width="22" height="14" rx="{rx}" fill="{fill}" '
                              f'stroke="{stroke}"{dash}/>')
            self.text(x + 30, y + 1, names[k], 11.5, MUTED, "start")
            x += 30 + 7.0 * len(names[k]) + 26

    def save(self, name: str) -> None:
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
                f'height="{self.h}" font-family="{FONT}" role="img" aria-label="{esc(self.title)}">\n'
                f'<title>{esc(self.title)}</title>\n'
                '<defs><marker id="head" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="9" markerHeight="9" '
                'orient="auto-start-reverse"><path d="M1,1 L8.5,5 L1,9" fill="none" stroke="' + LINE +
                '" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/></marker></defs>\n'
                f'<rect width="{self.w}" height="{self.h}" fill="#FFFFFF"/>\n')
        (OUT / f"{name}.svg").write_text(head + "\n".join(self.parts) + "\n</svg>\n", encoding="utf-8")


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


# =====================================================================================================
def overview() -> None:
    s = Svg(1810, 560, "Research agent: the whole workflow")
    y = 200
    s.node("in", 50, y, "In", "io", w=64)
    s.node("guard", 210, y, "Input guardrail", "code", sub="regex + Guard Agent")
    s.node("router", 405, y, "Router", "code", sub="sensitive domain?")
    s.node("planner", 590, y, "Planner", "llm", sub="3-6 subquestions")
    s.node("orch", 785, y, "Orchestrator", "llm", sub="who, how deep")
    s.node("tech", 1000, y - 85, "Technical", "llm", w=120)
    s.node("market", 1000, y, "Market", "llm", w=120)
    s.node("risk", 1000, y + 85, "Risk", "llm", w=120)
    s.node("analyst", 1200, y, "Evidence\nAnalyst", "llm")
    s.node("writer", 1365, y, "Writer", "llm", w=100)
    s.node("critic", 1365, y + 140, "Critic", "llm", w=110, sub="+ code checks")
    s.node("release", 1580, y, "Release gate", "code", w=200, sub="guardrail · token · impact")
    s.node("out", 1760, y, "Out", "io", w=64)
    s.node("human", 1580, y + 150, "Human review", "human")
    s.node("blocked", 210, y + 125, "Blocked", "stop", w=96)
    s.node("tools", 1000, 430, "search · scrape · extract claims · verify quote · save evidence", "tool", w=480)
    s.node("memory", 590, 430, "Long-term memory", "tool")

    s.arrow("in", "guard")
    s.arrow("guard", "router")
    s.arrow("guard", "blocked", "b", "t", label="attack / off-scope", label_dy=4, label_dx=58)
    s.arrow("router", "planner")
    s.arrow("planner", "orch")
    for k in ("tech", "market", "risk"):
        curve = {"tech": -18, "market": 0, "risk": 18}[k]
        s.arrow("orch", k, dashed=True, curve=curve)
        s.arrow(k, "analyst", dashed=True, curve=-curve)
    s.arrow("analyst", "writer")
    s.arrow("writer", "critic", "b", "t", curve=22, label="draft", label_dx=-26, label_dy=4)
    s.arrow("critic", "writer", "t", "b", curve=22, label="revise", label_dx=32, label_dy=4)
    s.arrow("writer", "release", label="PASS", label_dy=-10)
    s.arrow("release", "out")
    s.arrow("release", "human", "b", "t", dashed=True, label="needs review", label_dx=46, label_dy=4)
    s.arrow("human", "out", "r", "b", dashed=True, curve=20, label="approve", label_dx=34, label_dy=6)
    s.arrow("risk", "tools", "b", "t", dashed=True, label="each specialist", label_dx=56, label_dy=4)
    s.arrow("memory", "orch", "r", "b", dashed=True, curve=22)
    s.text(745, 360, "re-verified evidence", 12, anchor="end")
    s.text(1000, 70, "parallel specialists", 12, MUTED, weight="600")
    s.text(1365, 410, "max 3 rounds", 12)
    s.legend(["io", "llm", "code", "tool", "human"])
    s.save("overview")


def chaining() -> None:
    s = Svg(1420, 300, "Prompt chaining: plan → evidence → analysis → report")
    y = 130
    s.node("in", 50, y, "In", "io", w=64)
    s.node("plan", 205, y, "Planner", "llm", sub="ResearchPlan")
    s.node("gate", 415, y, "Gate", "code", sub="assignments valid?")
    s.node("res", 640, y, "Specialists", "llm", sub="verified Evidence")
    s.node("ana", 880, y, "Evidence Analyst", "llm", sub="EvidenceAnalysis")
    s.node("wr", 1120, y, "Writer", "llm", sub="ResearchReport")
    s.node("out", 1300, y, "Out", "io", w=64)
    s.node("fix", 415, y + 95, "Fix in code", "stop", w=120)
    s.arrow("in", "plan")
    s.arrow("plan", "gate", label="Output 1")
    s.arrow("gate", "res", label="Pass")
    s.arrow("gate", "fix", "b", "t", dashed=True, label="Fail", label_dx=18, label_dy=4)
    s.arrow("res", "ana", label="Output 2")
    s.arrow("ana", "wr", label="Output 3")
    s.arrow("wr", "out")
    s.text(710, 40, "Each step is a typed Pydantic output, stored and shown in the UI before the next step runs.",
           12.5)
    s.legend(["io", "llm", "code"])
    s.save("prompt-chaining")


def routing() -> None:
    s = Svg(1200, 400, "Routing: code routes the question, the model routes subquestions")
    y = 170
    s.node("in", 50, y, "In", "io", w=64)
    s.node("guard", 200, y, "Guard Agent", "llm", sub="verdict + domain")
    s.node("router", 400, y, "Code router", "code", sub="keywords + Guard domain")
    s.node("normal", 640, y - 70, "Normal path", "llm", sub="release if every check passes")
    s.node("sens", 640, y + 70, "Review path", "llm", sub="requires_review = True")
    s.node("out", 1060, y - 70, "Out", "io", w=64)
    s.node("human", 880, y + 70, "Human review", "human")
    s.node("out2", 1060, y + 70, "Out", "io", w=64)
    s.arrow("in", "guard")
    s.arrow("guard", "router")
    s.arrow("router", "normal", dashed=True, curve=-14, label="general", label_dy=-14, label_dx=-24)
    s.arrow("router", "sens", dashed=True, curve=14, label="financial · medical · legal", label_dy=26,
            label_dx=-70)
    s.arrow("normal", "out")
    s.arrow("sens", "human")
    s.arrow("human", "out2", label="approve")
    s.text(600, 330, "Inside each path the Planner types every subquestion (technical / market / risk) and the "
                     "Orchestrator sends it to that specialist.", 12.5)
    s.text(600, 350, "Python checks the result (validate_assignments): every subquestion assigned exactly once.",
           12.5)
    s.legend(["io", "llm", "code", "human"])
    s.save("routing")


def parallel() -> None:
    s = Svg(1200, 380, "Parallelization: three specialists at once, failures isolated")
    y = 165
    s.node("in", 50, y, "In", "io", w=64)
    s.node("orch", 210, y, "Orchestrator", "llm")
    s.node("t", 470, y - 85, "Technical researcher", "llm")
    s.node("m", 470, y, "Market researcher", "llm")
    s.node("r", 470, y + 85, "Risk researcher", "llm")
    s.node("agg", 760, y, "Aggregator", "code", sub="success · partial · failed · timeout")
    s.node("out", 1000, y, "Out", "io", w=64)
    s.arrow("in", "orch")
    for k, c in (("t", -18), ("m", 0), ("r", 18)):
        s.arrow("orch", k, dashed=True, curve=c)
        s.arrow(k, "agg", dashed=True, curve=-c)
    s.arrow("agg", "out")
    s.text(470, 300, "asyncio.gather · per-worker timeout · one retry · shared rate limiter", 12.5)
    s.text(760, 235, "a failed worker → gap written into limitations", 12)
    s.legend(["io", "llm", "code"])
    s.save("parallelization")


def orchestrator() -> None:
    s = Svg(1200, 380, "Orchestrator-workers: the model picks the workers")
    y = 165
    s.node("in", 50, y, "In", "io", w=64)
    s.node("plan", 200, y, "Planner", "llm")
    s.node("orch", 420, y, "Orchestrator", "llm", sub="which specialists, how deep")
    s.node("t", 680, y - 85, "Technical", "llm", w=130)
    s.node("m", 680, y, "Market", "llm", w=130)
    s.node("r", 680, y + 85, "Risk", "llm", w=130)
    s.node("syn", 910, y, "Synthesizer", "llm", sub="Evidence Analyst")
    s.node("out", 1100, y, "Out", "io", w=64)
    s.arrow("in", "plan")
    s.arrow("plan", "orch")
    for k, c in (("t", -18), ("m", 0), ("r", 18)):
        s.arrow("orch", k, dashed=True, curve=c)
        s.arrow(k, "syn", dashed=True, curve=-c)
    s.arrow("syn", "out")
    s.text(600, 305, "“What are the security risks of self-hosting an LLM?” → only the Risk "
                     "specialist runs.", 12.5, italic=True)
    s.text(600, 325, "The choice is the model's; Python only checks it is valid.", 12.5)
    s.legend(["io", "llm"])
    s.save("orchestrator-workers")


def evaluator() -> None:
    s = Svg(1200, 360, "Evaluator-optimizer: Writer and Critic, verdict decided in code")
    y = 150
    s.node("in", 50, y, "In", "io", w=64)
    s.node("wr", 250, y, "Writer", "llm", sub="generator")
    s.node("cr", 560, y, "Critic", "llm", sub="evaluator · RAG retrieval")
    s.node("v", 800, y, "Verdict", "code", sub="PASS · REVISE · FAIL")
    s.node("out", 1020, y, "Out", "io", w=64)
    s.node("hum", 800, y + 105, "Human review", "human")
    s.arrow("in", "wr")
    s.arrow("wr", "cr", label="Draft report", curve=-30, label_dy=-6)
    s.arrow("cr", "wr", "l", "r", label="Rejected + feedback", curve=-30, label_dy=20, gap=14)
    s.arrow("cr", "v")
    s.arrow("v", "out", label="PASS")
    s.arrow("v", "hum", "b", "t", dashed=True, label="FAIL", label_dx=18, label_dy=4)
    s.poly([(800, y - 22 - 10), (800, y - 70), (250, y - 70), (250, y - 22 - 10)], dashed=True,
           label="REVISE (max 3 rounds, stops early when nothing improves)", label_at=(525, y - 78))
    s.legend(["io", "llm", "code", "human"])
    s.save("evaluator-optimizer")


def augmented() -> None:
    s = Svg(1000, 380, "The augmented researcher: one specialist and its tools")
    s.node("in", 70, 170, "In", "io", w=64)
    s.node("llm", 420, 170, "Specialist", "llm", w=160, h=56)
    s.node("out", 760, 170, "Out", "io", w=64)
    s.node("search", 230, 60, "search_web", "tool")
    s.node("scrape", 420, 60, "scrape_page", "tool")
    s.node("extract", 610, 60, "extract_claims", "tool")
    s.node("save", 300, 290, "save_evidence", "tool")
    s.node("verify", 540, 290, "verify quote", "code", sub="fuzzy match on page text")
    s.arrow("in", "llm")
    s.arrow("llm", "out", label="WorkerFindings")
    for k in ("search", "scrape", "extract"):
        s.arrow("llm", k, "t", "b", dashed=True, curve={"search": 14, "scrape": 0, "extract": -14}[k])
    s.arrow("llm", "save", "b", "t", dashed=True, curve=-10)
    s.arrow("save", "verify")
    s.text(420, 22, "Tavily → DuckDuckGo → offline corpus · SSRF-safe fetch · injected text wrapped as untrusted",
           12)
    s.text(650, 316, "unverified quote → rejected", 12, anchor="start")
    s.legend(["io", "llm", "code", "tool"])
    s.save("augmented-researcher")


def release() -> None:
    s = Svg(1480, 360, "Release: nothing reaches the user without passing Python checks")
    y = 140
    s.node("in", 50, y, "In", "io", w=64, sub=None)
    s.text(50, y + 36, "final draft", 11.5)
    s.node("og", 255, y, "Output guardrail", "code", sub="citations · quotes · URLs · claims")
    s.node("conf", 520, y, "Confidence", "code", sub="computed, not asked")
    s.node("tok", 745, y, "Approval token", "code", sub="bound to report hash")
    s.node("gate", 985, y, "Impact gate", "code", sub="sensitive · high impact · < 0.6")
    s.node("gen", 1235, y, "generate_report", "tool")
    s.node("out", 1410, y, "Out", "io", w=64)
    s.node("rej", 255, y + 115, "Rejected", "stop", w=110)
    s.node("hum", 985, y + 115, "Human review", "human", sub="approve · reject · request changes")
    s.arrow("in", "og")
    s.arrow("og", "conf", label="Pass")
    s.arrow("og", "rej", "b", "t", dashed=True, label="Fail", label_dx=16, label_dy=4)
    s.arrow("conf", "tok")
    s.arrow("tok", "gate")
    s.arrow("gate", "gen", label="clear")
    s.arrow("gen", "out")
    s.arrow("gate", "hum", "b", "t", dashed=True, label="needs review", label_dx=44, label_dy=4)
    s.arrow("hum", "gen", "r", "b", dashed=True, curve=-20, label="approve", label_dx=30, label_dy=12)
    s.legend(["io", "code", "tool", "human"])
    s.save("release")


def memory() -> None:
    s = Svg(1200, 340, "Long-term memory: evidence, not conclusions")
    y = 130
    s.node("q", 60, y, "In", "io", w=64)
    s.node("look", 250, y, "Memory lookup", "code", sub="IDF word gate + embeddings")
    s.node("ttl", 485, y, "Fresh?", "code", sub="TTL by claim type")
    s.node("save", 720, y, "save_evidence", "tool", sub="quote verified again")
    s.node("run", 950, y, "Normal research", "llm", sub="covered → quick depth")
    s.node("out", 1130, y, "Out", "io", w=64)
    s.node("refetch", 485, y + 115, "Re-fetch + verify", "tool")
    s.node("commit", 950, y + 115, "Commit", "code", sub="completed runs only")
    s.arrow("q", "look")
    s.arrow("look", "ttl")
    s.arrow("ttl", "save", label="fresh")
    s.arrow("ttl", "refetch", "b", "t", dashed=True, label="stale", label_dx=18, label_dy=4)
    s.arrow("refetch", "save", "r", "b", dashed=True, curve=-20)
    s.arrow("save", "run")
    s.arrow("run", "out")
    s.arrow("run", "commit", "b", "t", dashed=True, label="after release", label_dx=46, label_dy=4)
    s.text(600, 300, "pricing 30 days · market/risk 90 days · technical 180 days · newer contradicting evidence "
                     "supersedes old", 12)
    s.legend(["io", "llm", "code", "tool"], y=330)
    s.save("memory")


if __name__ == "__main__":
    for fn in (overview, chaining, routing, parallel, orchestrator, evaluator, augmented, release, memory):
        fn()
    print("diagrams written to", OUT)
