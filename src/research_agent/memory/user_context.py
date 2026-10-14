"""User-stated context: only values the user wrote in the question, with the exact words.

Stored and shown (scripts/memory.py context, the UI). Never sent to any agent: an old statement
("we are 5 people") must not silently shape a new question. No inference: if the user did not
write it, it is not stored.
"""
from __future__ import annotations

import re

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("team_size", re.compile(r"\b(\d{1,4})[- ](?:person|people|member|developer|engineer)s?\b(?:[- ]team)?", re.I)),
    ("team_size", re.compile(r"\bteam of (\d{1,4})\b", re.I)),
    # A budget only when the user says so ("budget", "can spend", "afford"): a price in the question
    # ("is $0.50 per million tokens cheap?") is not their budget.
    ("budget", re.compile(r"\b(?:budget(?: of| is|:)?|(?:can|could) (?:only )?spend(?: up to| at most)?|"
                          r"afford(?: up to)?|spend(?: up to| at most))\s+([$€£]\s?\d[\d,.]*\s?[km]?"
                          r"(?:\s?(?:per|/|a)\s?(?:month|year|mo|yr))?)", re.I)),
    ("industry", re.compile(r"\bour (bank|hospital|clinic|law firm|startup|agency|school|university|insurer)\b",
                            re.I)),
]


def extract_stated_context(question: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for key, rx in PATTERNS:
        for m in rx.finditer(question or ""):
            span = m.group(0).strip()
            value = (m.group(1) if m.groups() and m.group(1) else span).lower()
            if (key, value) in seen:
                continue
            seen.add((key, value))
            out.append({"key": key, "value": value, "span": span})
    return out
