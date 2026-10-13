"""Quote verification: does this quote really appear in the source text?

Check 1 of 2 in grounding (check 2, whether the quote supports the claim, is
an LLM judgment in analysis.py). Pure Python, so the LLM cannot bypass it.
"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

MATCH_THRESHOLD = 0.90
MIN_QUOTE_WORDS = 4

_QUOTES = {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "−": "-"}
_WS = re.compile(r"\s+")
_PUNCT_EDGE = re.compile(r"^[\s\"'.,;:…-]+|[\s\"'.,;:…-]+$")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    for bad, good in _QUOTES.items():
        text = text.replace(bad, good)
    text = text.replace("...", " ").replace("…", " ")
    return _WS.sub(" ", text).strip().lower()


def verify_quote(quote: str, source_text: str, threshold: float = MATCH_THRESHOLD) -> tuple[bool, float]:
    """Return (verified, best_similarity).

    Exact match after normalisation passes immediately; otherwise the quote is
    compared against every window of the same word length (tolerates small
    differences such as punctuation or a changed word).
    """
    q = _PUNCT_EDGE.sub("", normalize(quote))
    src = normalize(source_text)
    q_words = q.split()
    if len(q_words) < MIN_QUOTE_WORDS:
        return False, 0.0
    if q in src:
        return True, 1.0

    src_words = src.split()
    n = len(q_words)
    if n > len(src_words):
        return False, 0.0
    best = 0.0
    # Cheap pre-filter: windows must share the quote's first or last word.
    anchors = {q_words[0], q_words[-1]}
    for i in range(0, len(src_words) - n + 1):
        window = src_words[i:i + n]
        if window[0] not in anchors and window[-1] not in anchors:
            continue
        ratio = SequenceMatcher(None, q, " ".join(window)).ratio()
        if ratio > best:
            best = ratio
            if best >= 0.99:
                break
    return best >= threshold, round(best, 3)
