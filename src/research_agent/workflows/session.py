"""Session memory (R10): follow-up questions inside one conversation, with the SDK's SQLiteSession.

    run 1  "Should a 5-person startup use cloud APIs or self-host?"   → report
    run 2  "Dig deeper into the cost side"                            → the Planner sees run 1

How it is used:
- The Planner runs with `session=` (Runner.run), so the SDK prepends the earlier turns and saves this one.
- After release, code adds one assistant message per run: the released report's title, summary and
  recommendation, or only the status when nothing was released (an unreleased draft never enters history).
- Blocked questions are never added: an attack does not become conversation context.
- The Guard sees the previous question as context, so a short follow-up is judged as part of a research
  conversation. The follow-up itself still goes through the full input guardrail.

Long-term memory (memory/) is different: it keeps verified evidence across conversations and days.
"""
from __future__ import annotations

import json
from typing import Any

from agents import SQLiteSession

from ..config import Settings
from ..models import ResearchReport

TURN_PREFIX = "[research run "


def open_session(settings: Settings, session_id: str) -> SQLiteSession:
    return SQLiteSession(session_id, str(settings.sessions_db_path))


async def previous_turns(session: SQLiteSession | None) -> list[dict[str, Any]]:
    """The runs recorded in this conversation (from our own assistant messages), oldest first."""
    if session is None:
        return []
    turns = []
    for item in await session.get_items():
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if item.get("role") == "assistant" and isinstance(content, str) and content.startswith(TURN_PREFIX):
            try:
                turns.append(json.loads(content.split("\n", 1)[1]))
            except (IndexError, json.JSONDecodeError):
                continue
    return turns


async def record_turn(session: SQLiteSession | None, *, run_id: str, question: str, main_question: str,
                      status: str, report: ResearchReport | None) -> None:
    if session is None:
        return
    turn: dict[str, Any] = {"run_id": run_id, "question": question, "main_question": main_question,
                            "status": status}
    if status == "completed" and report is not None:
        turn.update(title=report.title, executive_summary=report.executive_summary,
                    recommendation=report.recommendation)
    else:
        turn["note"] = "No report was released for this run."
    await session.add_items([{"role": "assistant",
                              "content": f"{TURN_PREFIX}{run_id}: {status}]\n{json.dumps(turn, ensure_ascii=False)}"}])


def follow_up_prompt(question: str, turns: list[dict[str, Any]]) -> str:
    """Planner input for a follow-up. The earlier turns arrive through the session; this says how to use them."""
    if not turns:
        return question
    last = turns[-1]
    return (f"{question}\n\n(This is a follow-up in an ongoing research conversation. The previous decision "
            f"question was: {last.get('main_question') or last.get('question')!r}. Resolve references such as "
            "'it', 'that' or 'the cost side' using the earlier turns, and write main_question as a full, "
            "standalone question.)")
