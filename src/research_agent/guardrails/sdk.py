"""The input check wired into the Agents SDK as an `input_guardrail` on the Planner.

Defence in depth: the pipeline runs the full input guardrail (regex + Guard Agent) before the
Planner. This SDK guardrail runs again inside `Runner.run(planner, ...)` (blocking, before the
model is called) and trips if
  - the pipeline's input check did not run or did not allow this run, or
  - the text reaching the Planner fails the regex pre-filter.
So the Planner cannot be reached with an unchecked or attacking input, even by calling it directly.
It never calls an LLM.
"""
from __future__ import annotations

from typing import Any

from agents import Agent, GuardrailFunctionOutput, RunContextWrapper, input_guardrail

from .input import prefilter


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(c.get("text", "") for c in content if isinstance(c, dict))
    return ""


def _text(value: Any) -> str:
    """The text to check: the input string, or - when a session prepends earlier turns - only the
    newest user message. Earlier turns were checked when they were new; assistant turns are our own
    released output, and checking the whole history would also trip the length limit."""
    if isinstance(value, str):
        return value
    items = [i for i in (value or []) if isinstance(i, dict)]
    users = [i for i in items if i.get("role") == "user"]
    if users:
        return _content_text(users[-1].get("content"))
    return "\n".join(_content_text(i.get("content")) for i in items)


def planner_input_check(context: Any, text: str) -> GuardrailFunctionOutput:
    check = getattr(context, "input_check", None)
    if check is None or not check.allowed:
        return GuardrailFunctionOutput(output_info={"reason": "input guardrail did not allow this run"},
                                       tripwire_triggered=True)
    blocked = prefilter(text)
    if blocked is not None:
        return GuardrailFunctionOutput(output_info=blocked.summary(), tripwire_triggered=True)
    return GuardrailFunctionOutput(output_info={"reason": "ok"}, tripwire_triggered=False)


@input_guardrail(name="Input guardrail (code)", run_in_parallel=False)
async def planner_input_guardrail(ctx: RunContextWrapper[Any], agent: Agent, input: Any) -> GuardrailFunctionOutput:
    return planner_input_check(ctx.context, _text(input))
