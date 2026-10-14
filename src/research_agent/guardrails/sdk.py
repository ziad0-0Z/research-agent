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


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    parts = []
    for item in value or []:
        content = item.get("content") if isinstance(item, dict) else None
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts += [c.get("text", "") for c in content if isinstance(c, dict)]
    return "\n".join(parts)


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
