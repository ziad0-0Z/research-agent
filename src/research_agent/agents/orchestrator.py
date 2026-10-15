"""Research Orchestrator (LLM-orchestrated): decides which specialists run,
which subquestions each one gets, and how deep to research.

Python then validates the decision (unknown ids dropped, unassigned
subquestions given to the matching specialist) and runs the workers in
parallel - the LLM plans, code executes.
"""
from __future__ import annotations

from functools import lru_cache

from agents import Agent

from ..llm import get_model, precise_settings
from ..models import OrchestrationDecision

ORCHESTRATOR_INSTRUCTIONS = """You coordinate a research team with three specialists:
- technical: performance, architecture, model quality, engineering effort
- market: costs, pricing, vendors, adoption
- risk: privacy, security, compliance, lock-in, operational failures

You receive a research plan. Decide:
- assignments: which specialists to use and which subquestion ids each researches.
  Every subquestion id must be assigned exactly once. Usually follow each subquestion's type,
  but you may give a subquestion to another specialist if it fits better.
  Use only the specialists that are needed; a narrow plan may need just one.
- depth: quick (2 sources per subquestion), standard (3), deep (4). Use standard unless the
  plan is very narrow (quick) or the decision is high-stakes and contested (deep).
- rationale: one or two sentences explaining your choices.

Use only subquestion ids that appear in the plan."""


@lru_cache(maxsize=1)
def orchestrator() -> Agent:
    return Agent(name="Research Orchestrator", instructions=ORCHESTRATOR_INSTRUCTIONS, model=get_model(),
                 model_settings=precise_settings(), output_type=OrchestrationDecision)
