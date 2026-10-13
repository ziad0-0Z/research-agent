import pytest
from agents import AgentOutputSchema
from pydantic import ValidationError

from research_agent import models as m


def _sq(i, type_="technical"):
    return {
        "id": f"sq{i}",
        "question": "What is the latency difference between options?",
        "type": type_,
        "priority": 1,
        "success_criteria": "Concrete latency numbers from benchmarks",
        "search_queries": ["llm api latency benchmark"],
    }


def test_plan_valid():
    plan = m.ResearchPlan(
        main_question="Cloud APIs or self-hosting?",
        decision_context="5-person startup",
        subquestions=[_sq(1), _sq(2, "market"), _sq(3, "risk")],
    )
    assert plan.types() == {"technical", "market", "risk"}


def test_plan_rejects_duplicate_ids():
    with pytest.raises(ValidationError, match="unique"):
        m.ResearchPlan(main_question="Cloud APIs or self-hosting?", decision_context="startup",
                       subquestions=[_sq(1), _sq(1)])


def test_plan_needs_two_subquestions():
    with pytest.raises(ValidationError):
        m.ResearchPlan(main_question="Cloud APIs or self-hosting?", decision_context="startup",
                       subquestions=[_sq(1)])


def test_subquestion_bad_id_and_blank_queries():
    bad = _sq(1)
    bad["id"] = "question-1"
    with pytest.raises(ValidationError):
        m.SubQuestion(**bad)
    blank = _sq(1)
    blank["search_queries"] = ["  "]
    with pytest.raises(ValidationError):
        m.SubQuestion(**blank)


def test_extra_fields_forbidden():
    # The support-agent project broke when the model passed unexpected args.
    with pytest.raises(ValidationError):
        m.GuardrailVerdict(verdict="allow", category="valid_research", domain="technology",
                           reason="fine", confidence=0.9)


def test_finding_requires_evidence():
    with pytest.raises(ValidationError):
        m.Finding(statement="APIs are cheaper at low volume.", evidence_ids=[], strength="strong")


def test_critique_revise_requires_instructions():
    with pytest.raises(ValidationError):
        m.Critique(verdict="REVISE", unsupported_claims=["x"], missing_sources=[],
                   contradictions_ignored=[], revision_instructions=[])
    ok = m.Critique(verdict="PASS", unsupported_claims=[], missing_sources=[],
                    contradictions_ignored=[], revision_instructions=[])
    assert ok.verdict == "PASS"


def test_report_has_no_llm_confidence_field():
    assert "confidence" not in m.ResearchReport.model_fields


def test_orchestration_dedupes_specialists():
    d = m.OrchestrationDecision(specialists=["risk", "risk", "market"], depth="quick", rationale="only risk+cost")
    assert d.specialists == ["risk", "market"]


def test_content_hash_ignores_whitespace_and_case():
    assert m.content_hash("src1", "Hello   World") == m.content_hash("src1", "hello world")
    assert m.content_hash("src1", "a quote") != m.content_hash("src2", "a quote")


@pytest.mark.parametrize("cls", [
    m.GuardrailVerdict, m.ResearchPlan, m.OrchestrationDecision, m.ClaimExtraction,
    m.EvidenceAnalysis, m.ResearchReport, m.Critique, m.ClaimSupportVerdict,
])
def test_llm_models_produce_strict_schema(cls):
    schema = AgentOutputSchema(cls).json_schema()
    assert schema["type"] == "object"
