import pytest

from research_agent.models import Evidence, HumanReview, Source
from research_agent.store import DuplicateEvidenceError, Store, UnknownSourceError


@pytest.fixture
def store():
    s = Store(":memory:")
    s.create_run("run_1", "Cloud APIs or self-hosting?")
    s.add_source(Source(id="src1", url="https://example.com/a", title="A", origin="offline"),
                 text="Self-hosting requires a GPU with at least 24 GB of VRAM.")
    yield s
    s.close()


def _ev(i, quote="Self-hosting requires a GPU with at least 24 GB of VRAM.", source="src1"):
    return Evidence.build(evidence_id=f"ev{i}", run_id="run_1", subquestion_id="sq1",
                          claim="Self-hosting needs a 24 GB GPU.", quote=quote,
                          source_id=source, verified=True)


def test_evidence_roundtrip(store):
    store.add_evidence(_ev(1))
    got = store.get_evidence("ev1")
    assert got and got.verified and got.source_id == "src1"
    assert [e.id for e in store.list_evidence("run_1")] == ["ev1"]


def test_quote_verified_and_claim_support_are_separate(store):
    # A real quote (verified=True) can still fail to support the claim.
    store.add_evidence(_ev(1))
    assert store.get_evidence("ev1").support == "unchecked"
    store.set_support("ev1", "not_supported")
    ev = store.get_evidence("ev1")
    assert ev.verified is True and ev.support == "not_supported"
    with pytest.raises(ValueError):
        store.set_support("ev1", "maybe")


def test_unknown_source_rejected(store):
    with pytest.raises(UnknownSourceError):
        store.add_evidence(_ev(1, source="src99"))


def test_duplicate_quote_rejected(store):
    store.add_evidence(_ev(1))
    with pytest.raises(DuplicateEvidenceError):
        store.add_evidence(_ev(2, quote="self-hosting requires a GPU with at least 24 GB   of VRAM."))


def test_source_text_kept_for_quote_verification(store):
    assert "24 GB" in store.get_source_text("src1")


def test_run_state_update(store):
    store.update_run("run_1", status="paused_for_review", state={"stage": "review"})
    row = store.get_run("run_1")
    assert row["status"] == "paused_for_review"


def test_review_queue(store):
    store.upsert_review(HumanReview(run_id="run_1", reasons=["financial domain"]))
    assert [r.run_id for r in store.pending_reviews()] == ["run_1"]
    review = store.get_review("run_1")
    review.status = "approved"
    store.upsert_review(review)
    assert store.pending_reviews() == []
