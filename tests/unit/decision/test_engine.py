from __future__ import annotations

from speccipher.decision.engine import DecisionEngine
from speccipher.domain.candidates import Candidate
from speccipher.domain.identity import StandardId
from speccipher.domain.models import DocumentChunk


class Catalog:
    def __init__(
        self,
        *,
        eligible: bool = True,
        stub: bool = False,
        status: str = "current",
        exclusions: str = "",
    ) -> None:
        self.eligible = eligible
        self.stub = stub
        self.status = status
        self.exclusions = exclusions

    def get_standard_by_code(self, code: str):
        return {
            "code": code,
            "title": "Concrete structures",
            "status": self.status,
            "is_eligible": int(self.eligible),
            "is_stub": int(self.stub),
        }

    def get_scope_text(self, standard_code: str):
        return {
            "standard_code": standard_code,
            "scope": "This standard covers concrete structures.",
            "exclusions": self.exclusions,
        }


def _candidate(
    chunk_id: str,
    *,
    eligible: bool = True,
    text: str = "This standard covers concrete structures and their design requirements.",
    rrf_score: float = 0.03,
) -> Candidate:
    return Candidate(
        chunk=DocumentChunk(
            chunk_id=chunk_id,
            standard_id=StandardId.parse("IS 456:2000"),
            logical_doc_id="pub_is456_2000",
            source_file_id="is456.pdf",
            absolute_page_start=10,
            absolute_page_end=11,
            section_number="1",
            section_title="Scope",
            text=text,
            is_eligible=eligible,
        ),
        dense_score=0.8,
        lexical_score=-1.2,
        rrf_score=rrf_score,
    )


def test_decision_engine_keeps_retrieval_scores_separate_and_assigns_high_confidence() -> None:
    engine = DecisionEngine(Catalog())

    decisions = engine.decide(
        "concrete structures",
        (_candidate("chunk-1"), _candidate("chunk-2")),
    )

    assert len(decisions) == 1
    decision = decisions[0]
    assert decision.confidence_level == "HIGH"
    assert decision.reason_codes == ("APPLICABILITY_CONFIRMED",)
    assert decision.applicability.overall > 0
    assert decision.rerank_score > 0
    assert decision.candidates[0].rrf_score == 0.03


def test_decision_engine_excludes_stubs_ineligible_and_explicitly_excluded_standards() -> None:
    candidates = (_candidate("chunk-1"),)

    assert DecisionEngine(Catalog(stub=True)).decide("concrete structures", candidates) == ()
    assert DecisionEngine(Catalog(eligible=False)).decide("concrete structures", candidates) == ()
    assert (
        DecisionEngine(Catalog(status="superseded")).decide("concrete structures", candidates) == ()
    )
    excluded = Catalog(exclusions="This standard does not apply to concrete structures.")
    assert DecisionEngine(excluded).decide("concrete structures", candidates) == ()
    excluded_run = DecisionEngine(excluded).analyze("concrete structures", candidates)
    assert "CONTRADICTORY_EVIDENCE" in excluded_run.reason_codes
    assert "NO_ELIGIBLE_MATCH" in excluded_run.reason_codes


def test_decision_engine_reports_superseded_and_unverified_identity_reasons() -> None:
    candidates = (_candidate("chunk-1"),)
    superseded = DecisionEngine(Catalog(status="superseded")).analyze(
        "concrete structures", candidates
    )
    assert "STATUS_SUPERSEDED" in superseded.reason_codes

    class MissingCatalog(Catalog):
        def get_standard_by_code(self, code: str):
            return None

    unverified = DecisionEngine(MissingCatalog()).analyze("concrete structures", candidates)
    assert "IDENTITY_UNVERIFIED" in unverified.reason_codes


def test_decision_engine_retains_one_chunk_as_medium_confidence() -> None:
    decisions = DecisionEngine(Catalog()).decide("concrete structures", (_candidate("chunk-1"),))

    assert decisions[0].confidence_level == "MEDIUM"
    assert decisions[0].reason_codes == ("INSUFFICIENT_EVIDENCE",)


def test_decision_engine_rejects_ineligible_evidence_chunks() -> None:
    decisions = DecisionEngine(Catalog()).decide(
        "concrete structures",
        (_candidate("chunk-1", eligible=False),),
    )

    assert decisions == ()


def test_single_rerank_orders_evidence_after_fusion() -> None:
    weaker = _candidate("rrf-first", text="concrete", rrf_score=0.04)
    stronger = _candidate("rerank-first", text="concrete structures", rrf_score=0.02)

    decisions = DecisionEngine(Catalog()).decide(
        "concrete structures",
        (weaker, stronger),
    )

    assert decisions[0].candidates[0].chunk.chunk_id == "rerank-first"


def test_explicit_standard_identity_restricts_candidates_to_requested_version() -> None:
    decisions = DecisionEngine(Catalog()).decide(
        "IS 875:2015 concrete structures",
        (_candidate("chunk-1"),),
    )

    assert decisions == ()


def test_configured_reranker_fails_open_to_token_overlap() -> None:
    class OfflineCrossEncoder:
        strategy = "bge_reranker_v2_m3"

        def score(self, requirement: str, candidates: tuple[Candidate, ...]):
            raise RuntimeError("model weights unavailable")

    decisions = DecisionEngine(Catalog(), reranker=OfflineCrossEncoder()).decide(
        "concrete structures",
        (_candidate("chunk-1"),),
    )

    assert len(decisions) == 1
    assert decisions[0].reranker_strategy == "token_overlap"
    assert decisions[0].warnings == ("bge_reranker_v2_m3 failed; token-overlap fallback used",)
