from __future__ import annotations

import pytest

from corpus.models import DocumentChunk, StandardMetadata, StandardRelationship
from intelligence.applicability import ApplicabilityAssessor, DeterministicRequirementInterpreter
from intelligence.config import ApplicabilityWeights, IntelligenceConfig
from intelligence.constraints import CandidateContext, ConstraintEvaluator
from intelligence.evidence import EvidenceLocator, format_evidence
from intelligence.models import ConstraintResult, ScoredCandidate
from intelligence.pipeline import IntelligenceEngine
from intelligence.relationships import CorpusKnowledgeBase, RelationshipGraph, StandardContext
from intelligence.reranking import TokenOverlapReranker
from retrieval.models import RetrievalResult


def chunk(identifier: str, standard: str, text: str, page: int = 1) -> DocumentChunk:
    return DocumentChunk(identifier, standard, f"{standard}.pdf", page, page, "1", "Scope", text)


def result(identifier: str, standard: str, text: str, rank: int, score: float = 0.5) -> RetrievalResult:
    return RetrievalResult(chunk(identifier, standard, text, rank), score, rank)


@pytest.fixture
def primary_metadata() -> StandardMetadata:
    return StandardMetadata("IS 2645:2003", "Waterproofing compound for concrete", 2003, None, None, "current", "IS2645.pdf", 4)


@pytest.fixture
def knowledge(primary_metadata) -> CorpusKnowledgeBase:
    reference = StandardRelationship("IS 2645:2003", "IS 3370 (Part 1):2021", "references", 2, "IS 3370 is a reference.")
    return CorpusKnowledgeBase(
        {"IS 2645:2003": StandardContext(primary_metadata, "This scope covers waterproofing compound for concrete reservoirs.")},
        (reference,),
    )


def test_constraints_keep_rejected_candidates_and_reasons(primary_metadata) -> None:
    evaluator = ConstraintEvaluator()
    withdrawn = StandardMetadata("IS 2:2020", "Old standard", 2020, None, None, "withdrawn", "old.pdf", 1)
    candidate = result("old", "IS 2:2020", "old material", 1)
    status = evaluator.evaluate("material", candidate, CandidateContext(withdrawn, None))
    assert not status.admissible
    assert status.reasons[0].code == "unusable_status"
    mismatch = evaluator.evaluate("Use IS 2645:2003", candidate, CandidateContext(primary_metadata, None))
    assert any(reason.code == "explicit_standard_mismatch" for reason in mismatch.reasons)
    excluded = evaluator.evaluate(
        "timber reservoir waterproofing", result("primary", "IS 2645:2003", "compound", 1),
        CandidateContext(primary_metadata, "This standard does not apply to timber structures."),
    )
    assert any(reason.code == "explicit_scope_exclusion" for reason in excluded.reasons)


def test_reranker_is_deterministic_and_not_applicability() -> None:
    candidates = (
        result("weak", "IS 1:2020", "steel bridge", 1),
        result("strong", "IS 2:2020", "waterproofing compound concrete", 2),
    )
    first = TokenOverlapReranker().rerank("waterproofing concrete", candidates)
    second = TokenOverlapReranker().rerank("waterproofing concrete", candidates)
    assert first == second
    assert first[0].candidate.chunk.chunk_id == "strong"
    assert first[0].result.strategy == "token_overlap"


def test_applicability_is_decomposed_without_metadata_alignment(primary_metadata) -> None:
    candidate = result("primary", "IS 2645:2003", "waterproofing compound for cement concrete reservoir", 1)
    features = DeterministicRequirementInterpreter().interpret("waterproofing compound for concrete reservoir")
    score = ApplicabilityAssessor(ApplicabilityWeights()).assess(
        features, candidate, CandidateContext(primary_metadata, "Scope covers waterproofing compound for concrete reservoir."),
    )
    assert score.product_match == score.material_match == score.application_match == 1.0
    assert score.scope_match == score.requirement_match == 1.0
    assert score.overall == 1.0
    assert not hasattr(score, "metadata_alignment")


def test_evidence_selection_preserves_all_provenance() -> None:
    candidate = result("evidence", "IS 2645:2003", "Direct source wording", 1)
    scored = ScoredCandidate(candidate, ConstraintResult(True), final_score=0.8)
    evidence = EvidenceLocator().select((scored,), 1)[0]
    assert evidence.chunk_id == "evidence"
    assert evidence.source_file == "IS 2645:2003.pdf"
    assert evidence.page_start == evidence.page_end == 1
    assert "chunk evidence" in format_evidence(evidence)


def test_relationship_graph_queries_explicit_facts_and_reverse_views() -> None:
    fact = StandardRelationship("IS 1:2020", "IS 2:2021", "references", 3, "IS 2 is referenced.")
    graph = RelationshipGraph((fact,))
    direct = graph.traverse("IS 1:2020")
    reverse = graph.traverse("IS 2:2021")
    assert direct[0].relationship_type == "references"
    assert reverse[0].relationship_type == "referenced_by"
    assert direct[0].evidence_text == "IS 2 is referenced."


def test_engine_end_to_end_orders_recommendations_and_retains_rejections(knowledge) -> None:
    candidates = (
        result("primary", "IS 2645:2003", "waterproofing compound for concrete reservoir", 2, 0.80),
        result("support", "IS 2645:2003", "compound concrete scope requirement", 3, 0.70),
        result("wrong-version", "IS 9999:2020", "waterproofing compound concrete reservoir", 1, 0.95),
    )
    run = IntelligenceEngine(knowledge_base=knowledge).recommend("Use IS 2645:2003 for waterproofing concrete reservoir", candidates)
    assert len(run.recommendations) == 1
    recommendation = run.recommendations[0]
    assert recommendation.standard_number == "IS 2645:2003"
    assert recommendation.rank == 1
    assert len(recommendation.evidence) == 2
    assert recommendation.related_standards[0].standard_number == "IS 3370 (Part 1):2021"
    assert any(not candidate.constraint_result.admissible for candidate in run.candidates)
    assert {"constraints", "reranking", "applicability", "relationships", "evidence"} <= run.metrics.stages.keys()


def test_empty_candidates_and_reranker_failure_are_handled(knowledge) -> None:
    empty = IntelligenceEngine(knowledge_base=knowledge).recommend("waterproofing", ())
    assert empty.recommendations == ()

    class FailingReranker:
        def rerank(self, requirement, candidates):
            raise RuntimeError("model unavailable")

    run = IntelligenceEngine(knowledge_base=knowledge, reranker=FailingReranker()).recommend(
        "waterproofing concrete", (result("primary", "IS 2645:2003", "waterproofing concrete", 1),)
    )
    assert run.recommendations
    assert "fallback" in run.recommendations[0].warnings[0]


def test_engine_rejects_candidates_with_zero_applicability(knowledge) -> None:
    candidate = result("unrelated", "IS 2645:2003", "unrelated source wording", 1)

    run = IntelligenceEngine(knowledge_base=knowledge).recommend(
        "lunar geology and medieval music", (candidate,)
    )

    assert run.recommendations == ()
    assert len(run.candidates) == 1
    assert run.candidates[0].applicability is not None
    assert run.candidates[0].applicability.overall == 0.0


def test_engine_rejects_candidates_without_product_alignment(knowledge) -> None:
    candidate = result(
        "building",
        "IS 2645:2003",
        "industrial water service requirements",
        1,
    )

    run = IntelligenceEngine(knowledge_base=knowledge).recommend(
        "general-purpose centrifugal pumps for industrial water service",
        (candidate,),
    )

    assert run.recommendations == ()
    assert run.candidates[0].applicability is not None
    assert run.candidates[0].applicability.product_match == 0.0


def test_engine_rejects_generic_requirement_even_when_a_chunk_is_retrieved(knowledge) -> None:
    run = IntelligenceEngine(knowledge_base=knowledge).recommend(
        "industrial equipment requirements", (result("candidate", "IS 2645:2003", "industrial equipment requirements", 1),)
    )
    assert run.recommendations == ()
    assert run.candidates[0].constraint_result.reasons[0].code == "insufficient_requirement_specificity"
