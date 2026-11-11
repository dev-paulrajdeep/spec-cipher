from __future__ import annotations

from corpus.models import DocumentChunk
from intelligence.config import IntelligenceConfig
from intelligence.pipeline import IntelligenceEngine
from intelligence.standards import canonical_is_code
from retrieval.models import RetrievalResult


def result(identifier: str, standard: str | None, score: float = 0.9) -> RetrievalResult:
    return RetrievalResult(
        DocumentChunk(
            identifier, standard, f"{identifier}.pdf", 1, 1, "1", "Scope",
            "fire safety requirements residential buildings",
        ),
        score,
        1,
    )


def test_is_code_normalization_preserves_part_and_year() -> None:
    assert canonical_is_code("IS456") == "IS 456"
    assert canonical_is_code("IS 456 : 2000") == "IS 456:2000"
    assert canonical_is_code("IS 875 (Part 3):2015") == "IS 875 (Part 3):2015"
    assert canonical_is_code("building standard") is None


def test_recommendation_id_accepts_independently_indexed_document_numbers() -> None:
    from intelligence.standards import canonical_recommendation_id

    assert canonical_recommendation_id("BOOK-18-ELECTRONICS-AND") == "BOOK-18-ELECTRONICS-AND"
    assert canonical_recommendation_id("IS 456:2000") == "IS 456:2000"
    assert canonical_recommendation_id("building standard") is None


def test_engine_returns_at_most_five_unique_explicit_standards() -> None:
    candidates = tuple(
        result(f"chunk-{index}", f"IS {index}:2024")
        for index in range(1, 8)
    ) + (result("duplicate", "IS456:2000"), result("duplicate-2", "IS 456 : 2000"))
    run = IntelligenceEngine(
        config=IntelligenceConfig(minimum_final_score=0.0),
    ).recommend("fire safety requirements for residential buildings", candidates)

    codes = [item.standard_number for item in run.recommendations]
    assert len(codes) <= 5
    assert len(codes) == len(set(codes))
    assert all(code.startswith("IS ") for code in codes)
    assert [item.final_score for item in run.recommendations] == sorted(
        (item.final_score for item in run.recommendations), reverse=True
    )


def test_engine_rejects_unknown_and_weak_standards() -> None:
    engine = IntelligenceEngine()
    run = engine.recommend(
        "fire safety requirements for residential buildings",
        (result("unknown", None), result("weak", "Building standard")),
    )
    assert run.recommendations == ()
