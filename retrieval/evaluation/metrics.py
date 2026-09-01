"""Retrieval metrics only; they do not estimate recommendation quality."""

from __future__ import annotations

from dataclasses import dataclass

from retrieval.models import RetrievalResult
from retrieval.search import Retriever


@dataclass(frozen=True)
class EvaluationCase:
    """A manually-curated query with relevant corpus standard and/or chunk IDs."""

    query: str
    relevant_standard_numbers: frozenset[str] = frozenset()
    relevant_chunk_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("Evaluation query must not be empty")
        if not self.relevant_standard_numbers and not self.relevant_chunk_ids:
            raise ValueError("An evaluation case needs at least one relevant standard or chunk ID")


@dataclass(frozen=True)
class EvaluationSummary:
    k: int
    recall_at_k: float
    precision_at_k: float
    mrr_at_k: float
    cases: int


def is_relevant(result: RetrievalResult, case: EvaluationCase) -> bool:
    return (
        result.chunk.chunk_id in case.relevant_chunk_ids
        or result.chunk.standard_number in case.relevant_standard_numbers
    )


def recall_at_k(results: tuple[RetrievalResult, ...] | list[RetrievalResult], case: EvaluationCase, k: int) -> float:
    """Whether any expected standard/chunk appears in the first K results (0 or 1)."""
    _validate_k(k)
    return float(any(is_relevant(result, case) for result in results[:k]))


def precision_at_k(results: tuple[RetrievalResult, ...] | list[RetrievalResult], case: EvaluationCase, k: int) -> float:
    """Relevant retrieved chunks divided by K; missing corpus slots count as non-relevant."""
    _validate_k(k)
    return sum(is_relevant(result, case) for result in results[:k]) / k


def mrr_at_k(results: tuple[RetrievalResult, ...] | list[RetrievalResult], case: EvaluationCase, k: int) -> float:
    """Reciprocal rank of the first relevant chunk, bounded to K."""
    _validate_k(k)
    for rank, result in enumerate(results[:k], start=1):
        if is_relevant(result, case):
            return 1.0 / rank
    return 0.0


def evaluate(retriever: Retriever, cases: tuple[EvaluationCase, ...] | list[EvaluationCase], k: int = 5) -> EvaluationSummary:
    """Evaluate a retriever against a manually curated gold set."""
    _validate_k(k)
    if not cases:
        raise ValueError("At least one evaluation case is required")
    result_lists = [retriever.search(case.query, top_k=k) for case in cases]
    return EvaluationSummary(
        k=k,
        recall_at_k=sum(recall_at_k(results, case, k) for results, case in zip(result_lists, cases, strict=True)) / len(cases),
        precision_at_k=sum(precision_at_k(results, case, k) for results, case in zip(result_lists, cases, strict=True)) / len(cases),
        mrr_at_k=sum(mrr_at_k(results, case, k) for results, case in zip(result_lists, cases, strict=True)) / len(cases),
        cases=len(cases),
    )


def _validate_k(k: int) -> None:
    if k < 1:
        raise ValueError("k must be positive")
