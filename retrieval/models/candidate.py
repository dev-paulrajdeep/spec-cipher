"""Non-destructive standard-level views over chunk-level retrieval evidence."""

from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict

from .result import RetrievalResult


@dataclass(frozen=True)
class StandardCandidate:
    """All returned chunks for one standard; no applicability judgement is implied."""

    standard_number: str
    results: tuple[RetrievalResult, ...]

    @property
    def best_similarity_score(self) -> float:
        return max(result.similarity_score for result in self.results)


def group_results_by_standard(results: list[RetrievalResult] | tuple[RetrievalResult, ...]) -> tuple[StandardCandidate, ...]:
    """Group without discarding chunks; candidate order follows first retrieved evidence."""
    grouped: OrderedDict[str, list[RetrievalResult]] = OrderedDict()
    for result in results:
        # Corpus validation normally disallows this; retain an explicit label if
        # callers constructed a raw chunk outside that contract.
        key = result.chunk.standard_number or "<unknown standard>"
        grouped.setdefault(key, []).append(result)
    return tuple(StandardCandidate(number, tuple(items)) for number, items in grouped.items())
