"""Small hand-curated smoke-evaluation set for candidate retrieval."""

from __future__ import annotations

from .metrics import EvaluationCase


DEMO_GOLD_CASES: tuple[EvaluationCase, ...] = (
    EvaluationCase(
        query="waterproofing compound for concrete reservoir",
        relevant_standard_numbers=frozenset({"IS 2645:2003"}),
    ),
    EvaluationCase(
        query="concrete structures for storage of liquids",
        relevant_standard_numbers=frozenset({"IS 3370 (Part 1):2021"}),
    ),
)
