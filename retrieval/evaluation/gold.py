"""Small hand-curated smoke-evaluation set for candidate retrieval."""

from __future__ import annotations

from .metrics import EvaluationCase


DEMO_GOLD_CASES: tuple[EvaluationCase, ...] = (
    EvaluationCase(
        query="industrial pump construction requirements",
        relevant_standard_numbers=frozenset({"IS 5120"}),
    ),
    EvaluationCase(
        query="pump performance testing and hydrostatic certificates",
        relevant_standard_numbers=frozenset({"IS 5120"}),
    ),
    EvaluationCase(
        query="centrifugal pumps for water supply",
        relevant_standard_numbers=frozenset({"IS 5120"}),
    ),
    EvaluationCase(
        query="electrical compatibility of industrial pumps",
        relevant_standard_numbers=frozenset({"IS 5120"}),
    ),
    EvaluationCase(
        query="piping and installation requirements for pumps",
        relevant_standard_numbers=frozenset({"IS 5120"}),
    ),
)
