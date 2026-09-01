"""Compose binary constraints while retaining every rejection reason."""

from __future__ import annotations

from dataclasses import dataclass

from corpus.models import StandardMetadata
from intelligence.models import ConstraintResult
from retrieval.models import RetrievalResult
from .scope import scope_reasons
from .status import status_reasons
from .version import version_reasons


@dataclass(frozen=True)
class CandidateContext:
    metadata: StandardMetadata | None = None
    scope_text: str | None = None


class ConstraintEvaluator:
    """Hard-filter candidates before costly reranking; no candidate is silently removed."""

    def evaluate(self, requirement: str, candidate: RetrievalResult, context: CandidateContext) -> ConstraintResult:
        reasons = (
            *status_reasons(context.metadata),
            *version_reasons(candidate.chunk.standard_number, requirement),
            *scope_reasons(requirement, context.scope_text),
        )
        return ConstraintResult(admissible=not reasons, reasons=tuple(reasons))
