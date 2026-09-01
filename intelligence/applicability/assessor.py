"""Compute decomposed applicability only after a candidate passes constraints."""

from __future__ import annotations

from intelligence.config import ApplicabilityWeights
from intelligence.constraints import CandidateContext
from intelligence.models import ApplicabilityScore
from retrieval.models import RetrievalResult
from .feature_extraction import RequirementFeatures
from intelligence.text import tokenize
from .rules import required_term_coverage


class ApplicabilityAssessor:
    """Deterministic fit assessor; it neither recommends nor estimates confidence."""

    def __init__(self, weights: ApplicabilityWeights) -> None:
        self._weights = weights

    def assess(self, requirement: RequirementFeatures, candidate: RetrievalResult, context: CandidateContext) -> ApplicabilityScore:
        title_tokens = tokenize(context.metadata.title) if context.metadata and context.metadata.title else frozenset()
        chunk_tokens = tokenize(candidate.chunk.text)
        scope_tokens = tokenize(context.scope_text) if context.scope_text else chunk_tokens
        product = required_term_coverage(requirement.product_terms, title_tokens.union(chunk_tokens))
        material = required_term_coverage(requirement.material_terms, title_tokens.union(chunk_tokens, scope_tokens))
        application = required_term_coverage(requirement.application_terms, title_tokens.union(chunk_tokens, scope_tokens))
        scope = required_term_coverage(requirement.requirement_tokens, scope_tokens)
        requirement_match = required_term_coverage(requirement.requirement_tokens, chunk_tokens)
        weighted = (
            self._weights.product_match * product + self._weights.material_match * material
            + self._weights.application_match * application + self._weights.scope_match * scope
            + self._weights.requirement_match * requirement_match
        )
        total_weight = sum(vars(self._weights).values())
        return ApplicabilityScore(product, material, application, scope, requirement_match, weighted / total_weight)
