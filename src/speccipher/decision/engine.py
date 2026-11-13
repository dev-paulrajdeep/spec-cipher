"""Deterministic release-backed admissibility, applicability, and confidence policy."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace
from math import isfinite
from threading import RLock
from typing import Any, Protocol

from speccipher.decision.reranking import CandidateReranker, TokenOverlapReranker
from speccipher.domain.candidates import Candidate, RequirementFeatures, RerankScore
from speccipher.domain.identity import StandardId
from speccipher.domain.requirements import Requirement, tokenize

_EXCLUSION_RE = re.compile(r"\b(?:does\s+not\s+apply\s+to|excluding)\s+([^.;\n]+)", re.I)
_EXPLICIT_STANDARD_RE = re.compile(
    r"\b(?:ISO|IEC|IS|SP|EN|BS|IRC)\s*\d{1,6}"
    r"(?:\s*(?:\(\s*Part\s*\d+[A-Za-z]?\)|[-/]\d+[A-Za-z]?))?"
    r"(?:\s*:\s*\d{4})?\b",
    re.I,
)
_UNUSABLE_STATUSES = frozenset({"withdrawn", "superseded", "cancelled"})
_WEIGHTS = {
    "product": 0.25,
    "material": 0.15,
    "application": 0.20,
    "scope": 0.20,
    "requirement": 0.20,
}


class DecisionCatalog(Protocol):
    """Read-only catalog operations required by serving decisions."""

    def get_standard_by_code(self, code: str) -> dict[str, Any] | None: ...

    def get_scope_text(self, standard_code: str) -> dict[str, Any] | None: ...


@dataclass(frozen=True)
class ApplicabilityBreakdown:
    product: float
    material: float
    application: float
    scope: float
    requirement: float
    overall: float


@dataclass(frozen=True)
class Decision:
    """An admissible, evidence-backed retrieval candidate after the decision stages."""

    standard_code: str
    title: str
    status: str | None
    candidates: tuple[Candidate, ...]
    rerank_score: float
    rerank_raw_score: float
    reranker_strategy: str
    applicability: ApplicabilityBreakdown
    final_score: float
    confidence_level: str
    reason_codes: tuple[str, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class DecisionRun:
    """Primary decisions plus deterministic reasons for an abstained result."""

    decisions: tuple[Decision, ...]
    reason_codes: tuple[str, ...]


def _coverage(required: frozenset[str], available: frozenset[str]) -> float:
    return len(required & available) / len(required) if required else 0.0


def _excluded(requirement: RequirementFeatures, exclusions: str | None) -> bool:
    if not exclusions:
        return False
    requirement_terms = requirement.requirement_tokens
    return any(
        requirement_terms & tokenize(match.group(1)) for match in _EXCLUSION_RE.finditer(exclusions)
    )


def _applicability(
    features: RequirementFeatures,
    *,
    title: str,
    evidence_text: str,
    scope_text: str,
) -> ApplicabilityBreakdown:
    title_terms = tokenize(title)
    evidence_terms = tokenize(evidence_text)
    scope_terms = tokenize(scope_text)
    product = _coverage(features.product_terms, title_terms | evidence_terms)
    material = _coverage(features.material_terms, title_terms | evidence_terms | scope_terms)
    application = _coverage(features.application_terms, title_terms | evidence_terms | scope_terms)
    scope = _coverage(features.requirement_tokens, scope_terms)
    requirement = _coverage(features.requirement_tokens, evidence_terms)
    overall = sum(
        weight * value
        for weight, value in zip(
            _WEIGHTS.values(), (product, material, application, scope, requirement), strict=True
        )
    )
    return ApplicabilityBreakdown(product, material, application, scope, requirement, overall)


class DecisionEngine:
    """Apply conservative catalog constraints and deterministic evidence scoring."""

    def __init__(
        self,
        catalog: DecisionCatalog,
        *,
        minimum_applicability: float = 0.20,
        minimum_final_score: float = 0.20,
        evidence_limit: int = 3,
        rerank_weight: float = 0.35,
        reranker: CandidateReranker | None = None,
    ) -> None:
        if not 0.0 <= minimum_applicability <= 1.0:
            raise ValueError("minimum_applicability must be between 0 and 1")
        if not 0.0 <= minimum_final_score <= 1.0:
            raise ValueError("minimum_final_score must be between 0 and 1")
        if evidence_limit < 1:
            raise ValueError("evidence_limit must be positive")
        if not 0.0 <= rerank_weight <= 1.0:
            raise ValueError("rerank_weight must be between 0 and 1")
        self.catalog = catalog
        self.minimum_applicability = minimum_applicability
        self.minimum_final_score = minimum_final_score
        self.evidence_limit = evidence_limit
        self.rerank_weight = rerank_weight
        self.reranker = reranker or TokenOverlapReranker()
        self._reranker_lock = RLock()

    def decide(
        self, requirement_text: str, candidates: tuple[Candidate, ...]
    ) -> tuple[Decision, ...]:
        """Return eligible standard-level decisions in deterministic score order."""
        return self.analyze(requirement_text, candidates).decisions

    def analyze(self, requirement_text: str, candidates: tuple[Candidate, ...]) -> DecisionRun:
        """Return decisions and reason codes explaining when no primary is eligible."""
        requirement = Requirement.from_text(requirement_text)
        rejection_reasons: set[str] = set()
        warnings: tuple[str, ...] = ()
        with self._reranker_lock:
            active_reranker = self.reranker
            try:
                rerank_scores = active_reranker.score(requirement_text, candidates)
                expected_ids = {candidate.chunk.chunk_id for candidate in candidates}
                if set(rerank_scores) != expected_ids:
                    raise ValueError("reranker must return exactly one score per candidate")
                if any(
                    not isinstance(score, RerankScore)
                    or not isfinite(score.raw_score)
                    or not isfinite(score.normalized_score)
                    or not 0.0 <= score.normalized_score <= 1.0
                    for score in rerank_scores.values()
                ):
                    raise ValueError("reranker returned an invalid score")
            except Exception:
                reranker_name = active_reranker.strategy
                logging.getLogger(__name__).exception(
                    "Reranker %s failed; using deterministic token-overlap fallback",
                    reranker_name,
                )
                close = getattr(active_reranker, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        logging.getLogger(__name__).exception(
                            "Failed to release unavailable reranker %s", reranker_name
                        )
                active_reranker = TokenOverlapReranker()
                rerank_scores = active_reranker.score(requirement_text, candidates)
                warnings = (f"{reranker_name} failed; token-overlap fallback used",)
                self.reranker = active_reranker
            reranker_strategy = active_reranker.strategy
        explicit_codes: set[str] = set()
        for match in _EXPLICIT_STANDARD_RE.finditer(requirement_text):
            try:
                parsed_code = StandardId.parse(match.group()).code
                meta = self.catalog.get_standard_by_code(parsed_code)
                if meta and bool(meta.get("is_eligible", 0)) and not bool(meta.get("is_stub", 0)):
                    explicit_codes.add(parsed_code)
            except Exception:
                continue
        grouped: dict[str, list[Candidate]] = {}
        for candidate in candidates:
            code = candidate.chunk.standard_id.code if candidate.chunk.standard_id else None
            if code:
                grouped.setdefault(code, []).append(candidate)
            else:
                rejection_reasons.add("IDENTITY_UNVERIFIED")

        provisional: list[tuple[Decision, float]] = []
        for code, group in grouped.items():
            if explicit_codes and code not in explicit_codes:
                rejection_reasons.add("NO_ELIGIBLE_MATCH")
                continue
            metadata = self.catalog.get_standard_by_code(code)
            if not metadata or bool(metadata.get("is_stub", 0)):
                rejection_reasons.add("IDENTITY_UNVERIFIED")
                continue
            if not bool(metadata.get("is_eligible", 0)):
                rejection_reasons.add("NO_ELIGIBLE_MATCH")
                continue
            status = str(metadata["status"]).strip().lower() if metadata.get("status") else None
            if status in _UNUSABLE_STATUSES:
                rejection_reasons.add("STATUS_SUPERSEDED")
                continue
            scope_record = self.catalog.get_scope_text(code) or {}
            scope_text = str(scope_record.get("scope") or "")
            exclusions = str(scope_record.get("exclusions") or "")
            if _excluded(requirement.features, exclusions):
                rejection_reasons.add("CONTRADICTORY_EVIDENCE")
                continue

            admissible_chunks = tuple(
                candidate
                for candidate in group
                if candidate.chunk.is_eligible and candidate.chunk.chunk_id
            )
            if not admissible_chunks:
                rejection_reasons.add("INSUFFICIENT_EVIDENCE")
                continue
            reranked = sorted(
                admissible_chunks,
                key=lambda candidate: (
                    -rerank_scores[candidate.chunk.chunk_id].normalized_score,
                    -candidate.rrf_score,
                    candidate.chunk.chunk_id,
                ),
            )
            evidence = tuple(
                replace(
                    candidate,
                    rerank_score=rerank_scores[candidate.chunk.chunk_id].normalized_score,
                )
                for candidate in reranked[: self.evidence_limit]
            )
            title = str(metadata.get("title") or "")
            evidence_text = " ".join(candidate.chunk.text for candidate in evidence)
            applicability = _applicability(
                requirement.features,
                title=title,
                evidence_text=evidence_text,
                scope_text=scope_text,
            )
            if applicability.product <= 0 or applicability.overall < self.minimum_applicability:
                rejection_reasons.add("INSUFFICIENT_EVIDENCE")
                continue

            best_rerank: RerankScore = rerank_scores[evidence[0].chunk.chunk_id]
            rerank_score = best_rerank.normalized_score
            final_score = (
                self.rerank_weight * rerank_score
                + (1.0 - self.rerank_weight) * applicability.overall
            )
            if not isfinite(final_score) or final_score < self.minimum_final_score:
                rejection_reasons.add("INSUFFICIENT_EVIDENCE")
                continue
            provisional.append(
                (
                    Decision(
                        standard_code=code,
                        title=title,
                        status=status,
                        candidates=evidence,
                        rerank_score=rerank_score,
                        rerank_raw_score=best_rerank.raw_score,
                        reranker_strategy=reranker_strategy,
                        applicability=applicability,
                        final_score=final_score,
                        confidence_level="MEDIUM",
                        reason_codes=("INSUFFICIENT_EVIDENCE",),
                        warnings=warnings,
                    ),
                    final_score,
                )
            )

        provisional.sort(key=lambda item: (-item[1], item[0].standard_code))
        decisions: list[Decision] = []
        for index, (decision, score) in enumerate(provisional):
            next_score = provisional[index + 1][1] if index + 1 < len(provisional) else 0.0
            independent_chunks = {item.chunk.chunk_id for item in decision.candidates}
            if (
                decision.applicability.scope > 0
                and len(independent_chunks) >= 2
                and score - next_score > 0.15
            ):
                decision = replace(
                    decision,
                    confidence_level="HIGH",
                    reason_codes=("APPLICABILITY_CONFIRMED",),
                )
            decisions.append(decision)
        if not decisions:
            rejection_reasons.add("NO_ELIGIBLE_MATCH")
        return DecisionRun(tuple(decisions), tuple(sorted(rejection_reasons)))

    def close(self) -> None:
        """Release optional reranker weights when the serving container closes."""
        with self._reranker_lock:
            close = getattr(self.reranker, "close", None)
            if callable(close):
                close()
