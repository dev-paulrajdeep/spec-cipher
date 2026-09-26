"""Decision pipeline: constraints -> rerank -> applicability -> facts -> evidence."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
import logging
import re
from time import perf_counter

from intelligence.applicability import ApplicabilityAssessor, DeterministicRequirementInterpreter, RequirementInterpreter
from intelligence.config import IntelligenceConfig
from intelligence.constraints import ConstraintEvaluator
from intelligence.evidence import EvidenceLocator
from intelligence.instrumentation import PipelineMetrics
from intelligence.models import Recommendation, ScoredCandidate
from intelligence.relationships import CorpusKnowledgeBase
from intelligence.reranking import CrossEncoderReranker, Reranker, TokenOverlapReranker
from intelligence.standards import canonical_is_code, canonical_recommendation_id
from retrieval.models import RetrievalResult

logger = logging.getLogger(__name__)


class IntelligenceRun:
    """Recommendations plus retained rejected/intermediate candidates for diagnostics."""

    def __init__(self, recommendations: tuple[Recommendation, ...], candidates: tuple[ScoredCandidate, ...], metrics: PipelineMetrics) -> None:
        self.recommendations = recommendations
        self.candidates = candidates
        self.metrics = metrics


class IntelligenceEngine:
    """Explainable decision engine with replaceable interpreter and reranker boundaries."""

    def __init__(
        self,
        config: IntelligenceConfig | None = None,
        *,
        knowledge_base: CorpusKnowledgeBase | None = None,
        interpreter: RequirementInterpreter | None = None,
        reranker: Reranker | None = None,
        constraint_evaluator: ConstraintEvaluator | None = None,
        assessor: ApplicabilityAssessor | None = None,
        evidence_locator: EvidenceLocator | None = None,
    ) -> None:
        self.config = config or IntelligenceConfig()
        if self.config.enable_optional_llm_interpreter and interpreter is None:
            raise ValueError("An enabled optional LLM interpreter must be injected explicitly")
        self.knowledge_base = knowledge_base or CorpusKnowledgeBase()
        self.interpreter = interpreter or DeterministicRequirementInterpreter()
        self.reranker = reranker or self._configured_reranker()
        self.constraint_evaluator = constraint_evaluator or ConstraintEvaluator()
        self.assessor = assessor or ApplicabilityAssessor(self.config.applicability_weights)
        self.evidence_locator = evidence_locator or EvidenceLocator()

    def recommend(self, requirement: str, candidates: tuple[RetrievalResult, ...] | list[RetrievalResult]) -> IntelligenceRun:
        """Produce ranked standard recommendations while preserving all diagnostic candidates."""
        metrics = PipelineMetrics()
        source = self._promote_explicit_matches(
            requirement,
            tuple(candidates),
            indexed_standards=self.knowledge_base.indexed_standards(),
        )
        started = perf_counter()
        features = self.interpreter.interpret(requirement)
        metrics.record("requirement_interpretation", started, 1)

        started = perf_counter()
        constrained = tuple(
            ScoredCandidate(
                candidate=candidate,
                constraint_result=self.constraint_evaluator.evaluate(
                    requirement, candidate, self.knowledge_base.context_for(candidate.chunk.standard_number)
                ),
            )
            for candidate in source
        )
        metrics.record("constraints", started, len(constrained))
        admissible = tuple(item.candidate for item in constrained if item.constraint_result.admissible)

        started = perf_counter()
        rerank_warnings: tuple[str, ...] = ()
        try:
            reranked = self.reranker.rerank(requirement, admissible)
        except Exception as exc:
            logger.exception("Reranking failed; using deterministic token-overlap fallback")
            reranked = TokenOverlapReranker().rerank(requirement, admissible)
            rerank_warnings = (f"Configured reranker failed; deterministic fallback used: {exc}",)
            metrics.record("reranking", started, len(reranked), exc)
        else:
            metrics.record("reranking", started, len(reranked))

        started = perf_counter()
        rerank_by_id = {item.candidate.chunk.chunk_id: item.result for item in reranked}
        scored: list[ScoredCandidate] = []
        for item in constrained:
            if not item.constraint_result.admissible:
                scored.append(item)
                continue
            context = self.knowledge_base.context_for(item.candidate.chunk.standard_number)
            try:
                applicability = self.assessor.assess(features, item.candidate, context)
            except Exception as exc:
                logger.exception("Applicability scoring failed for chunk %s", item.candidate.chunk.chunk_id)
                scored.append(ScoredCandidate(
                    candidate=item.candidate,
                    constraint_result=item.constraint_result,
                    rerank_result=rerank_by_id[item.candidate.chunk.chunk_id],
                    warnings=(f"Applicability scoring failed: {exc}",),
                ))
                continue
            rerank = rerank_by_id[item.candidate.chunk.chunk_id]
            final_score = (
                self.config.final_rerank_weight * rerank.normalized_score
                + (1 - self.config.final_rerank_weight) * applicability.overall
            )
            scored.append(ScoredCandidate(
                candidate=item.candidate,
                constraint_result=item.constraint_result,
                rerank_result=rerank,
                applicability=applicability,
                final_score=final_score,
                warnings=rerank_warnings,
            ))
        metrics.record("applicability", started, len(scored))

        recommendations = self._recommendations(tuple(scored), metrics, requirement)
        logger.info(
            "Intelligence pipeline completed: candidates=%d admissible=%d recommendations=%d",
            len(source), len(admissible), len(recommendations),
        )
        return IntelligenceRun(recommendations, tuple(scored), metrics)

    def _recommendations(
        self,
        candidates: tuple[ScoredCandidate, ...],
        metrics: PipelineMetrics,
        requirement: str,
    ) -> tuple[Recommendation, ...]:
        eligible = [
            item for item in candidates
            if canonical_recommendation_id(item.candidate.chunk.standard_number)
            and self.knowledge_base.has_independent_standard(
                canonical_recommendation_id(item.candidate.chunk.standard_number)
            )
            and item.applicability
            and item.applicability.product_match > 0
            and item.applicability.overall >= self.config.minimum_applicability
            and item.rerank_result
            and item.final_score is not None
            and item.final_score >= self.config.minimum_final_score
        ]
        groups: OrderedDict[str, list[ScoredCandidate]] = OrderedDict()
        for item in eligible:
            standard = canonical_recommendation_id(item.candidate.chunk.standard_number)
            if standard:
                groups.setdefault(standard, []).append(item)
        ranked_groups = sorted(
            groups.items(),
            key=lambda entry: (-max(item.final_score or 0.0 for item in entry[1]), entry[0]),
        )[: self.config.final_top_k]
        relationship_started = perf_counter()
        related_by_standard: dict[str, tuple] = {}
        relationship_warnings: dict[str, tuple[str, ...]] = {}
        for standard, _ in ranked_groups:
            try:
                related_by_standard[standard] = self.knowledge_base.related(standard, self.config.relationship_depth)
                relationship_warnings[standard] = ()
            except Exception as exc:
                logger.exception("Relationship traversal failed for %s", standard)
                related_by_standard[standard] = ()
                relationship_warnings[standard] = (f"Relationship traversal failed: {exc}",)
        metrics.record("relationships", relationship_started, sum(len(items) for items in related_by_standard.values()))

        evidence_started = perf_counter()
        output: list[Recommendation] = []
        for rank, (standard, group) in enumerate(ranked_groups, start=1):
            primary = sorted(
                group,
                key=lambda item: (-(item.final_score or 0.0), item.candidate.rank, item.candidate.chunk.chunk_id),
            )[0]
            context = self.knowledge_base.context_for(standard)
            evidence = self.evidence_locator.select(tuple(group), self.config.evidence_limit)
            title = context.metadata.title if context.metadata else self._explicit_title(standard, primary.candidate.chunk.text, requirement)
            output.append(Recommendation(
                standard_number=standard,
                title=title,
                final_score=primary.final_score or 0.0,
                rank=rank,
                constraint_result=primary.constraint_result,
                rerank_result=primary.rerank_result,
                applicability=primary.applicability,
                evidence=evidence,
                related_standards=related_by_standard[standard],
                status=context.metadata.status if context.metadata else None,
                version=(context.metadata.revision or context.metadata.edition) if context.metadata else None,
                warnings=(*primary.warnings, *relationship_warnings[standard]),
            ))
        metrics.record("evidence", evidence_started, sum(len(item.evidence) for item in output))
        return tuple(output)

    @staticmethod
    def _promote_explicit_matches(
        requirement: str,
        candidates: tuple[RetrievalResult, ...],
        *,
        indexed_standards: frozenset[str] = frozenset(),
    ) -> tuple[RetrievalResult, ...]:
        """Promote only when the cited code is independently indexed in the corpus."""
        if not indexed_standards:
            return candidates
        phrase = re.escape(" ".join(requirement.split()))
        pattern = re.compile(
            rf"\b((?:IS|SP)\s*\d{{1,5}}\s*:\s*\d{{4}})\s+{phrase}(?=\s|$)",
            re.IGNORECASE,
        )
        promoted: list[RetrievalResult] = list(candidates)
        seen = {item.chunk.chunk_id for item in candidates}
        for candidate in candidates:
            match = pattern.search(" ".join(candidate.chunk.text.split()))
            if not match:
                continue
            standard = canonical_is_code(match.group(1))
            if not standard or standard not in indexed_standards:
                continue
            promoted_chunk = replace(candidate.chunk, standard_number=standard)
            promoted_id = f"{candidate.chunk.chunk_id}:{standard}"
            promoted_chunk = replace(promoted_chunk, chunk_id=promoted_id)
            if promoted_id not in seen:
                promoted.append(replace(candidate, chunk=promoted_chunk))
                seen.add(promoted_id)
        return tuple(promoted)

    @staticmethod
    def _explicit_title(standard: str, text: str, requirement: str) -> str | None:
        phrase = re.escape(" ".join(requirement.split()))
        standard_pattern = (
            re.escape(standard)
            .replace("\\ ", r"\s*")
            .replace(":", r"\s*:\s*")
        )
        if re.search(rf"\b{standard_pattern}\s+{phrase}(?=\s|$)", " ".join(text.split()), re.IGNORECASE):
            return requirement
        return None

    def _configured_reranker(self) -> Reranker:
        if self.config.reranker == "cross_encoder":
            return CrossEncoderReranker(self.config.cross_encoder_model or "", self.config.cross_encoder_batch_size)
        return TokenOverlapReranker()
