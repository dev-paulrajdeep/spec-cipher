"""Evidence selection from scored retrieval candidates with provenance intact."""

from __future__ import annotations

from intelligence.models import EvidenceItem, ScoredCandidate


class EvidenceLocator:
    """Choose already-retrieved chunks only; never invent or reparse evidence."""

    def select(self, candidates: tuple[ScoredCandidate, ...], limit: int) -> tuple[EvidenceItem, ...]:
        if limit < 1:
            raise ValueError("Evidence limit must be positive")
        ordered = sorted(
            candidates,
            key=lambda item: (-(item.final_score or 0.0), item.candidate.rank, item.candidate.chunk.chunk_id),
        )
        evidence: list[EvidenceItem] = []
        seen: set[str] = set()
        for item in ordered:
            chunk = item.candidate.chunk
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            evidence.append(EvidenceItem(
                standard_number=chunk.standard_number or "<unknown standard>",
                chunk_id=chunk.chunk_id,
                source_file=chunk.source_file,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section_number=chunk.section_number,
                section_title=chunk.section_title,
                text=chunk.text,
                relevance_score=item.final_score or 0.0,
            ))
            if len(evidence) == limit:
                break
        return tuple(evidence)
