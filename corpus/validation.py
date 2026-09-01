"""Validation of canonical corpus output before it reaches retrieval."""

from __future__ import annotations

from corpus.models import StandardDocument, ValidationIssue


_RELATIONSHIP_TYPES = {
    "references",
    "referenced_by",
    "amendment_of",
    "revision_of",
    "supersedes",
    "superseded_by",
    "related_to",
}


class CorpusValidator:
    """Validate recoverable issues without silently dropping imperfect source data."""

    def validate(self, document: StandardDocument) -> tuple[ValidationIssue, ...]:
        issues: list[ValidationIssue] = []
        metadata = document.metadata
        if not metadata.source_file.strip():
            issues.append(ValidationIssue("error", "missing_source_file", "Metadata source_file is required."))
        if metadata.page_count != len(document.pages):
            issues.append(ValidationIssue("error", "page_count_mismatch", "Metadata page_count does not match pages."))
        if not metadata.standard_number:
            issues.append(ValidationIssue("warning", "missing_standard_id", "No standard identifier was extracted."))
        if not document.pages:
            issues.append(ValidationIssue("error", "empty_document", "Document contains no pages."))
        for page in document.pages:
            if not page.raw_text.strip() and not page.cleaned_text.strip():
                issues.append(ValidationIssue("warning", "empty_page", "Source page contains no extracted text.", page.page_number))
        identifiers: set[str] = set()
        for chunk in document.chunks:
            if chunk.chunk_id in identifiers:
                issues.append(ValidationIssue("error", "duplicate_chunk_id", f"Duplicate chunk id: {chunk.chunk_id}"))
            identifiers.add(chunk.chunk_id)
            if not chunk.text.strip():
                issues.append(ValidationIssue("error", "empty_chunk", f"Chunk {chunk.chunk_id} is empty."))
            if not chunk.source_file or not chunk.standard_number:
                issues.append(ValidationIssue("error", "chunk_missing_provenance", f"Chunk {chunk.chunk_id} lacks source or standard provenance."))
            if chunk.page_start < 1 or chunk.page_end < chunk.page_start or chunk.page_end > metadata.page_count:
                issues.append(ValidationIssue("error", "invalid_chunk_page_range", f"Chunk {chunk.chunk_id} has invalid page range."))
        for relation in document.relationships:
            if relation.relationship_type not in _RELATIONSHIP_TYPES or not relation.target_standard_number:
                issues.append(ValidationIssue("error", "malformed_relationship", "Relationship has an invalid type or target."))
            if relation.page_number < 1 or relation.page_number > metadata.page_count:
                issues.append(ValidationIssue("error", "invalid_relationship_page", "Relationship evidence page is invalid."))
            if not relation.evidence_text.strip():
                issues.append(ValidationIssue("error", "missing_relationship_evidence", "Relationship lacks source evidence."))
        return tuple(issues)
