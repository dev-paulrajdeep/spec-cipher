"""Extract explicit IS references and stated version relationships only."""

from __future__ import annotations

import re

from corpus.models import CorpusPage, Section, StandardRelationship
from corpus.processing.normalizer import find_standard_references


_RELATION_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("superseded_by", re.compile(r"\bsuperseded\s+by\b", re.I)),
    ("supersedes", re.compile(r"\bsupersedes\b", re.I)),
    ("amendment_of", re.compile(r"\bamend(?:ment|ed)\b", re.I)),
    ("revision_of", re.compile(r"\brevis(?:ion|ed)\b", re.I)),
    ("referenced_by", re.compile(r"\breferenced\s+by\b", re.I)),
)


class RelationshipExtractor:
    """Create relationship facts, never semantic guesses or a graph data store."""

    def extract(
        self,
        pages: tuple[CorpusPage, ...],
        source_standard_number: str | None,
        sections: tuple[Section, ...] = (),
    ) -> tuple[StandardRelationship, ...]:
        relationships: list[StandardRelationship] = []
        seen: set[tuple[str | None, str, str, int]] = set()
        reference_pages = {
            page_number
            for section in sections
            if section.title and re.search(r"\breferences?\b", section.title, re.I)
            for page_number in range(section.page_start, section.page_end + 1)
        }
        for page in pages:
            for line in page.cleaned_text.splitlines():
                for target in find_standard_references(line):
                    if target == source_standard_number and not self._explicit_relation(line):
                        continue
                    relationship_type = self._relationship_type(line, page.page_number in reference_pages)
                    key = (source_standard_number, target, relationship_type, page.page_number)
                    if key in seen:
                        continue
                    seen.add(key)
                    relationships.append(
                        StandardRelationship(
                            source_standard_number=source_standard_number,
                            target_standard_number=target,
                            relationship_type=relationship_type,
                            page_number=page.page_number,
                            evidence_text=line.strip(),
                        )
                    )
        return tuple(relationships)

    @staticmethod
    def _explicit_relation(line: str) -> bool:
        return any(pattern.search(line) for _, pattern in _RELATION_RULES)

    @staticmethod
    def _relationship_type(line: str, is_reference_section: bool = False) -> str:
        for relationship_type, pattern in _RELATION_RULES:
            if pattern.search(line):
                return relationship_type
        if is_reference_section or re.search(r"\b(?:reference|references|normative reference)\b", line, re.I):
            return "references"
        return "related_to"
