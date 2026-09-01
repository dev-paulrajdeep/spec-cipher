"""Read-only views over corpus relationship facts and metadata; no graph database."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import json
from pathlib import Path

from corpus.models import StandardDocument, StandardMetadata, StandardRelationship
from intelligence.models import RelatedStandard


_INVERSE_TYPES = {
    "references": "referenced_by",
    "referenced_by": "references",
    "amendment_of": "amendment_of",
    "revision_of": "revision_of",
    "supersedes": "superseded_by",
    "superseded_by": "supersedes",
    "related_to": "related_to",
}
_ALLOWED_TYPES = frozenset(_INVERSE_TYPES)


@dataclass(frozen=True)
class StandardContext:
    metadata: StandardMetadata | None
    scope_text: str | None = None


class RelationshipGraph:
    """Query-only adjacency view derived from explicit corpus relationship facts."""

    def __init__(self, relationships: tuple[StandardRelationship, ...] = ()) -> None:
        self._adjacency: dict[str, list[tuple[str, str, StandardRelationship]]] = defaultdict(list)
        for relationship in relationships:
            if relationship.relationship_type not in _ALLOWED_TYPES or not relationship.source_standard_number:
                continue
            self._adjacency[relationship.source_standard_number].append(
                (relationship.target_standard_number, relationship.relationship_type, relationship)
            )
            inverse_type = _INVERSE_TYPES[relationship.relationship_type]
            self._adjacency[relationship.target_standard_number].append(
                (relationship.source_standard_number, inverse_type, relationship)
            )

    def traverse(self, standard_number: str, max_depth: int = 1) -> tuple[RelatedStandard, ...]:
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        if not standard_number or max_depth == 0:
            return ()
        output: list[RelatedStandard] = []
        visited: set[tuple[str, str, str]] = set()
        queue: deque[tuple[str, int]] = deque([(standard_number, 0)])
        visited_nodes = {standard_number}
        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for target, relation_type, fact in self._adjacency.get(current, []):
                edge_key = (current, target, relation_type)
                if edge_key not in visited:
                    visited.add(edge_key)
                    output.append(RelatedStandard(
                        standard_number=target,
                        relationship_type=relation_type,
                        source_standard_number=current,
                        page_number=fact.page_number,
                        evidence_text=fact.evidence_text,
                        depth=depth + 1,
                    ))
                if target not in visited_nodes:
                    visited_nodes.add(target)
                    queue.append((target, depth + 1))
        return tuple(output)


class CorpusKnowledgeBase:
    """Read-only corpus metadata/scope/relationship lookup used by Intelligence."""

    def __init__(self, contexts: dict[str, StandardContext] | None = None, relationships: tuple[StandardRelationship, ...] = ()) -> None:
        self._contexts = dict(contexts or {})
        self._graph = RelationshipGraph(relationships)

    @classmethod
    def from_documents(cls, documents: tuple[StandardDocument, ...]) -> "CorpusKnowledgeBase":
        contexts: dict[str, StandardContext] = {}
        relationships: list[StandardRelationship] = []
        for document in documents:
            metadata = document.metadata
            if metadata.standard_number:
                scope_text = "\n".join(
                    section.text for section in document.sections
                    if section.title and section.title.strip().lower() == "scope"
                ) or None
                contexts[metadata.standard_number] = StandardContext(metadata, scope_text)
            relationships.extend(document.relationships)
        return cls(contexts, tuple(relationships))

    @classmethod
    def from_canonical_json(cls, paths: tuple[str | Path, ...]) -> "CorpusKnowledgeBase":
        """Load facts from canonical corpus JSON artifacts, not source PDFs."""
        contexts: dict[str, StandardContext] = {}
        relationships: list[StandardRelationship] = []
        for raw_path in paths:
            payload = json.loads(Path(raw_path).read_text(encoding="utf-8"))
            metadata_data = payload.get("metadata")
            if not isinstance(metadata_data, dict):
                raise ValueError("Canonical corpus JSON is missing metadata")
            metadata = StandardMetadata(**metadata_data)
            scope_text = "\n".join(
                section.get("text", "") for section in payload.get("sections", [])
                if isinstance(section, dict) and str(section.get("title", "")).strip().lower() == "scope"
            ) or None
            if metadata.standard_number:
                contexts[metadata.standard_number] = StandardContext(metadata, scope_text)
            for row in payload.get("relationships", []):
                if not isinstance(row, dict):
                    raise ValueError("Canonical corpus relationship records must be objects")
                relationships.append(StandardRelationship(**row))
        return cls(contexts, tuple(relationships))

    def context_for(self, standard_number: str | None) -> StandardContext:
        return self._contexts.get(standard_number or "", StandardContext(None, None))

    def related(self, standard_number: str, max_depth: int = 1) -> tuple[RelatedStandard, ...]:
        return self._graph.traverse(standard_number, max_depth)
