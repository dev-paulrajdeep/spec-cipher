"""Project extracted canonical documents into immutable-release catalog records."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Any

from speccipher.domain.identity import StandardId
from speccipher.ingestion.assembly.multipart import assemble_multipart_publication
from speccipher.ingestion.overrides.loader import validate_operator_override


class CanonicalRecordError(ValueError):
    """Raised when extracted documents lack enough evidence for release projection."""


def _text(value: Any, *, label: str, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip():
        raise CanonicalRecordError(f"{label} must be a non-empty string")
    return value.strip()


def _digest(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None
    ):
        raise CanonicalRecordError(f"{label} must be a SHA-256 hex digest")
    return value.lower()


def _parse_identity(value: Any, *, label: str) -> StandardId | None:
    if value is None:
        return None
    try:
        return StandardId.parse(value)
    except (TypeError, ValueError):
        return None


def _page_count(metadata: Mapping[str, Any], label: str) -> int:
    value = metadata.get("page_count")
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise CanonicalRecordError(f"{label} page_count must be a positive integer")
    return value


def _source_record(
    document: Mapping[str, Any],
    *,
    volume_layouts: Mapping[str, Mapping[str, Any]],
    operator_overrides: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    metadata = document.get("metadata")
    if not isinstance(metadata, Mapping):
        raise CanonicalRecordError("Canonical document metadata must be an object")
    source_path = _text(
        metadata.get("source_path") or metadata.get("source_file"),
        label="metadata.source_path",
    )
    filename = _text(metadata.get("source_file"), label="metadata.source_file")
    fingerprint = _digest(metadata.get("document_fingerprint"), label="document_fingerprint")
    _text(metadata.get("document_id"), label="document_id")
    page_count = _page_count(metadata, source_path)
    identity_status = metadata.get("identity_status")
    identity_source = metadata.get("identity_source")
    raw_code = metadata.get("standard_number")
    identity = _parse_identity(raw_code, label="standard_number")
    resolved = identity_status == "resolved" and identity is not None

    raw_override = operator_overrides.get(source_path)
    override = None
    if raw_override is not None:
        try:
            override = validate_operator_override(dict(raw_override))
        except (TypeError, ValueError) as exc:
            raise CanonicalRecordError(f"Invalid operator override for {source_path!r}: {exc}") from exc
        if not override.evidence_sources:
            raise CanonicalRecordError(
                f"Operator override for {source_path!r} requires explicit evidence_sources"
            )
        if resolved and override.asserted_identity != identity:
            raise CanonicalRecordError(
                f"Operator override identity conflicts with extracted identity for {source_path!r}"
            )
        identity = override.asserted_identity

    layout = volume_layouts.get(source_path)
    if layout is not None and not isinstance(layout, Mapping):
        raise CanonicalRecordError(f"Volume layout for {source_path!r} must be an object")
    page_start = 1 if layout is None else layout.get("page_start")
    page_end = page_start + page_count - 1 if layout is None else layout.get("page_end")
    part_index = None if layout is None else layout.get("part_index")
    publication_id = None if layout is None else layout.get("logical_document_id")
    if not isinstance(page_start, int) or isinstance(page_start, bool) or page_start < 1:
        raise CanonicalRecordError(f"Declared page_start for {source_path!r} must be positive")
    if not isinstance(page_end, int) or isinstance(page_end, bool) or page_end < page_start:
        raise CanonicalRecordError(f"Declared page_end for {source_path!r} is invalid")
    if page_end - page_start + 1 != page_count:
        raise CanonicalRecordError(
            f"Declared page range for {source_path!r} does not match extracted page_count"
        )
    if part_index is not None and (
        not isinstance(part_index, int) or isinstance(part_index, bool) or part_index < 1
    ):
        raise CanonicalRecordError(f"Declared part_index for {source_path!r} must be positive")
    if publication_id is not None and (
        not isinstance(publication_id, str) or not publication_id.strip()
    ):
        raise CanonicalRecordError(
            f"Declared logical_document_id for {source_path!r} must be a non-empty string"
        )

    source_file_id = "src_" + hashlib.sha256(
        f"{source_path}\0{fingerprint}".encode()
    ).hexdigest()[:20]
    code = identity.code if identity is not None else None
    document_type = metadata.get("document_type")
    validation = document.get("validation", [])
    has_validation_errors = isinstance(validation, list) and any(
        isinstance(issue, Mapping) and issue.get("severity") == "error" for issue in validation
    )
    source_recommendation_eligible = (
        (resolved and metadata.get("recommendation_eligible") is True)
        or (override is not None and document_type not in {"catalog", "list", "index"})
    )
    return {
        "document": document,
        "metadata": metadata,
        "source_path": source_path,
        "filename": filename,
        "fingerprint": fingerprint,
        "source_file_id": source_file_id,
        "physical_page_count": page_count,
        "page_start": page_start,
        "page_end": page_end,
        "part_index": part_index,
        "logical_document_id": publication_id.strip() if publication_id is not None else None,
        "identity": identity,
        "identity_code": code,
        "identity_status": "resolved" if resolved or override is not None else "unresolved",
        "identity_source": "operator_override" if override is not None else identity_source,
        "override": override,
        "eligible": source_recommendation_eligible and not has_validation_errors,
        "document_type": document_type,
    }


def records_from_canonical_documents(
    documents: Sequence[Mapping[str, Any]],
    *,
    volume_layouts: Mapping[str, Mapping[str, Any]] | None = None,
    operator_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    missing_volumes: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Project canonical extraction records without inventing identities or pages.

    `volume_layouts` and `operator_overrides` are keyed by canonical
    `metadata.source_path`. Multipart groups require an explicit page range for
    every physical source. Missing spans must be declared in `missing_volumes`.
    """
    layouts = volume_layouts or {}
    overrides = operator_overrides or {}
    missing_by_identity = missing_volumes or {}
    sources = [
        _source_record(document, volume_layouts=layouts, operator_overrides=overrides)
        for document in documents
    ]
    if not sources:
        raise CanonicalRecordError("At least one canonical document is required")
    if len({source["source_path"] for source in sources}) != len(sources):
        raise CanonicalRecordError("Canonical source_path values must be unique")

    groups: dict[str, list[dict[str, Any]]] = {}
    for source in sources:
        identity_code = source["identity_code"]
        group_key = (
            f"standard:{identity_code}"
            if identity_code is not None
            else f"document:{source['metadata']['document_id']}"
        )
        groups.setdefault(group_key, []).append(source)

    document_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []
    relationship_rows: list[dict[str, Any]] = []
    scope_rows: list[dict[str, Any]] = []
    override_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    indexed_codes: set[str] = set()

    for group_key in sorted(groups):
        group = groups[group_key]
        if len(group) > 1 and any(source["source_path"] not in layouts for source in group):
            raise CanonicalRecordError(
                f"Every source in multipart group {group_key!r} needs a declared page range"
            )
        declared_publication_ids = {
            source["logical_document_id"] for source in group
            if source["logical_document_id"] is not None
        }
        if len(group) > 1 and (
            len(declared_publication_ids) != 1
            or any(source["logical_document_id"] is None for source in group)
        ):
            raise CanonicalRecordError(
                f"Multipart group {group_key!r} requires one explicit logical_document_id "
                "shared by every source; matching standard numbers alone do not prove identity"
            )
        code_values = {source["identity_code"] for source in group if source["identity_code"]}
        if len(code_values) > 1:
            raise CanonicalRecordError(f"Conflicting identities in publication group {group_key!r}")
        identity_code = next(iter(code_values), None)
        standard_id = _parse_identity(identity_code, label="asserted identity")
        if identity_code is not None and standard_id is None:
            raise CanonicalRecordError(f"Unsupported standard identity in group {group_key!r}")
        logical_doc_id = next(iter(declared_publication_ids), None) or (
            "pub_" + re.sub(r"[^a-z0-9]+", "_", standard_id.code.lower()).strip("_")
            if standard_id is not None
            else group[0]["metadata"]["document_id"]
        )

        source_overrides = [source["override"] for source in group if source["override"]]
        if source_overrides and any(
            override.asserted_identity.code != identity_code for override in source_overrides
        ):
            raise CanonicalRecordError(f"Conflicting operator assertions in group {group_key!r}")
        declared_missing = missing_by_identity.get(
            identity_code or logical_doc_id,
            missing_by_identity.get(logical_doc_id, ()),
        )
        volume_inputs = [
            {
                "source_path": source["source_path"],
                "page_start": source["page_start"],
                "page_end": source["page_end"],
                "page_count": source["physical_page_count"],
                "asserted_identity": source["identity_code"],
            }
            for source in group
        ]
        try:
            assembly = assemble_multipart_publication(
                volume_inputs,
                missing_volumes=declared_missing,
            )
        except ValueError as exc:
            raise CanonicalRecordError(f"Invalid page assembly for {group_key!r}: {exc}") from exc
        assembled_by_path = {volume["source_path"]: volume for volume in assembly["volumes"]}
        ordered_sources = [
            next(source for source in group if source["source_path"] == volume["source_path"])
            for volume in assembly["volumes"]
        ]
        for source, volume in zip(ordered_sources, assembly["volumes"], strict=True):
            declared_part_index = source["part_index"]
            if declared_part_index is not None and declared_part_index != volume["volume_order"]:
                raise CanonicalRecordError(
                    f"Declared part_index for {source['source_path']!r} does not match its "
                    "page-range order"
                )

        statuses = {
            source["metadata"].get("status")
            for source in ordered_sources
            if source["metadata"].get("status") is not None
        }
        if len(statuses) > 1:
            raise CanonicalRecordError(f"Conflicting publication statuses in {group_key!r}")
        editions = {
            source["metadata"].get("edition")
            for source in ordered_sources
            if source["metadata"].get("edition") is not None
        }
        if len(editions) > 1:
            raise CanonicalRecordError(f"Conflicting publication editions in {group_key!r}")

        eligible = bool(standard_id) and all(source["eligible"] for source in ordered_sources)
        scope_parts: list[str] = []
        exclusion_parts: list[str] = []
        chunk_ordinal = 0
        title = next(
            (source["metadata"].get("title") for source in ordered_sources if source["metadata"].get("title")),
            None,
        )
        document_rows.append(
            {
                "id": logical_doc_id,
                "family": standard_id.family if standard_id else None,
                "number": standard_id.number if standard_id else None,
                "part": standard_id.part if standard_id else None,
                "year": standard_id.year if standard_id else None,
                "code": identity_code,
                "title": title,
                "status": next(iter(statuses), None),
                "edition": next(iter(editions), None),
                "identity_source": (
                    "operator_override"
                    if source_overrides
                    else ordered_sources[0]["identity_source"]
                ),
                "identity_status": "resolved" if standard_id else "unresolved",
                "is_eligible": int(eligible),
                "is_stub": 0,
                "page_count": assembly["total_pages"],
            }
        )
        if identity_code:
            indexed_codes.add(identity_code)

        for source in ordered_sources:
            volume = assembled_by_path[source["source_path"]]
            source_rows.append(
                {
                    "id": source["source_file_id"],
                    "logical_doc_id": logical_doc_id,
                    "file_name": source["filename"],
                    "relative_path": source["source_path"],
                    "sha256": source["fingerprint"],
                    "page_count": source["physical_page_count"],
                    "part_index": source["part_index"] or volume["volume_order"],
                    "page_offset": volume["page_offset"],
                    "extraction_status": _extraction_status(source["document"]),
                }
            )
            raw_chunks = source["document"].get("chunks", [])
            if not isinstance(raw_chunks, list):
                raise CanonicalRecordError(f"Chunks for {source['source_path']!r} must be an array")
            normalized_chunks = sorted(
                raw_chunks,
                key=lambda chunk: (
                    chunk.get("page_start", 0),
                    chunk.get("page_end", 0),
                    chunk.get("section_number") or "",
                    chunk.get("chunk_id") or "",
                ),
            )
            for chunk in normalized_chunks:
                if not isinstance(chunk, Mapping):
                    raise CanonicalRecordError("Every canonical chunk must be an object")
                physical_start = chunk.get("page_start")
                physical_end = chunk.get("page_end")
                if (
                    not isinstance(physical_start, int)
                    or isinstance(physical_start, bool)
                    or not isinstance(physical_end, int)
                    or isinstance(physical_end, bool)
                    or physical_start < 1
                    or physical_end < physical_start
                    or physical_end > source["physical_page_count"]
                ):
                    raise CanonicalRecordError(
                        f"Invalid physical page range in chunk from {source['source_path']!r}"
                    )
                chunk_ordinal += 1
                # Each publication gets a globally ordered chunk sequence even if
                # legacy per-volume chunkers restarted their local sequence.
                chunk_rows.append(
                    {
                        "chunk_id": f"{logical_doc_id}_c{chunk_ordinal:05d}",
                        "source_chunk_id": chunk.get("chunk_id"),
                        "logical_doc_id": logical_doc_id,
                        "standard_code": identity_code,
                        "source_file_id": source["source_file_id"],
                        "page_start": physical_start,
                        "page_end": physical_end,
                        "absolute_page_start": volume["page_offset"] + physical_start,
                        "absolute_page_end": volume["page_offset"] + physical_end,
                        "section_number": chunk.get("section_number"),
                        "section_title": chunk.get("section_title"),
                        "text": chunk.get("text"),
                        "is_eligible": int(
                            eligible
                            and (
                                source["override"] is not None
                                or chunk.get("recommendation_eligible") is True
                            )
                        ),
                    }
                )

            for relation in source["document"].get("relationships", []):
                if not isinstance(relation, Mapping):
                    raise CanonicalRecordError("Every canonical relationship must be an object")
                target = _text(relation.get("target_standard_number"), label="relationship target")
                page_number = relation.get("page_number")
                if (
                    not isinstance(page_number, int)
                    or isinstance(page_number, bool)
                    or not 1 <= page_number <= source["physical_page_count"]
                ):
                    raise CanonicalRecordError("Relationship page_number is outside its source file")
                source_code = identity_code
                if source_code is not None:
                    relationship_rows.append(
                        {
                            "source_code": source_code,
                            "target_code": target,
                            "relationship_type": relation.get("relationship_type"),
                            "page_number": volume["page_offset"] + page_number,
                            "evidence_text": relation.get("evidence_text") or "",
                        }
                    )
            if source["override"] is not None:
                override = source["override"]
                override_rows.append(
                    {
                        "override_id": "override_" + source["source_file_id"],
                        "logical_doc_id": logical_doc_id,
                        "source_file_id": source["source_file_id"],
                        "asserted_identity": override.asserted_identity.code,
                        "operator_id": override.operator_id,
                        "timestamp": override.timestamp,
                        "reason": override.reason,
                        "evidence_sources": list(override.evidence_sources),
                    }
                )

            sections = source["document"].get("sections", [])
            if not isinstance(sections, list):
                raise CanonicalRecordError("Canonical sections must be an array")
            for section in sections:
                if not isinstance(section, Mapping):
                    continue
                section_title = str(section.get("title") or "").strip().lower()
                section_text = section.get("text")
                if not isinstance(section_text, str) or not section_text.strip():
                    continue
                if "scope" in section_title:
                    scope_parts.append(section_text.strip())
                if "exclusion" in section_title:
                    exclusion_parts.append(section_text.strip())

        if identity_code and (scope_parts or exclusion_parts):
            scope_rows.append(
                {
                    "standard_code": identity_code,
                    "scope": "\n\n".join(scope_parts) if scope_parts else None,
                    "exclusions": "\n\n".join(exclusion_parts) if exclusion_parts else None,
                }
            )

        for missing in assembly["missing_volumes"]:
            missing_rows.append(
                {
                    "missing_volume_id": f"{logical_doc_id}_missing_{missing['volume_order']}",
                    "logical_doc_id": logical_doc_id,
                    "volume_id": missing["volume_id"],
                    "absolute_page_start": missing["page_start"],
                    "absolute_page_end": missing["page_end"],
                    "reason": missing.get("reason"),
                }
            )

    stub_rows = _make_stubs(relationship_rows, indexed_codes)
    document_rows.extend(stub_rows)
    document_ids = [row["id"] for row in document_rows]
    if len(set(document_ids)) != len(document_ids):
        raise CanonicalRecordError("Projected logical document IDs are not unique")
    return {
        "documents": document_rows,
        "source_files": source_rows,
        "chunks": chunk_rows,
        "relationships": _unique_rows(relationship_rows),
        "scope_texts": scope_rows,
        "operator_overrides": override_rows,
        "missing_volumes": missing_rows,
    }


def _extraction_status(document: Mapping[str, Any]) -> str:
    pages = document.get("pages", [])
    statuses = {
        page.get("extraction_status")
        for page in pages
        if isinstance(page, Mapping) and isinstance(page.get("extraction_status"), str)
    } if isinstance(pages, list) else set()
    if "ocr_required" in statuses:
        return "ocr_required"
    if "ocr_applied" in statuses:
        return "ocr_applied"
    return "processed"


def _unique_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    unique: list[dict[str, Any]] = []
    for row in rows:
        key = tuple(row.get(field) for field in sorted(row))
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def _make_stubs(
    relationships: list[dict[str, Any]], indexed_codes: set[str]
) -> list[dict[str, Any]]:
    targets: dict[str, StandardId] = {}
    for relationship in relationships:
        target_code = relationship["target_code"]
        try:
            target_id = StandardId.parse(target_code)
        except ValueError:
            continue
        if target_id.code not in indexed_codes:
            targets[target_id.code] = target_id
    return [
        {
            "id": "stub_" + hashlib.sha256(code.encode("utf-8")).hexdigest()[:16],
            "family": identity.family,
            "number": identity.number,
            "part": identity.part,
            "year": identity.year,
            "code": code,
            "title": None,
            "status": None,
            "edition": None,
            "identity_source": "explicit_reference",
            "identity_status": "unverified",
            "is_eligible": 0,
            "is_stub": 1,
            "page_count": 0,
        }
        for code, identity in sorted(targets.items())
    ]
