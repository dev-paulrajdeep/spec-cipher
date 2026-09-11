"""Conservative document-level identity resolution for arbitrary technical PDFs.

Mentions of other document numbers inside a PDF are not treated as independently
indexed documents. Identity must come from a clear filename, PDF metadata title,
or a single document-level header signal.

Catalog/list/index documents may be retained as reference-only records. Synthetic
filenames such as ``temp_123.pdf`` never become ``standard_number`` values.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal

from corpus.processing.normalizer import (
    find_document_numbers,
    normalize_document_number,
    normalize_text,
)


DocumentType = Literal[
    "standard",
    "catalog",
    "list",
    "index",
    "manual",
    "technical_document",
    "unknown",
]

_CATALOG_HINT_RE = re.compile(
    r"\b(list of|annexure|products notified|compulsory\s+marking|simplified procedure|"
    r"is number\s+title|grant of licence|product\s+category|schedule of|index of standards|"
    r"product list|catalogue|catalog)\b",
    re.IGNORECASE,
)
_LIST_HINT_RE = re.compile(r"\b(list of|products notified|product list|schedule of)\b", re.I)
_INDEX_HINT_RE = re.compile(r"\b(index of standards|contents index|subject index)\b", re.I)
_MANUAL_HINT_RE = re.compile(
    r"\b(manual|handbook|guide|installation instructions|user guide|operating instructions)\b",
    re.I,
)
_FAMILY_PREFIX_RE = re.compile(r"^(?:ISO|IEC|IS|SP|EN|BS|ASTM)\b", re.I)
# Distinct base identifiers beyond this count without a filename identity are
# treated as catalog/list documents rather than a single publication.
_MULTI_DOCUMENT_THRESHOLD = 3
_AMBIGUOUS_FILENAME_STEMS = frozenset(
    {
        "temp",
        "tmp",
        "document",
        "download",
        "file",
        "untitled",
        "scan",
        "image",
        "export",
        "copy",
        "new",
        "pdf",
        "doc",
        "docs",
        "attachment",
        "unnamed",
    }
)
_AMBIGUOUS_FILENAME_RE = re.compile(
    r"^(?:temp|tmp|document|download|file|untitled|scan|image|export|copy|new|doc|pdf)"
    r"(?:[_\-\s]?\d+)?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DocumentIdentity:
    """Document-level identity outcome for one source PDF."""

    document_id: str | None
    title: str | None
    document_number: str | None
    confidence: float
    source: str
    status: Literal["resolved", "unresolved"]
    reason: str
    document_type: DocumentType = "unknown"
    recommendation_eligible: bool = False
    candidates: tuple[str, ...] = ()
    ingestible: bool = False

    @property
    def resolved(self) -> bool:
        return self.status == "resolved"

    @property
    def standard_number(self) -> str | None:
        """Compatibility alias used by existing corpus models."""
        return self.document_number


# Backward-compatible name used by existing imports/tests.
IdentityResolution = DocumentIdentity


def _base_document_number(value: str) -> str:
    """Strip trailing year so dated and undated forms collapse to one base id."""
    return re.sub(r"[:/]\d{2,4}$", "", value).strip()


def _is_ambiguous_filename(filename: str) -> bool:
    stem = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    stem = re.sub(r"\.[^.]+$", "", stem).strip()
    if not stem:
        return True
    compact = re.sub(r"[_\-\s]+", "", stem).lower()
    if compact in _AMBIGUOUS_FILENAME_STEMS:
        return True
    if _AMBIGUOUS_FILENAME_RE.match(stem):
        return True
    head = re.split(r"[_\-\s]+", stem, maxsplit=1)[0].upper()
    return head in {"TEMP", "TMP", "FILE", "DOWNLOAD", "DOCUMENT", "UNTITLED", "SCAN", "COPY"}


def identity_from_filename(filename: str) -> str | None:
    """Return a document number when the filename itself is a clear identity."""
    if _is_ambiguous_filename(filename):
        return None
    stem = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    stem = re.sub(r"\.[^.]+$", "", stem).strip()
    if not stem:
        return None
    candidates = (
        stem,
        stem.replace("_", " "),
        stem.replace("-", " "),
        stem.replace("_", "-"),
    )
    for candidate in candidates:
        found = normalize_document_number(candidate)
        if found:
            return found
    numbers = find_document_numbers(stem.replace("_", " "))
    if len(numbers) == 1:
        return numbers[0]
    return None


def classify_document_type(
    *,
    leading_text: str,
    document_number: str | None,
    catalog_like: bool,
) -> DocumentType:
    """Classify the publication type without promoting mentioned references."""
    if _INDEX_HINT_RE.search(leading_text):
        return "index"
    if catalog_like or _CATALOG_HINT_RE.search(leading_text):
        if _LIST_HINT_RE.search(leading_text):
            return "list"
        return "catalog"
    if _MANUAL_HINT_RE.search(leading_text):
        return "manual"
    if document_number:
        if _FAMILY_PREFIX_RE.match(document_number):
            return "standard"
        return "technical_document"
    return "unknown"


def resolve_document_identity(
    *,
    source_filename: str,
    leading_text: str,
    pdf_title: str | None = None,
    document_fingerprint: str | None = None,
) -> DocumentIdentity:
    """Resolve the document-owning identity or fail closed without inventing numbers."""
    filename_id = identity_from_filename(source_filename)
    title_id = normalize_document_number(pdf_title) if pdf_title else None
    candidates = find_document_numbers(leading_text)
    base_candidates = tuple(dict.fromkeys(_base_document_number(item) for item in candidates))
    catalog_like = bool(_CATALOG_HINT_RE.search(leading_text)) or (
        len(base_candidates) >= _MULTI_DOCUMENT_THRESHOLD and filename_id is None
    )
    header_title = _title_from_leading_text(leading_text, filename_id or title_id)
    synthetic_id = f"doc_{(document_fingerprint or 'unknown')[:16]}"

    def _reference_only(
        *,
        document_type: DocumentType,
        source: str,
        reason: str,
        title: str | None = None,
        confidence: float = 0.0,
    ) -> DocumentIdentity:
        return DocumentIdentity(
            document_id=synthetic_id if document_fingerprint else None,
            title=title or header_title or ((pdf_title or "").strip() or None),
            document_number=None,
            confidence=confidence,
            source=source,
            status="unresolved",
            reason=reason,
            document_type=document_type,
            recommendation_eligible=False,
            candidates=candidates,
            ingestible=True,
        )

    if filename_id:
        if title_id and _base_document_number(title_id) != _base_document_number(filename_id):
            return DocumentIdentity(
                document_id=None,
                title=None,
                document_number=None,
                confidence=0.0,
                source="conflict",
                status="unresolved",
                reason=(
                    f"Filename identity {filename_id} conflicts with PDF title identity "
                    f"{title_id}. Rename the PDF or supply a single-document source."
                ),
                document_type="unknown",
                recommendation_eligible=False,
                candidates=candidates,
                ingestible=False,
            )
        filename_base = _base_document_number(filename_id)
        filename_supported_by_content = filename_base in base_candidates or (
            title_id is not None and _base_document_number(title_id) == filename_base
        )
        # Catalog/list bodies win over download-style filenames that only look like ids.
        if catalog_like and not filename_supported_by_content:
            document_type = classify_document_type(
                leading_text=leading_text,
                document_number=None,
                catalog_like=True,
            )
            return _reference_only(
                document_type=document_type,
                source="catalog",
                reason=(
                    "Document looks like a catalog/list that mentions multiple document "
                    "numbers, and the filename identity does not appear as the document "
                    "owner in the leading pages. Retained as reference-only; mentioned "
                    "numbers are not promoted."
                ),
                title=header_title or ((pdf_title or "").strip() or None),
            )
        document_type = classify_document_type(
            leading_text=leading_text,
            document_number=filename_id,
            catalog_like=False,
        )
        return DocumentIdentity(
            document_id=_stable_document_id(filename_id),
            title=header_title or (pdf_title.strip() if pdf_title else None),
            document_number=filename_id,
            confidence=0.95,
            source="filename",
            status="resolved",
            reason=f"Resolved from filename ({source_filename}).",
            document_type=document_type,
            recommendation_eligible=document_type not in {"catalog", "list", "index"},
            candidates=candidates,
            ingestible=True,
        )

    if title_id and not catalog_like:
        document_type = classify_document_type(
            leading_text=leading_text,
            document_number=title_id,
            catalog_like=False,
        )
        return DocumentIdentity(
            document_id=_stable_document_id(title_id),
            title=(pdf_title or "").strip() or header_title,
            document_number=title_id,
            confidence=0.85,
            source="pdf_title",
            status="resolved",
            reason="Resolved from PDF metadata title.",
            document_type=document_type,
            recommendation_eligible=document_type not in {"catalog", "list", "index"},
            candidates=candidates,
            ingestible=True,
        )

    if catalog_like:
        document_type = classify_document_type(
            leading_text=leading_text,
            document_number=None,
            catalog_like=True,
        )
        return _reference_only(
            document_type=document_type,
            source="catalog",
            reason=(
                "Document looks like a catalog/list/index that mentions multiple document "
                "numbers. Retained as reference-only; mentioned references are not "
                "promoted to document identity."
            ),
            title=header_title or ((pdf_title or "").strip() or None),
        )

    if len(base_candidates) == 1:
        chosen = candidates[0]
        document_type = classify_document_type(
            leading_text=leading_text,
            document_number=chosen,
            catalog_like=False,
        )
        return DocumentIdentity(
            document_id=_stable_document_id(chosen),
            title=header_title,
            document_number=chosen,
            confidence=0.75,
            source="header",
            status="resolved",
            reason="Resolved from a single document-level identifier in leading pages.",
            document_type=document_type,
            recommendation_eligible=document_type not in {"catalog", "list", "index"},
            candidates=candidates,
            ingestible=True,
        )

    if not candidates:
        document_type = classify_document_type(
            leading_text=leading_text,
            document_number=None,
            catalog_like=False,
        )
        title = header_title or ((pdf_title or "").strip() or None)
        return DocumentIdentity(
            document_id=synthetic_id if document_fingerprint else None,
            title=title,
            document_number=None,
            confidence=0.25,
            source="unresolved",
            status="unresolved",
            reason=(
                "Could not establish a document number from filename, title, or leading "
                "pages. Document may still be retained as reference-only."
            ),
            document_type=document_type,
            recommendation_eligible=False,
            candidates=candidates,
            ingestible=True,
        )

    document_type = classify_document_type(
        leading_text=leading_text,
        document_number=None,
        catalog_like=False,
    )
    return _reference_only(
        document_type=document_type if document_type != "unknown" else "unknown",
        source="ambiguous",
        reason=(
            "Ambiguous document identity; multiple document numbers appear without a "
            f"clear document owner ({', '.join(candidates[:8])}"
            f"{'...' if len(candidates) > 8 else ''}). "
            "Retained as reference-only; mentioned numbers are not promoted."
        ),
        title=header_title or ((pdf_title or "").strip() or None),
        confidence=0.0,
    )


def _stable_document_id(document_number: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", document_number).strip("_")


def _title_from_leading_text(text: str, document_number: str | None) -> str | None:
    lines = [line.strip() for line in normalize_text(text).splitlines() if line.strip()]
    if document_number:
        for index, line in enumerate(lines):
            found = normalize_document_number(line)
            if found and _base_document_number(found) == _base_document_number(document_number):
                for following in lines[index + 1 : index + 4]:
                    if _looks_like_title(following):
                        return following
    for line in lines[:8]:
        if _looks_like_title(line) and not normalize_document_number(line):
            return line
    return None


def _looks_like_title(value: str) -> bool:
    if not (4 <= len(value) <= 220):
        return False
    lowered = value.lower()
    if lowered.startswith(("copyright", "bureau of", "page ", "table ")):
        return False
    if re.match(
        r"^(?:\d+(?:\.\d+)*\.?\s+|annex\b|appendix\b|first\b|second\b|third\b|fourth\b)",
        value,
        re.I,
    ):
        return False
    return bool(re.search(r"[A-Za-z]", value))


def _title_near_identity(text: str, standard_number: str) -> str | None:
    """Backward-compatible helper retained for older call sites."""
    return _title_from_leading_text(text, standard_number)
