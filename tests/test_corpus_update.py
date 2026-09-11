"""Generic PDF corpus update workflow tests.

Fixture PDFs are synthetic test inputs only. They are never treated as official
source PDFs and must not enter the real ``officialBIS/`` corpus.
"""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf
import pytest

from corpus.update.identity import resolve_document_identity
from corpus.update.workflow import (
    document_fingerprint,
    update_corpus_from_source,
)
from retrieval.corpus_loader import CorpusChunkLoader


def write_technical_pdf(
    path: Path,
    *,
    header_lines: list[str],
    body: str,
    pages: int = 1,
    repeated_footer: str | None = None,
) -> None:
    """Write a deterministic multi-line technical PDF for tests only."""
    doc = pymupdf.open()
    raw_lines = [*header_lines, "", *body.splitlines()]
    if repeated_footer:
        raw_lines.extend(["", repeated_footer])
    lines: list[str] = []
    for line in raw_lines:
        if not line:
            lines.append("")
            continue
        text = line
        while text:
            lines.append(text[:100])
            text = text[100:]
    # Spread content across pages so oversized sections remain extractable.
    per_page = max(12, (len(lines) + pages - 1) // max(pages, 1))
    for page_index in range(max(pages, 1)):
        page = doc.new_page()
        chunk = lines[page_index * per_page : (page_index + 1) * per_page]
        if not chunk and page_index == 0:
            chunk = lines
        y = 72.0
        for line in chunk:
            if line:
                page.insert_text((72, y), line, fontsize=11, fontname="helv")
            y += 14
            if y > 800:
                break
        page.insert_text((72, 820), f"Page {page_index + 1}", fontsize=9, fontname="helv")
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    doc.close()


def test_identity_resolves_generic_owner_and_rejects_catalogs() -> None:
    catalog = resolve_document_identity(
        source_filename="product-list.pdf",
        leading_text="List of products\nTECH-1001\nTECH-1002\nTECH-1003\nTECH-1004",
    )
    assert catalog.resolved is False
    assert catalog.ingestible is False
    assert catalog.document_number is None

    owned = resolve_document_identity(
        source_filename="TECH-1001.pdf",
        leading_text="TECH-1001\nTechnical requirements\nReferences\nSTD-A\nDOC-2026-001",
    )
    assert owned.resolved is True
    assert owned.document_number == "TECH-1001"
    assert owned.ingestible is True


def test_identity_supports_iso_and_unresolved_manual() -> None:
    iso = resolve_document_identity(
        source_filename="ISO_9001.pdf",
        leading_text="ISO 9001\nQuality management systems\nRequirements",
    )
    assert iso.resolved is True
    assert iso.document_number == "ISO 9001"

    manual = resolve_document_identity(
        source_filename="engineering_manual.pdf",
        leading_text="Engineering installation manual\n1 Scope\nThis manual covers site assembly.",
    )
    assert manual.resolved is False
    assert manual.ingestible is True
    assert manual.document_number is None


def test_update_corpus_one_pdf_and_provenance(tmp_path: Path) -> None:
    source = tmp_path / "officialBIS"
    write_technical_pdf(
        source / "TECH-1001.pdf",
        header_lines=["TECH-1001", "Technical requirements for general-purpose centrifugal pumps."],
        body="1 Scope\nThis document covers general-purpose centrifugal pumps for industrial water service.",
    )
    root = tmp_path / "project"
    data_dir = root / "corpus" / "data"
    index_dir = root / "retrieval-index"

    summary = update_corpus_from_source(
        project_root=root,
        source_dir=source,
        corpus_data_dir=data_dir,
        index_dir=index_dir,
        rebuild_index=False,
    )
    assert summary.successfully_processed == 1
    assert summary.standards_indexed == 1
    assert summary.chunks_generated >= 1
    assert summary.failed_documents == 0

    loaded = CorpusChunkLoader().load(data_dir / "corpus.chunks.jsonl")
    chunk = loaded.chunks[0]
    assert chunk.standard_number == "TECH-1001"
    assert chunk.source_file == "TECH-1001.pdf"
    assert chunk.source_path == "TECH-1001.pdf"
    assert chunk.document_fingerprint
    assert chunk.document_id
    assert chunk.page_start >= 1
    assert "centrifugal" in chunk.text.lower()

    document = json.loads((data_dir / "documents" / "TECH_1001.json").read_text(encoding="utf-8"))
    assert document["metadata"]["standard_number"] == "TECH-1001"
    assert document["metadata"]["document_fingerprint"] == chunk.document_fingerprint
    assert document["metadata"]["source_path"] == "TECH-1001.pdf"


def test_update_corpus_multiple_independent_pdfs(tmp_path: Path) -> None:
    source = tmp_path / "officialBIS"
    write_technical_pdf(
        source / "TECH-1001.pdf",
        header_lines=["TECH-1001", "Centrifugal pumps"],
        body="1 Scope\nGeneral-purpose centrifugal pumps.",
    )
    write_technical_pdf(
        source / "STD-A.pdf",
        header_lines=["STD-A", "Waterproofing compound"],
        body="1 Scope\nWaterproofing for concrete.",
    )
    root = tmp_path / "project"

    summary = update_corpus_from_source(
        project_root=root,
        source_dir=source,
        corpus_data_dir=root / "corpus" / "data",
        index_dir=root / "retrieval-index",
        rebuild_index=False,
    )
    assert summary.successfully_processed == 2
    assert summary.standards_indexed == 2
    standards = {item.standard_number for item in summary.processed}
    assert standards == {"TECH-1001", "STD-A"}


def test_update_corpus_is_idempotent(tmp_path: Path) -> None:
    source = tmp_path / "officialBIS"
    write_technical_pdf(
        source / "TECH-1001.pdf",
        header_lines=["TECH-1001", "Centrifugal pumps"],
        body="1 Scope\nGeneral-purpose centrifugal pumps.",
    )
    root = tmp_path / "project"
    kwargs = dict(
        project_root=root,
        source_dir=source,
        corpus_data_dir=root / "corpus" / "data",
        index_dir=root / "retrieval-index",
        rebuild_index=False,
    )
    first = update_corpus_from_source(**kwargs)
    chunks_first = (root / "corpus" / "data" / "corpus.chunks.jsonl").read_text(encoding="utf-8")
    manifest_first = (root / "corpus" / "data" / "manifest.json").read_text(encoding="utf-8")
    second = update_corpus_from_source(**kwargs)
    assert first.chunks_generated == second.chunks_generated
    assert chunks_first == (root / "corpus" / "data" / "corpus.chunks.jsonl").read_text(encoding="utf-8")
    assert manifest_first == (root / "corpus" / "data" / "manifest.json").read_text(encoding="utf-8")


def test_duplicate_pdf_bytes_are_skipped(tmp_path: Path) -> None:
    source = tmp_path / "officialBIS"
    write_technical_pdf(
        source / "TECH-1001.pdf",
        header_lines=["TECH-1001", "Centrifugal pumps"],
        body="1 Scope\nGeneral-purpose centrifugal pumps.",
    )
    duplicate = source / "copy" / "TECH-1001.pdf"
    duplicate.parent.mkdir(parents=True)
    duplicate.write_bytes((source / "TECH-1001.pdf").read_bytes())
    root = tmp_path / "project"

    summary = update_corpus_from_source(
        project_root=root,
        source_dir=source,
        corpus_data_dir=root / "corpus" / "data",
        index_dir=root / "retrieval-index",
        rebuild_index=False,
    )
    assert summary.duplicates_skipped == 1
    assert summary.successfully_processed == 1


def test_replacing_pdf_changes_fingerprint(tmp_path: Path) -> None:
    source = tmp_path / "officialBIS"
    pdf = source / "TECH-1001.pdf"
    write_technical_pdf(
        pdf,
        header_lines=["TECH-1001", "Centrifugal pumps"],
        body="1 Scope\nOriginal body text for pumps.",
    )
    root = tmp_path / "project"
    kwargs = dict(
        project_root=root,
        source_dir=source,
        corpus_data_dir=root / "corpus" / "data",
        index_dir=root / "retrieval-index",
        rebuild_index=False,
    )
    first = update_corpus_from_source(**kwargs)
    first_fp = first.processed[0].document_fingerprint
    write_technical_pdf(
        pdf,
        header_lines=["TECH-1001", "Centrifugal pumps"],
        body="1 Scope\nReplacement body text for pumps with more detail.",
    )
    assert document_fingerprint(pdf.read_bytes()) != first_fp
    second = update_corpus_from_source(**kwargs)
    assert second.processed[0].document_fingerprint != first_fp
    check = update_corpus_from_source(**kwargs, check_only=True)
    assert check.would_change is False


def test_ambiguous_identity_fails_actionably(tmp_path: Path) -> None:
    source = tmp_path / "officialBIS"
    write_technical_pdf(
        source / "mixed.pdf",
        header_lines=["Product list", "Annexure"],
        body="TECH-1001 Alpha\nTECH-1002 Beta\nTECH-1003 Gamma\nTECH-1004 Delta",
    )
    root = tmp_path / "project"
    with pytest.raises(RuntimeError, match="No authentic PDFs produced"):
        update_corpus_from_source(
            project_root=root,
            source_dir=source,
            corpus_data_dir=root / "corpus" / "data",
            index_dir=root / "retrieval-index",
            rebuild_index=False,
        )


def test_mentioned_documents_are_not_promoted(tmp_path: Path) -> None:
    source = tmp_path / "officialBIS"
    write_technical_pdf(
        source / "TECH-1001.pdf",
        header_lines=["TECH-1001", "Technical requirements for general-purpose centrifugal pumps."],
        body="1 Scope\nCovers centrifugal pumps.\n2 References\nReferences include STD-A and DOC-2026-001 for related equipment only.",
    )
    root = tmp_path / "project"
    summary = update_corpus_from_source(
        project_root=root,
        source_dir=source,
        corpus_data_dir=root / "corpus" / "data",
        index_dir=root / "retrieval-index",
        rebuild_index=False,
    )
    assert {item.standard_number for item in summary.processed} == {"TECH-1001"}
    document = json.loads((root / "corpus" / "data" / "documents" / "TECH_1001.json").read_text(encoding="utf-8"))
    related = {row["target_standard_number"] for row in document.get("relationships", [])}
    assert "STD-A" in related or "DOC-2026-001" in related
    assert document["metadata"]["standard_number"] == "TECH-1001"


def test_unresolved_manual_is_ingested_without_invented_number(tmp_path: Path) -> None:
    source = tmp_path / "officialBIS"
    write_technical_pdf(
        source / "engineering_manual.pdf",
        header_lines=["Engineering installation manual"],
        body="1 Scope\nThis manual covers site assembly of modular pump skids.",
        pages=2,
    )
    root = tmp_path / "project"
    summary = update_corpus_from_source(
        project_root=root,
        source_dir=source,
        corpus_data_dir=root / "corpus" / "data",
        index_dir=root / "retrieval-index",
        rebuild_index=False,
    )
    assert summary.successfully_processed == 1
    assert summary.unresolved_identity == 1
    item = summary.processed[0]
    assert item.standard_number is None
    assert item.document_id.startswith("doc_")
    chunk = CorpusChunkLoader().load(root / "corpus" / "data" / "corpus.chunks.jsonl").chunks[0]
    assert chunk.standard_number is None
    assert chunk.document_id == item.document_id


def test_sections_and_large_section_split(tmp_path: Path) -> None:
    source = tmp_path / "officialBIS"
    long_paragraph = " ".join(
        ["Requirement clause detailing performance limits and tolerances."] * 45
    )
    write_technical_pdf(
        source / "DOC-2026-001.pdf",
        header_lines=["DOC-2026-001", "Multi-section technical specification"],
        body=(
            f"1 Scope\nCovers multi-section equipment.\n\n"
            f"2 Requirements\n{long_paragraph}\n\n"
            f"2.1 Materials\nSteel and elastomer materials.\n\n"
            f"3 Testing\nFactory acceptance testing."
        ),
        pages=4,
        repeated_footer="CONFIDENTIAL FOOTER DO NOT INDEX AS IDENTITY",
    )
    root = tmp_path / "project"
    summary = update_corpus_from_source(
        project_root=root,
        source_dir=source,
        corpus_data_dir=root / "corpus" / "data",
        index_dir=root / "retrieval-index",
        rebuild_index=False,
    )
    assert summary.successfully_processed == 1
    assert summary.chunks_generated >= 2
    document = json.loads(
        (root / "corpus" / "data" / "documents" / "DOC_2026_001.json").read_text(encoding="utf-8")
    )
    assert document["metadata"]["standard_number"] == "DOC-2026-001"
    assert len(document["sections"]) >= 2


def test_check_mode_does_not_write(tmp_path: Path) -> None:
    source = tmp_path / "officialBIS"
    write_technical_pdf(
        source / "TECH-1001.pdf",
        header_lines=["TECH-1001", "Centrifugal pumps"],
        body="1 Scope\nGeneral-purpose centrifugal pumps.",
    )
    root = tmp_path / "project"
    summary = update_corpus_from_source(
        project_root=root,
        source_dir=source,
        corpus_data_dir=root / "corpus" / "data",
        index_dir=root / "retrieval-index",
        check_only=True,
        rebuild_index=False,
    )
    assert summary.check_only is True
    assert summary.would_change is True
    assert not (root / "corpus" / "data" / "corpus.chunks.jsonl").exists()
    assert not (root / "retrieval-index").exists()


def test_corrupt_pdf_failure_is_surfaced(tmp_path: Path) -> None:
    source = tmp_path / "officialBIS"
    bad = source / "broken.pdf"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"not a pdf")
    root = tmp_path / "project"
    with pytest.raises(RuntimeError, match="No authentic PDFs produced"):
        update_corpus_from_source(
            project_root=root,
            source_dir=source,
            corpus_data_dir=root / "corpus" / "data",
            index_dir=root / "retrieval-index",
            rebuild_index=False,
        )


def test_same_document_number_different_content_is_not_merged(tmp_path: Path) -> None:
    source = tmp_path / "officialBIS"
    write_technical_pdf(
        source / "TECH-1001.pdf",
        header_lines=["TECH-1001", "Centrifugal pumps"],
        body="1 Scope\nFirst authoritative body.",
    )
    write_technical_pdf(
        source / "archive" / "TECH-1001.pdf",
        header_lines=["TECH-1001", "Centrifugal pumps"],
        body="1 Scope\nSecond distinct body that must not overwrite the first.",
    )
    root = tmp_path / "project"
    summary = update_corpus_from_source(
        project_root=root,
        source_dir=source,
        corpus_data_dir=root / "corpus" / "data",
        index_dir=root / "retrieval-index",
        rebuild_index=False,
    )
    assert summary.successfully_processed == 1
    assert any(item.error_class == "document_number_conflict" for item in summary.failures)
