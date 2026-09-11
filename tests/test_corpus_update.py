"""Generic PDF corpus update workflow tests.

Fixture PDFs are synthetic test inputs only. They are never treated as official
source PDFs and must not enter the real ``officialBIS/`` corpus.
"""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf
import pytest

from corpus.processing.normalizer import normalize_document_number
from corpus.update.identity import identity_from_filename, resolve_document_identity
from corpus.update.workflow import (
    document_fingerprint,
    update_corpus_from_source,
)
from retrieval.corpus_loader import CorpusChunkLoader
from retrieval.indexing.persistence import IndexPersistence
from retrieval.pipeline import RetrievalPipeline
from retrieval.config import RetrievalConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_BIS = PROJECT_ROOT / "officialBIS"


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


class StaticEncoder:
    model_name = "static-test"
    backend_name = "custom"
    model_version = "test"
    normalize_embeddings = True
    dimension = 4

    def encode_batch(self, texts: list[str]):
        import numpy as np

        vectors = []
        for text in texts:
            seed = sum(ord(ch) for ch in text) % 97
            raw = np.array([seed + 1, seed + 2, seed + 3, seed + 4], dtype=np.float32)
            vectors.append(raw / np.linalg.norm(raw))
        return np.asarray(vectors, dtype=np.float32)

    def encode_text(self, text: str):
        return self.encode_batch([text])[0]


def test_temp_filename_never_becomes_standard_number() -> None:
    assert identity_from_filename("temp_123.pdf") is None
    assert identity_from_filename("temp.pdf") is None
    assert identity_from_filename("document.pdf") is None
    assert identity_from_filename("download.pdf") is None
    assert identity_from_filename("file.pdf") is None
    assert normalize_document_number("TEMP-123") is None
    assert normalize_document_number("temp_2131689472") is None
    outcome = resolve_document_identity(
        source_filename="temp_123.pdf",
        leading_text="List of products notified\nIS 616\nIS 13252\nIS 10322\nIS 302",
        document_fingerprint="f" * 64,
    )
    assert outcome.standard_number is None
    assert outcome.status == "unresolved"
    assert outcome.recommendation_eligible is False
    assert outcome.document_type in {"catalog", "list"}
    assert outcome.document_id.startswith("doc_")


def test_identity_resolves_generic_owner_and_rejects_catalog_promotion() -> None:
    catalog = resolve_document_identity(
        source_filename="product-list.pdf",
        leading_text="List of products\nTECH-1001\nTECH-1002\nTECH-1003\nTECH-1004",
        document_fingerprint="a" * 64,
    )
    assert catalog.resolved is False
    assert catalog.ingestible is True
    assert catalog.document_number is None
    assert catalog.recommendation_eligible is False
    assert catalog.document_type == "list"

    owned = resolve_document_identity(
        source_filename="TECH-1001.pdf",
        leading_text="TECH-1001\nTechnical requirements\nReferences\nSTD-A\nDOC-2026-001",
        document_fingerprint="b" * 64,
    )
    assert owned.resolved is True
    assert owned.document_number == "TECH-1001"
    assert owned.ingestible is True
    assert owned.recommendation_eligible is True


def test_mentioned_family_numbers_are_not_document_identity() -> None:
    outcome = resolve_document_identity(
        source_filename="engineering_manual.pdf",
        leading_text=(
            "Engineering notes\nReferences include IS 5120, ISO 9001, IEC 60034, "
            "ASTM A36 and EN 10025 for related equipment only."
        ),
        document_fingerprint="c" * 64,
    )
    assert outcome.document_number is None
    assert outcome.recommendation_eligible is False
    assert outcome.status == "unresolved"


def test_identity_supports_iso_and_unresolved_manual() -> None:
    iso = resolve_document_identity(
        source_filename="ISO_9001.pdf",
        leading_text="ISO 9001\nQuality management systems\nRequirements",
        document_fingerprint="d" * 64,
    )
    assert iso.resolved is True
    assert iso.document_number == "ISO 9001"
    assert iso.document_type == "standard"
    assert iso.recommendation_eligible is True

    manual = resolve_document_identity(
        source_filename="engineering_manual.pdf",
        leading_text="Engineering installation manual\n1 Scope\nThis manual covers site assembly.",
        document_fingerprint="e" * 64,
    )
    assert manual.resolved is False
    assert manual.ingestible is True
    assert manual.document_number is None
    assert manual.recommendation_eligible is False
    assert manual.document_type == "manual"


def test_update_corpus_one_pdf_and_provenance(tmp_path: Path) -> None:
    source = tmp_path / "officialBIS"
    write_technical_pdf(
        source / "TECH-1001.pdf",
        header_lines=["TECH-1001", "Technical requirements for general-purpose equipment."],
        body="1 Scope\nThis document covers general-purpose equipment for industrial water service.",
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
    assert summary.documents_eligible == 1
    assert summary.chunks_generated >= 1
    assert summary.chunks_eligible == summary.chunks_generated
    assert summary.failed_documents == 0

    loaded = CorpusChunkLoader().load(data_dir / "corpus.chunks.jsonl")
    chunk = loaded.chunks[0]
    assert chunk.standard_number == "TECH-1001"
    assert chunk.source_file == "TECH-1001.pdf"
    assert chunk.source_path == "TECH-1001.pdf"
    assert chunk.document_fingerprint
    assert chunk.document_id
    assert chunk.recommendation_eligible is True
    assert chunk.document_type == "technical_document"
    assert chunk.page_start >= 1
    assert "equipment" in chunk.text.lower()
    assert "TEMP-" not in (chunk.standard_number or "")

    document = json.loads((data_dir / "documents" / "TECH_1001.json").read_text(encoding="utf-8"))
    assert document["metadata"]["standard_number"] == "TECH-1001"
    assert document["metadata"]["document_fingerprint"] == chunk.document_fingerprint
    assert document["metadata"]["source_path"] == "TECH-1001.pdf"
    assert document["metadata"]["recommendation_eligible"] is True


def test_catalog_with_many_references_is_reference_only(tmp_path: Path) -> None:
    source = tmp_path / "officialBIS"
    write_technical_pdf(
        source / "temp_123.pdf",
        header_lines=["List of Major Consumer Products Notified for Compulsory Marking"],
        body=(
            "IS 616 Audio\nIS 13252 IT equipment\nIS 10322 Luminaires\n"
            "IS 302 Appliances\nIS 15885 LED drivers"
        ),
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
    assert summary.documents_catalog_list == 1
    assert summary.documents_eligible == 0
    assert summary.chunks_eligible == 0
    item = summary.processed[0]
    assert item.standard_number is None
    assert item.recommendation_eligible is False
    assert item.document_type in {"catalog", "list"}
    assert not str(item.standard_number or "").startswith("TEMP")
    chunk = CorpusChunkLoader().load(root / "corpus" / "data" / "corpus.chunks.jsonl").chunks[0]
    assert chunk.standard_number is None
    assert chunk.recommendation_eligible is False
    eligible = (root / "corpus" / "data" / "corpus.eligible.chunks.jsonl").read_text(encoding="utf-8")
    assert eligible.strip() == ""


def test_unresolved_docs_excluded_from_recommendation_index(tmp_path: Path) -> None:
    source = tmp_path / "officialBIS"
    write_technical_pdf(
        source / "engineering_manual.pdf",
        header_lines=["Engineering installation manual"],
        body="1 Scope\nThis manual covers site assembly of modular equipment skids.",
        pages=2,
    )
    write_technical_pdf(
        source / "TECH-1001.pdf",
        header_lines=["TECH-1001", "Technical requirements"],
        body="1 Scope\nGeneral-purpose equipment requirements.",
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
    assert summary.documents_total == 2
    assert summary.documents_eligible == 1
    eligible_path = data_dir / "corpus.eligible.chunks.jsonl"
    loaded_eligible = CorpusChunkLoader().load(eligible_path)
    assert len(loaded_eligible.chunks) == summary.chunks_eligible
    assert all(chunk.recommendation_eligible for chunk in loaded_eligible.chunks)
    assert all(chunk.standard_number == "TECH-1001" for chunk in loaded_eligible.chunks)


def test_faiss_vector_count_matches_eligible_chunks(tmp_path: Path) -> None:
    source = tmp_path / "officialBIS"
    write_technical_pdf(
        source / "temp_99.pdf",
        header_lines=["List of products"],
        body="IS 1 Alpha\nIS 2 Beta\nIS 3 Gamma\nIS 4 Delta",
    )
    write_technical_pdf(
        source / "DOC-2026-001.pdf",
        header_lines=["DOC-2026-001", "Generic technical document"],
        body="1 Scope\nCovers generic non-domain equipment acceptance tests.",
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
    assert summary.documents_eligible == 1
    eligible = CorpusChunkLoader().load(data_dir / "corpus.eligible.chunks.jsonl").chunks
    pipeline = RetrievalPipeline(
        RetrievalConfig(embedding_model="static-test", embedding_dimension=4, allow_cpu_fallback=True),
        encoder=StaticEncoder(),
    )
    pipeline.build_and_persist(data_dir / "corpus.eligible.chunks.jsonl", index_dir)
    bundle = IndexPersistence().load(
        index_dir,
        expected_embedding_backend="custom",
        expected_embedding_model="static-test",
        expected_embedding_model_version="test",
        expected_dimension=4,
        expected_normalize_embeddings=True,
    )
    assert bundle.index.count == len(eligible)
    assert bundle.metadata.chunk_count == len(eligible)


def test_empty_eligible_index_is_allowed(tmp_path: Path) -> None:
    source = tmp_path / "officialBIS"
    write_technical_pdf(
        source / "product-list.pdf",
        header_lines=["List of products"],
        body="TECH-1001 Alpha\nTECH-1002 Beta\nTECH-1003 Gamma\nTECH-1004 Delta",
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
    assert summary.documents_eligible == 0
    assert summary.chunks_eligible == 0
    pipeline = RetrievalPipeline(
        RetrievalConfig(embedding_model="static-test", embedding_dimension=4, allow_cpu_fallback=True),
        encoder=StaticEncoder(),
    )
    pipeline.build_and_persist(data_dir / "corpus.eligible.chunks.jsonl", index_dir)
    bundle = IndexPersistence().load(
        index_dir,
        expected_embedding_backend="custom",
        expected_embedding_model="static-test",
        expected_embedding_model_version="test",
        expected_dimension=4,
        expected_normalize_embeddings=True,
    )
    assert bundle.index.count == 0
    assert bundle.metadata.chunk_count == 0


def test_update_corpus_multiple_independent_pdfs(tmp_path: Path) -> None:
    source = tmp_path / "officialBIS"
    write_technical_pdf(
        source / "TECH-1001.pdf",
        header_lines=["TECH-1001", "Equipment requirements"],
        body="1 Scope\nGeneral-purpose equipment.",
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
    assert summary.documents_eligible == 2
    standards = {item.standard_number for item in summary.processed}
    assert standards == {"TECH-1001", "STD-A"}


def test_update_corpus_is_idempotent(tmp_path: Path) -> None:
    source = tmp_path / "officialBIS"
    write_technical_pdf(
        source / "TECH-1001.pdf",
        header_lines=["TECH-1001", "Equipment requirements"],
        body="1 Scope\nGeneral-purpose equipment.",
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
        header_lines=["TECH-1001", "Equipment requirements"],
        body="1 Scope\nGeneral-purpose equipment.",
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
        header_lines=["TECH-1001", "Equipment requirements"],
        body="1 Scope\nOriginal body text for equipment.",
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
        header_lines=["TECH-1001", "Equipment requirements"],
        body="1 Scope\nReplacement body text for equipment with more detail.",
    )
    assert document_fingerprint(pdf.read_bytes()) != first_fp
    second = update_corpus_from_source(**kwargs)
    assert second.processed[0].document_fingerprint != first_fp
    check = update_corpus_from_source(**kwargs, check_only=True)
    assert check.would_change is False


def test_ambiguous_identity_fails_closed_as_reference_only(tmp_path: Path) -> None:
    source = tmp_path / "officialBIS"
    write_technical_pdf(
        source / "mixed.pdf",
        header_lines=["Introduction"],
        body="This note mentions TECH-1001 and STD-A without owning either publication.",
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
    item = summary.processed[0]
    assert item.standard_number is None
    assert item.recommendation_eligible is False
    assert item.identity_status == "unresolved"


def test_mentioned_documents_are_not_promoted(tmp_path: Path) -> None:
    source = tmp_path / "officialBIS"
    write_technical_pdf(
        source / "TECH-1001.pdf",
        header_lines=["TECH-1001", "Technical requirements for general-purpose equipment."],
        body="1 Scope\nCovers equipment.\n2 References\nReferences include STD-A and DOC-2026-001 for related equipment only.",
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
        body="1 Scope\nThis manual covers site assembly of modular equipment skids.",
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
    assert summary.documents_eligible == 0
    item = summary.processed[0]
    assert item.standard_number is None
    assert item.document_id.startswith("doc_")
    chunk = CorpusChunkLoader().load(root / "corpus" / "data" / "corpus.chunks.jsonl").chunks[0]
    assert chunk.standard_number is None
    assert chunk.document_id == item.document_id
    assert chunk.recommendation_eligible is False


def test_generic_non_is_document_parses_and_chunks(tmp_path: Path) -> None:
    source = tmp_path / "officialBIS"
    write_technical_pdf(
        source / "DOC-2026-001.pdf",
        header_lines=["DOC-2026-001", "Multi-section technical specification"],
        body=(
            "1 Scope\nCovers multi-section equipment.\n\n"
            "2 Requirements\nRequirement clause detailing performance limits.\n\n"
            "2.1 Materials\nSteel and elastomer materials.\n\n"
            "3 Testing\nFactory acceptance testing."
        ),
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
    assert summary.documents_eligible == 1
    item = summary.processed[0]
    assert item.standard_number == "DOC-2026-001"
    assert item.document_type == "technical_document"
    assert "IS " not in item.standard_number
    assert item.document_id == "DOC_2026_001"


def test_no_domain_specific_parser_assumptions_in_identity_module() -> None:
    source = (PROJECT_ROOT / "corpus" / "update" / "identity.py").read_text(encoding="utf-8")
    for banned in ("IS 5120", "IS 1520", "IS 8034", "centrifugal pump", "IS 325", "IS 9137"):
        assert banned not in source


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
        header_lines=["TECH-1001", "Equipment requirements"],
        body="1 Scope\nGeneral-purpose equipment.",
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
        header_lines=["TECH-1001", "Equipment requirements"],
        body="1 Scope\nFirst authoritative body.",
    )
    write_technical_pdf(
        source / "archive" / "TECH-1001.pdf",
        header_lines=["TECH-1001", "Equipment requirements"],
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


def test_fixture_pdfs_never_written_into_officialbis(tmp_path: Path) -> None:
    before = {path.name for path in OFFICIAL_BIS.glob("*.pdf")} if OFFICIAL_BIS.is_dir() else set()
    source = tmp_path / "fixtures" / "temp"
    write_technical_pdf(
        source / "temp_fixture_only.pdf",
        header_lines=["Fixture only"],
        body="1 Scope\nSynthetic fixture content for tests.",
    )
    root = tmp_path / "project"
    update_corpus_from_source(
        project_root=root,
        source_dir=source,
        corpus_data_dir=root / "corpus" / "data",
        index_dir=root / "retrieval-index",
        rebuild_index=False,
    )
    after = {path.name for path in OFFICIAL_BIS.glob("*.pdf")} if OFFICIAL_BIS.is_dir() else set()
    assert after == before
    assert not (OFFICIAL_BIS / "temp_fixture_only.pdf").exists()
