from __future__ import annotations

import json

import pymupdf

from corpus import CorpusBuilder
from corpus.chunking import ChunkingConfig, DocumentChunker
from corpus.models import DocumentChunk, Section, SectionFragment, StandardDocument, StandardMetadata, StandardRelationship
from corpus.validation import CorpusValidator


def section(text: str) -> Section:
    return Section("4.1", "Requirements", 2, 2, 2, text, (SectionFragment(2, text),))


def test_chunking_is_deterministic_and_keeps_provenance() -> None:
    text = "Paragraph one contains a tightly coupled clause.\n\n" + "Paragraph two contains a second clause.\n\n" + "Paragraph three completes the requirement."
    chunker = DocumentChunker(ChunkingConfig(max_chars=100, overlap_chars=20))
    first = chunker.chunk((section(text),), "IS 2645:2003", "is2645.pdf")
    second = chunker.chunk((section(text),), "IS 2645:2003", "is2645.pdf")
    assert first == second
    assert len(first) > 1
    assert all(item.standard_number == "IS 2645:2003" and item.page_start == 2 for item in first)


def test_validator_reports_invalid_provenance_and_relationship() -> None:
    metadata = StandardMetadata("IS 1:2020", None, 2020, None, None, None, "a.pdf", 1)
    bad_chunk = DocumentChunk("same", None, "", 2, 1, None, None, "")
    relationship = StandardRelationship("IS 1:2020", "", "wrong", 3, "")
    document = StandardDocument("1.0", metadata, (), (), (relationship,), (bad_chunk, bad_chunk))
    codes = {issue.code for issue in CorpusValidator().validate(document)}
    assert {"empty_document", "duplicate_chunk_id", "chunk_missing_provenance", "malformed_relationship"} <= codes


def test_pipeline_serializes_json_and_jsonl_deterministically(tmp_path) -> None:
    source = tmp_path / "IS2645.txt"
    source.write_text(
        "IS 2645:2003\nWaterproofing compound for cement concrete\n1 Scope\nThis document applies to concrete.\n2 References\nIS 3370 (Part 1):2021\n",
        encoding="utf-8",
    )
    output = tmp_path / "standard.json"
    chunks = tmp_path / "chunks.jsonl"
    document = CorpusBuilder().build_and_write(source, output, chunks)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["metadata"]["standard_number"] == "IS 2645:2003"
    assert payload["chunks"]
    rows = [json.loads(line) for line in chunks.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {key: getattr(chunk, key) for key in ("chunk_id", "standard_number", "source_file", "page_start", "page_end", "section_number", "section_title", "text")}
        for chunk in document.chunks
    ]


def test_pipeline_processes_a_page_preserving_pdf_end_to_end(tmp_path) -> None:
    source = tmp_path / "IS2645.pdf"
    pdf = pymupdf.open()
    for text in (
        "IS 2645:2003\nWaterproofing compound\n1 Scope\nThis applies to concrete.",
        "2 References\nIS 3370 (Part 1):2021",
    ):
        page = pdf.new_page()
        page.insert_text((72, 72), text)
    pdf.save(source)
    pdf.close()
    document = CorpusBuilder().build(source)
    assert len(document.pages) == 2
    assert document.metadata.standard_number == "IS 2645:2003"
    assert {chunk.page_start for chunk in document.chunks} == {1, 2}
    assert any(item.relationship_type == "references" for item in document.relationships)
