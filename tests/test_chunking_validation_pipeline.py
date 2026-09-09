from __future__ import annotations

import json
from pathlib import Path

import pymupdf
import pytest

from corpus import CorpusBuilder
from corpus.chunking import ChunkingConfig, DocumentChunker
from corpus.models import DocumentChunk, Section, SectionFragment, StandardDocument, StandardMetadata, StandardRelationship
from corpus.processing.normalizer import find_standard_references, normalize_standard_number
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
        {
            **{
                key: getattr(chunk, key)
                for key in ("chunk_id", "standard_number", "source_file", "page_start", "page_end", "section_number", "section_title", "text")
            },
            "referenced_standards": list(chunk.referenced_standards),
        }
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


def test_standard_reference_normalization_preserves_year_and_part_details() -> None:
    assert normalize_standard_number("IS 1641 : 2013") == "IS 1641:2013"
    assert normalize_standard_number("IS 875 (Part 3):2015") == "IS 875 (Part 3):2015"
    assert normalize_standard_number("IS 302 (Part 2/Sec 201)") == "IS 302 (Part 2/Sec 201)"
    assert find_standard_references("IS 1641 : 2013 and IS 875 (Part 3):2015") == (
        "IS 1641:2013",
        "IS 875 (Part 3):2015",
    )


def test_chunk_keeps_standard_title_context_and_exact_page_fragment() -> None:
    sections = (
        Section(
            None,
            "Document preamble",
            0,
            1,
            2,
            "IS 12444:2020\nCopper wire rods for electrical applications\nSpecification",
            (
                SectionFragment(1, "IS 12444:2020\nCopper wire rods for electrical applications\nSpecification"),
                SectionFragment(2, "A separate second-page context."),
            ),
        ),
    )
    chunks = DocumentChunker(ChunkingConfig(max_chars=200, overlap_chars=0)).chunk(sections, None, "truth.pdf")
    first = chunks[0]
    assert first.page_start == first.page_end == 1
    assert "IS 12444:2020" in first.text
    assert "Copper wire rods for electrical applications" in first.text
    assert first.referenced_standards == ("IS 12444:2020",)


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / "truth").is_dir(),
    reason="external truth PDF directory is not available",
)
def test_truth_pdfs_ingest_with_page_and_source_provenance() -> None:
    truth = Path(__file__).resolve().parents[2] / "truth"
    sources = sorted(truth.glob("*.pdf"))
    assert [source.name for source in sources] == [
        "IS_Codes_for_Industrial_Pumps_Engineer_Checklist.pdf",
        "List-of-Products-Under-Simplified-Procedure.pdf",
        "temp_2131689472.pdf",
        "temp_2131689475.pdf",
    ]
    for source in sources:
        document = CorpusBuilder().build(source)
        assert document.pages
        assert document.chunks
        assert all(chunk.source_file == source.name for chunk in document.chunks)
        assert all(1 <= chunk.page_start <= chunk.page_end <= len(document.pages) for chunk in document.chunks)
