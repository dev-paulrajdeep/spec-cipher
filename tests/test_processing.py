from __future__ import annotations

from corpus.ingestion import ExtractedDocument, ExtractedPage
from corpus.models import CorpusPage
from corpus.processing import SectionDetector, TextCleaner


def test_cleaner_normalizes_hyphenation_headers_and_page_numbers() -> None:
    document = ExtractedDocument(
        source_file="sample.pdf",
        source_format="pdf",
        page_count=3,
        pages=[
            ExtractedPage(1, "IS HEADER\nwater-\nproofing compound\n1\n"),
            ExtractedPage(2, "IS HEADER\nfor concrete.\n2\n"),
            ExtractedPage(3, "IS HEADER\nFinal requirement.\n3\n"),
        ],
    )
    pages = TextCleaner().clean_document(document)
    assert pages[0].raw_text.startswith("IS HEADER")
    assert "waterproofing compound" in pages[0].cleaned_text
    assert "IS HEADER" not in pages[0].cleaned_text
    assert pages[1].cleaned_text == "for concrete."


def test_section_detector_preserves_preamble_and_nested_sections() -> None:
    pages = (
        CorpusPage(1, "", "Cover text\n1 Scope\nScope body\n4 Requirements\nRequirement body"),
        CorpusPage(2, "", "4.1 Material\nMaterial body\n4.1.1 Grade\nGrade body\nAnnex A (normative) Test method"),
    )
    sections = SectionDetector().detect(pages)
    assert [item.section_number for item in sections] == [None, "1", "4", "4.1", "4.1.1", "Annex A"]
    assert sections[0].text == "Cover text"
    assert sections[-1].page_start == 2
    assert "Grade body" in sections[-2].text
