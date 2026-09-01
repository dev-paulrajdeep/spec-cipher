from __future__ import annotations

import pymupdf
import pytest

from corpus.ingestion import DocumentExtractor, OCRRouter, PDFReader


def make_pdf_bytes(pages: list[str]) -> bytes:
    document = pymupdf.open()
    for text in pages:
        page = document.new_page()
        if text:
            page.insert_text((72, 72), text)
    data = document.tobytes()
    document.close()
    return data


def test_pdf_reader_preserves_in_memory_pages() -> None:
    result = PDFReader().read_bytes(make_pdf_bytes(["first page", "second page"]), "sample.pdf")
    assert result.source_file == "sample.pdf"
    assert result.page_count == 2
    assert [page.page_number for page in result.pages] == [1, 2]
    assert "second page" in result.pages[1].text


def test_invalid_and_missing_inputs_raise(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        PDFReader().read(tmp_path / "absent.pdf")
    not_pdf = tmp_path / "not-a-pdf.txt"
    not_pdf.write_text("plain text", encoding="utf-8")
    with pytest.raises(ValueError):
        PDFReader().read(not_pdf)
    with pytest.raises(ValueError):
        DocumentExtractor().extract_bytes(b"", "empty.pdf", "pdf")


class StubOCR:
    def extract_page(self, document, page_number: int) -> str:
        return f"OCR text for page {page_number}"


def test_sparse_page_routing_only_targets_sparse_pages() -> None:
    document = DocumentExtractor().extract_bytes(make_pdf_bytes(["enough source text to exceed the sparse threshold", ""]), "mixed.pdf", "pdf")
    routed = OCRRouter().route(document, StubOCR())
    assert routed.ocr_page_numbers == (2,)
    assert routed.document.pages[0].text == document.pages[0].text
    assert routed.document.pages[1].text == "OCR text for page 2"


def test_sparse_page_without_provider_is_visible() -> None:
    document = DocumentExtractor().extract_bytes(make_pdf_bytes([""]), "scan.pdf", "pdf")
    routed = OCRRouter().route(document)
    assert routed.ocr_page_numbers == ()
    assert routed.warnings
