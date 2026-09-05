from __future__ import annotations

import pymupdf
import pytest

from corpus import CorpusBuilder
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


def test_pdf_multimodal_extraction_preserves_geometry_assets_and_provenance(tmp_path) -> None:
    pdf = pymupdf.open()
    page = pdf.new_page(width=360, height=360)
    page.insert_text((30, 30), "IS 2645:2003\n1 Scope\nConcrete requirements.")

    # A ruled table is detected deterministically by PyMuPDF's table finder.
    for x in (30, 130, 230):
        page.draw_line((x, 90), (x, 150))
    for y in (90, 110, 130, 150):
        page.draw_line((30, y), (230, y))
    for point, text in [((35, 105), "Grade"), ((135, 105), "Value"), ((35, 125), "A"), ((135, 125), "10")]:
        page.insert_text(point, text)

    image = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.Rect(0, 0, 8, 8), False)
    image.clear_with(0x336699)
    page.insert_image(pymupdf.Rect(250, 90, 330, 150), stream=image.tobytes("png"))
    page.draw_rect(pymupdf.Rect(250, 190, 330, 250))
    data = pdf.tobytes()
    pdf.close()

    document = CorpusBuilder().build_bytes(
        data, "multimodal.pdf", "pdf", raw_asset_directory=tmp_path / "raw-assets"
    )
    corpus_page = document.pages[0]

    assert corpus_page.width == pytest.approx(360)
    assert corpus_page.height == pytest.approx(360)
    assert corpus_page.bbox is not None and corpus_page.bbox.x1 == pytest.approx(360)
    assert corpus_page.asset_path is not None and (tmp_path / "raw-assets" / "pages" / "page-0001.png").is_file()
    assert any("Concrete requirements" in block.content for block in corpus_page.text_blocks)
    assert corpus_page.tables and corpus_page.tables[0].headers == ("Grade", "Value")
    assert "Grade: A" in corpus_page.tables[0].content
    assert any(figure.type == "image" and figure.asset_path for figure in corpus_page.figures)
    assert any(figure.type == "vector_figure" for figure in corpus_page.figures)
    for element in (*corpus_page.text_blocks, *corpus_page.tables, *corpus_page.figures):
        assert element.standard_id == "IS 2645:2003"
        assert element.provenance.source_file == "multimodal.pdf"
        assert element.provenance.page_number == 1
        assert element.provenance.section_numbers == ("1",)
        assert element.provenance.chunk_ids == tuple(chunk.chunk_id for chunk in document.chunks)
        assert element.bbox.x1 >= element.bbox.x0
    assert corpus_page.tables[0].id in document.chunks[0].element_ids
    payload = document.to_dict()
    assert payload["raw_asset_directory"] == str(tmp_path / "raw-assets")
    assert payload["pages"][0]["tables"][0]["headers"] == ["Grade", "Value"]
