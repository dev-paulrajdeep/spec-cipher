"""
PDF reader for the SpecCipher corpus ingestion pipeline.

This module is responsible only for:
    - Opening PDF documents (from disk or in-memory bytes)
    - Reading page-level text blocks, tables, images, and vector-drawing signals
    - Preserving page boundaries, dimensions, and bounding boxes
    - Persisting raw visual assets when an asset directory is supplied
    - Returning OCR-routing signals without performing OCR itself

Text cleaning, metadata extraction, section detection, and chunking
belong to downstream corpus modules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
import hashlib

import pymupdf  # PyMuPDF

from corpus.models import BoundingBox, ElementProvenance, Figure, Table, TextBlock

logger = logging.getLogger(__name__)

# Below this average character count per page, a page is treated as
# text-sparse and likely to need OCR (e.g. a scanned image page).
OCR_CHAR_THRESHOLD = 20

# PyMuPDF's get_text() modes that return a plain string. Modes like
# "words", "blocks", "dict", and "rawdict" return lists/dicts instead
# and are not compatible with PDFPage.text: str.
_STRING_EXTRACTION_MODES = frozenset({"text", "html", "xml", "xhtml"})


@dataclass(frozen=True)
class PDFPage:
    """Raw text extracted from a single PDF page."""

    page_number: int
    text: str
    width: float | None = None
    height: float | None = None
    bbox: BoundingBox | None = None
    asset_path: str | None = None
    text_blocks: tuple[TextBlock, ...] = ()
    tables: tuple[Table, ...] = ()
    figures: tuple[Figure, ...] = ()
    # Internal: plain-text extraction used to judge text density, kept
    # separate from `text` so a non-default extraction_mode (e.g. "html")
    # can never mask a blank/scanned page behind markup boilerplate.
    # None means "extraction_mode was already 'text'", so `text` itself
    # is the reference; use field(compare=False) so it doesn't affect
    # equality between otherwise-identical pages.
    _sparsity_text: str | None = field(default=None, repr=False, compare=False)

    @property
    def char_count(self) -> int:
        """
        Number of characters used to judge text density.

        Always measured from plain-text extraction regardless of
        `extraction_mode`, so markup added by "html"/"xml"/"xhtml"
        modes never masks a page that is actually blank or scanned.
        """
        reference = self.text if self._sparsity_text is None else self._sparsity_text
        return len(reference.strip())

    @property
    def is_text_sparse(self) -> bool:
        """
        Whether this page yielded little to no extractable text.

        A sparse page is a strong signal that the page is a scanned
        image (or otherwise non-text) and should be routed to OCR by
        the ingestion pipeline. This flags the condition only; it does
        not attempt OCR itself, which is out of scope for this module.
        """
        return self.char_count < OCR_CHAR_THRESHOLD


@dataclass(frozen=True)
class PDFDocument:
    """Page-preserving representation of an extracted PDF."""

    source_file: str
    page_count: int
    pages: list[PDFPage] = field(default_factory=list)
    asset_directory: str | None = None

    @property
    def sparse_page_numbers(self) -> list[int]:
        """Page numbers (1-indexed) that likely need OCR."""
        return [p.page_number for p in self.pages if p.is_text_sparse]

    @property
    def likely_scanned(self) -> bool:
        """
        Whether the document as a whole looks scanned/image-based.

        True when the majority of pages are text-sparse. Downstream
        ingestion can use this to route the whole document to an OCR
        step rather than checking pages one by one.
        """
        if not self.pages:
            return False
        return len(self.sparse_page_numbers) > self.page_count / 2


class PDFReadError(RuntimeError):
    """Raised when a PDF cannot be opened or read."""


class EncryptedPDFError(PDFReadError):
    """Raised when a PDF is password-protected and cannot be opened."""


class PDFReader:
    """
    Read PDF documents and extract page-level raw text.

    The reader deliberately does not modify or clean the extracted text.
    Downstream processing is responsible for normalization and cleaning.
    """

    def __init__(self, extraction_mode: str = "text") -> None:
        """
        Args:
            extraction_mode: PyMuPDF text extraction mode passed to
                `page.get_text()`. Must be one of "text" (default),
                "html", "xml", or "xhtml" — the modes that return a
                plain string. Layout-structured modes ("words",
                "blocks", "dict", "rawdict") return lists/dicts and
                are out of scope for this module's str-based PDFPage.

        Raises:
            ValueError: If `extraction_mode` is not a string-returning mode.
        """
        if extraction_mode not in _STRING_EXTRACTION_MODES:
            raise ValueError(
                f"Unsupported extraction_mode '{extraction_mode}'. "
                f"Must be one of {sorted(_STRING_EXTRACTION_MODES)}."
            )
        self.extraction_mode = extraction_mode

    def read(self, file_path: str | Path, *, asset_directory: str | Path | None = None) -> PDFDocument:
        """
        Read a PDF from disk and return its page-preserving raw text.

        Args:
            file_path: Path to the PDF file.

        Returns:
            PDFDocument containing raw text for every page.

        Raises:
            FileNotFoundError: If the PDF does not exist.
            ValueError: If the supplied file is not a PDF.
            EncryptedPDFError: If the PDF is password-protected.
            PDFReadError: If PyMuPDF cannot open or read the document.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got: {path.suffix}")

        document = self._open(path, stream=None)
        return self._extract(document, source_name=path.name, asset_directory=asset_directory)

    def read_bytes(
        self, data: bytes, source_name: str = "<upload>", *, asset_directory: str | Path | None = None
    ) -> PDFDocument:
        """
        Read a PDF from an in-memory byte string (e.g. a Flask file
        upload) without requiring it to be written to disk first.

        Args:
            data: Raw PDF file bytes.
            source_name: Label to store as `PDFDocument.source_file`,
                e.g. the original uploaded filename.

        Returns:
            PDFDocument containing raw text for every page.

        Raises:
            ValueError: If `data` is empty.
            EncryptedPDFError: If the PDF is password-protected.
            PDFReadError: If PyMuPDF cannot open or read the document.
        """
        if not data:
            raise ValueError("No PDF data provided.")

        document = self._open(path=None, stream=data)
        return self._extract(document, source_name=source_name, asset_directory=asset_directory)

    def _open(
        self, path: Path | None, stream: bytes | None
    ) -> pymupdf.Document:
        """Open a PDF via PyMuPDF, raising specific errors for common failure modes."""
        try:
            document = (
                pymupdf.open(stream=stream, filetype="pdf")
                if stream is not None
                else pymupdf.open(path)
            )
        except Exception as exc:
            label = path if path is not None else "<in-memory PDF>"
            raise PDFReadError(f"Failed to open PDF '{label}': {exc}") from exc

        if not document.is_pdf:
            document.close()
            label = path if path is not None else "<in-memory PDF>"
            raise ValueError(f"File is not a valid PDF: {label}")

        if document.needs_pass:
            document.close()
            label = path if path is not None else "<in-memory PDF>"
            raise EncryptedPDFError(
                f"PDF '{label}' is password-protected and cannot be read."
            )

        return document

    def _extract(
        self, document: pymupdf.Document, source_name: str, asset_directory: str | Path | None
    ) -> PDFDocument:
        """Extract deterministic text, layout, tables, and visual assets from a PDF."""
        asset_root = Path(asset_directory) if asset_directory is not None else None
        try:
            pages = [
                self._extract_page(page, page_number, source_name, asset_root)
                for page_number, page in enumerate(document, start=1)
            ]
            page_count = len(document)
        except Exception as exc:
            raise PDFReadError(
                f"Failed to extract text from PDF '{source_name}': {exc}"
            ) from exc
        finally:
            document.close()

        pdf_document = PDFDocument(
            source_file=source_name,
            page_count=page_count,
            pages=pages,
            asset_directory=str(asset_root) if asset_root is not None else None,
        )

        if pdf_document.likely_scanned:
            logger.warning(
                "PDF '%s' looks scanned (%d/%d pages text-sparse); "
                "consider routing to OCR.",
                source_name,
                len(pdf_document.sparse_page_numbers),
                page_count,
            )

        return pdf_document

    def _extract_page(
        self, page: pymupdf.Page, page_number: int, source_name: str, asset_root: Path | None
    ) -> PDFPage:
        rect = page.rect
        page_bbox = self._bbox(rect)
        provenance = ElementProvenance(source_name, None, page_number)
        page_asset = self._render_page(page, page_number, asset_root)
        text_blocks = tuple(
            TextBlock(
                id=self._element_id(source_name, page_number, "text", index),
                standard_id=None,
                page_number=page_number,
                bbox=BoundingBox(float(x0), float(y0), float(x1), float(y1)),
                type="text",
                content=text.strip(),
                provenance=provenance,
            )
            for index, (x0, y0, x1, y1, text, _, block_type) in enumerate(
                page.get_text("blocks", sort=True), start=1
            )
            if block_type == 0 and text.strip()
        )
        tables = self._extract_tables(page, page_number, source_name, provenance)
        figures = [*self._extract_images(page, page_number, source_name, provenance, asset_root)]
        vector = self._extract_vector_figure(
            page, page_number, source_name, provenance, page_asset, tables
        )
        if vector is not None:
            figures.append(vector)
        return PDFPage(
            page_number=page_number,
            text=page.get_text(self.extraction_mode),
            width=float(rect.width),
            height=float(rect.height),
            bbox=page_bbox,
            asset_path=page_asset,
            text_blocks=text_blocks,
            tables=tables,
            figures=tuple(figures),
            _sparsity_text=page.get_text("text") if self.extraction_mode != "text" else None,
        )

    @staticmethod
    def _bbox(rect: pymupdf.Rect) -> BoundingBox:
        return BoundingBox(float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))

    @staticmethod
    def _element_id(source_name: str, page_number: int, kind: str, ordinal: int) -> str:
        identity = f"{source_name}|{page_number}|{kind}|{ordinal}"
        return f"{kind}_{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:20]}"

    @staticmethod
    def _asset_path(asset_root: Path, *parts: str) -> Path:
        path = asset_root.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _render_page(self, page: pymupdf.Page, page_number: int, asset_root: Path | None) -> str | None:
        if asset_root is None:
            return None
        target = self._asset_path(asset_root, "pages", f"page-{page_number:04d}.png")
        page.get_pixmap(dpi=144, alpha=False).save(target)
        return str(target)

    def _extract_tables(
        self, page: pymupdf.Page, page_number: int, source_name: str, provenance: ElementProvenance
    ) -> tuple[Table, ...]:
        try:
            found = page.find_tables()
        except Exception as exc:
            logger.warning("Table detection failed on %s page %d: %s", source_name, page_number, exc)
            return ()
        tables: list[Table] = []
        for index, table in enumerate(found.tables, start=1):
            extracted_rows = tuple(
                tuple("" if cell is None else str(cell).strip() for cell in row)
                for row in table.extract()
            )
            header = tuple("" if name is None else str(name).strip() for name in table.header.names)
            if not any(header) and extracted_rows:
                header, extracted_rows = extracted_rows[0], extracted_rows[1:]
            elif not table.header.external and extracted_rows and extracted_rows[0] == header:
                extracted_rows = extracted_rows[1:]
            tables.append(
                Table(
                    id=self._element_id(source_name, page_number, "table", index),
                    standard_id=None,
                    page_number=page_number,
                    bbox=BoundingBox(*map(float, table.bbox)),
                    type="table",
                    headers=header,
                    rows=extracted_rows,
                    content=self._table_text(header, extracted_rows),
                    provenance=provenance,
                )
            )
        return tuple(tables)

    @staticmethod
    def _table_text(headers: tuple[str, ...], rows: tuple[tuple[str, ...], ...]) -> str:
        labels = headers or tuple(
            f"Column {index + 1}" for index in range(max((len(row) for row in rows), default=0))
        )
        lines = ["Table: " + " | ".join(labels)] if labels else ["Table"]
        for row in rows:
            values = [
                f"{labels[index] if index < len(labels) else f'Column {index + 1}'}: {value}"
                for index, value in enumerate(row)
            ]
            lines.append("; ".join(values))
        return "\n".join(lines)

    def _extract_images(
        self, page: pymupdf.Page, page_number: int, source_name: str, provenance: ElementProvenance,
        asset_root: Path | None,
    ) -> tuple[Figure, ...]:
        figures: list[Figure] = []
        seen_xrefs: set[int] = set()
        for image in page.get_images(full=True):
            xref = image[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)
            try:
                image_data = page.parent.extract_image(xref)
                extension = image_data.get("ext", "bin")
                asset_path = None
                if asset_root is not None:
                    target = self._asset_path(asset_root, "images", f"image-{xref}.{extension}")
                    target.write_bytes(image_data["image"])
                    asset_path = str(target)
                for occurrence, rect in enumerate(page.get_image_rects(xref), start=1):
                    figures.append(
                        Figure(
                            id=self._element_id(source_name, page_number, "image", len(figures) + occurrence),
                            standard_id=None,
                            page_number=page_number,
                            bbox=self._bbox(rect),
                            type="image",
                            asset_path=asset_path or "",
                            provenance=provenance,
                            content=f"Embedded image xref {xref}",
                        )
                    )
            except Exception as exc:
                logger.warning(
                    "Image extraction failed on %s page %d xref %d: %s",
                    source_name,
                    page_number,
                    xref,
                    exc,
                )
        return tuple(figures)

    def _extract_vector_figure(
        self, page: pymupdf.Page, page_number: int, source_name: str, provenance: ElementProvenance,
        page_asset: str | None, tables: tuple[Table, ...],
    ) -> Figure | None:
        try:
            drawings = [
                drawing["rect"] for drawing in page.get_drawings() if drawing["rect"].get_area() > 0
            ]
        except Exception as exc:
            logger.warning("Vector drawing detection failed on %s page %d: %s", source_name, page_number, exc)
            return None
        if not drawings or page_asset is None:
            return None
        visual_rects = [
            rect for rect in drawings if not any(self._contained_in(rect, table.bbox) for table in tables)
        ]
        if not visual_rects:
            return None
        union = pymupdf.Rect(visual_rects[0])
        for rect in visual_rects[1:]:
            union.include_rect(rect)
        return Figure(
            id=self._element_id(source_name, page_number, "vector_figure", 1),
            standard_id=None,
            page_number=page_number,
            bbox=self._bbox(union),
            type="vector_figure",
            asset_path=page_asset,
            provenance=provenance,
            content=f"{len(visual_rects)} vector drawing(s); use the full-page render and bounding box.",
        )

    @staticmethod
    def _contained_in(rect: pymupdf.Rect, bbox: BoundingBox) -> bool:
        return rect.x0 >= bbox.x0 and rect.y0 >= bbox.y0 and rect.x1 <= bbox.x1 and rect.y1 <= bbox.y1
