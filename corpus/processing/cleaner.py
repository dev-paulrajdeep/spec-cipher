"""Conservative page-aware cleaning that always retains raw extracted text."""

from __future__ import annotations

from collections import Counter
import re

from corpus.ingestion.extractor import ExtractedDocument
from corpus.models import CorpusPage
from .normalizer import normalize_text


_PAGE_NUMBER_RE = re.compile(r"^\s*(?:page\s*)?\d{1,4}\s*(?:of\s*\d{1,4})?\s*$", re.I)
_HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s+\D|(?:Annex\s+[A-Z]|Appendix)\b)", re.I
)
_STANDARD_LINE_RE = re.compile(r"^\s*IS\s*\d{1,5}\b", re.I)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


class TextCleaner:
    """Clean extraction artifacts while preserving page identity and all raw content."""

    def clean_document(self, document: ExtractedDocument) -> tuple[CorpusPage, ...]:
        """Return cleaned pages; repeated margins are identified corpus-wide first."""
        raw_lines = [normalize_text(page.text).splitlines() for page in document.pages]
        repeated_margins = self._repeated_margins(raw_lines)
        pages: list[CorpusPage] = []
        for source_page, lines in zip(document.pages, raw_lines, strict=True):
            cleaned = self.clean_page_lines(lines, repeated_margins)
            pages.append(
                CorpusPage(
                    page_number=source_page.page_number,
                    raw_text=source_page.text,
                    cleaned_text=cleaned,
                    bbox=source_page.bbox,
                    width=source_page.width,
                    height=source_page.height,
                    asset_path=source_page.asset_path,
                    text_blocks=source_page.text_blocks,
                    tables=source_page.tables,
                    figures=source_page.figures,
                )
            )
        return tuple(pages)

    def clean_page_lines(self, lines: list[str], repeated_margins: set[str] | None = None) -> str:
        """Clean a page's lines with no cross-page joining or content deletion by default."""
        margins = repeated_margins or set()
        normalized = [_CONTROL_RE.sub("", line).strip() for line in lines]
        retained = [
            line
            for line in normalized
            if line and line not in margins and not _PAGE_NUMBER_RE.match(line)
        ]
        retained = self._repair_hyphenation(retained)
        return self._join_wrapped_lines(retained)

    def _repeated_margins(self, pages: list[list[str]]) -> set[str]:
        """Remove only identical, short first/last lines repeated on at least three pages."""
        candidates: list[str] = []
        for lines in pages:
            non_empty = [line.strip() for line in lines if line.strip()]
            if non_empty:
                candidates.extend((non_empty[0], non_empty[-1]))
        threshold = max(3, (len(pages) + 1) // 2)
        return {
            line for line, count in Counter(candidates).items()
            if count >= threshold and len(line) <= 120 and not _HEADING_RE.match(line)
        }

    @staticmethod
    def _repair_hyphenation(lines: list[str]) -> list[str]:
        repaired: list[str] = []
        index = 0
        while index < len(lines):
            line = lines[index]
            if (
                index + 1 < len(lines)
                and re.search(r"[A-Za-z]{2,}-$", line)
                and re.match(r"^[a-z][A-Za-z]*\b", lines[index + 1])
            ):
                repaired.append(line[:-1] + lines[index + 1])
                index += 2
            else:
                repaired.append(line)
                index += 1
        return repaired

    @staticmethod
    def _join_wrapped_lines(lines: list[str]) -> str:
        """Join clear prose wraps while retaining detected headings and paragraph boundaries."""
        output: list[str] = []
        for line in lines:
            if not output:
                output.append(line)
                continue
            previous = output[-1]
            join = (
                previous
                and not _HEADING_RE.match(previous)
                and not _HEADING_RE.match(line)
                and not _STANDARD_LINE_RE.match(previous)
                and not re.search(r"[.!?;:]$", previous)
                and not re.match(r"^(?:[-•*]|\d+[.)])\s+", line)
            )
            if join:
                output[-1] = f"{previous} {line}"
            else:
                output.append(line)
        return "\n".join(output)
