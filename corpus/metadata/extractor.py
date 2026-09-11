"""Conservative metadata extraction from the document's leading pages."""

from __future__ import annotations

import re

from corpus.models import CorpusPage, StandardMetadata
from corpus.processing.normalizer import find_document_numbers, normalize_document_number
from .parser import StandardNumberParser


_EDITION_RE = re.compile(r"\b((?:first|second|third|fourth|\d+(?:st|nd|rd|th))\s+edition)\b", re.I)
_REVISION_RE = re.compile(r"\b((?:first|second|third|fourth|\d+(?:st|nd|rd|th))\s+revision|revision\s*\d+)\b", re.I)
_STATUS_RE = re.compile(r"\b(reaffirmed|withdrawn|current|superseded|cancelled)\b", re.I)


class MetadataExtractor:
    """Extract metadata when explicitly present; leave the rest unknown."""

    def __init__(self, parser: StandardNumberParser | None = None) -> None:
        self._parser = parser or StandardNumberParser()

    def extract(self, pages: tuple[CorpusPage, ...], source_file: str) -> StandardMetadata:
        leading_text = "\n".join(page.cleaned_text for page in pages[:3])
        # Never invent identity from multi-document bodies or ambiguous filenames.
        # The corpus update identity layer is authoritative for recommendation use.
        content_numbers = find_document_numbers(leading_text)
        base_numbers = tuple(dict.fromkeys(_base(number) for number in content_numbers))
        filename_number = self._parser.parse(source_file)
        if len(base_numbers) == 1:
            chosen_base = base_numbers[0]
            standard_number = next(
                number for number in content_numbers if _base(number) == chosen_base
            )
        elif filename_number and not content_numbers:
            standard_number = filename_number
        elif filename_number and any(
            _base(number) == _base(filename_number) for number in content_numbers
        ):
            filename_base = _base(filename_number)
            standard_number = next(
                number for number in content_numbers if _base(number) == filename_base
            )
        else:
            standard_number = None
        title = self._title_from_leading_pages(pages, standard_number)
        edition = self._first_group(_EDITION_RE, leading_text)
        revisions = _REVISION_RE.findall(leading_text)
        revision = revisions[-1] if revisions else None
        status = self._first_group(_STATUS_RE, leading_text)
        year = self._parser.year_from(leading_text) or self._parser.year_from(source_file)
        return StandardMetadata(
            standard_number=standard_number,
            title=title,
            year=year,
            edition=edition.title() if edition else None,
            revision=revision.title() if revision else None,
            status=status.lower() if status else None,
            source_file=source_file,
            page_count=len(pages),
        )

    @staticmethod
    def _first_group(pattern: re.Pattern[str], text: str) -> str | None:
        match = pattern.search(text)
        return match.group(1) if match else None

    @staticmethod
    def _title_from_leading_pages(pages: tuple[CorpusPage, ...], standard_number: str | None) -> str | None:
        """Select an obvious title line near the identifier, otherwise retain uncertainty."""
        for page in pages[:3]:
            lines = [line.strip() for line in page.cleaned_text.splitlines() if line.strip()]
            for index, line in enumerate(lines):
                if standard_number and normalize_document_number(line) == standard_number:
                    for following in lines[index + 1 : index + 4]:
                        if (
                            len(following) >= 4
                            and len(following) <= 220
                            and not following.lower().startswith(("copyright", "bureau of"))
                            and not re.match(
                                r"^(?:\d+(?:\.\d+)*\.?\s+|annex\b|appendix\b|first\b|second\b|third\b|fourth\b)",
                                following,
                                re.I,
                            )
                        ):
                            return following
        return None


def _base(value: str) -> str:
    return re.sub(r"[:/]\d{2,4}$", "", value).strip()
