"""Rule-based, non-destructive detection of common standards headings."""

from __future__ import annotations

from dataclasses import dataclass
import re

from corpus.models import CorpusPage, Section, SectionFragment


_NUMBERED_HEADING_RE = re.compile(
    r"^(?P<number>\d+(?:\.\d+)*)\.?\s+(?P<title>[A-Za-z][^\n]{0,180})$"
)
_ANNEX_HEADING_RE = re.compile(
    r"^(?P<kind>Annex|Appendix)\s+(?P<letter>[A-Z])(?:\s*\([^)]*\))?\s*(?P<title>.*)$",
    re.I,
)


@dataclass(frozen=True)
class _Heading:
    page_number: int
    line_index: int
    number: str
    title: str
    level: int


class SectionDetector:
    """Detect clause and annex headings without treating uncertain lines as headings."""

    def detect(self, pages: tuple[CorpusPage, ...]) -> tuple[Section, ...]:
        """Partition cleaned text into sections; a preamble retains undetected content."""
        page_lines = [(page.page_number, page.cleaned_text.splitlines()) for page in pages]
        headings = self._find_headings(page_lines)
        if not headings:
            return self._preamble_only(page_lines)
        flattened = [
            (page_number, line_index, line)
            for page_number, lines in page_lines
            for line_index, line in enumerate(lines)
        ]
        heading_by_position = {(item.page_number, item.line_index): item for item in headings}
        starts = [
            index
            for index, (page_number, line_index, _) in enumerate(flattened)
            if (page_number, line_index) in heading_by_position
        ]
        sections: list[Section] = []
        # Text before the first confirmed heading is deliberately represented as a preamble.
        if starts and starts[0] > 0:
            sections.append(self._make_section(None, self._fragments(flattened[: starts[0]])))
        for start, end in zip(starts, [*starts[1:], len(flattened)], strict=True):
            page_number, line_index, _ = flattened[start]
            heading = heading_by_position[(page_number, line_index)]
            sections.append(self._make_section(heading, self._fragments(flattened[start:end])))
        return tuple(section for section in sections if section.text.strip())

    @staticmethod
    def _fragments(lines: list[tuple[int, int, str]]) -> list[SectionFragment]:
        grouped: dict[int, list[str]] = {}
        for page_number, _, line in lines:
            grouped.setdefault(page_number, []).append(line)
        return [SectionFragment(page_number, "\n".join(text).strip()) for page_number, text in grouped.items()]

    def _find_headings(self, page_lines: list[tuple[int, list[str]]]) -> list[_Heading]:
        headings: list[_Heading] = []
        for page_number, lines in page_lines:
            for line_index, line in enumerate(lines):
                candidate = line.strip()
                numbered = _NUMBERED_HEADING_RE.match(candidate)
                if numbered and self._is_plausible_numbered_heading(numbered.group("number"), numbered.group("title")):
                    number = numbered.group("number")
                    headings.append(
                        _Heading(page_number, line_index, number, numbered.group("title").strip(), number.count(".") + 1)
                    )
                    continue
                annex = _ANNEX_HEADING_RE.match(candidate)
                if annex:
                    number = f"{annex.group('kind').title()} {annex.group('letter').upper()}"
                    title = annex.group("title").strip() or number
                    headings.append(_Heading(page_number, line_index, number, title, 1))
        return headings

    @staticmethod
    def _is_plausible_numbered_heading(number: str, title: str) -> bool:
        """Reject common numeric prose/table rows while retaining standard clause headings."""
        if len(title) > 180 or not re.search(r"[A-Za-z]", title):
            return False
        if re.match(r"^(?:mm|cm|m|kg|%|MPa|°C)\b", title, re.I):
            return False
        return not (number.count(".") > 5)

    def _preamble_only(self, page_lines: list[tuple[int, list[str]]]) -> tuple[Section, ...]:
        fragments = tuple(
            SectionFragment(page_number=page_number, text="\n".join(lines).strip())
            for page_number, lines in page_lines
            if "\n".join(lines).strip()
        )
        return (self._make_section(None, list(fragments)),) if fragments else ()

    @staticmethod
    def _make_section(heading: _Heading | None, fragments: list[SectionFragment]) -> Section:
        useful = tuple(fragment for fragment in fragments if fragment.text.strip())
        pages = [fragment.page_number for fragment in useful]
        text = "\n".join(fragment.text for fragment in useful).strip()
        if not pages:
            return Section(None, "Document preamble", 0, 1, 1, "", ())
        return Section(
            section_number=heading.number if heading else None,
            title=heading.title if heading else "Document preamble",
            level=heading.level if heading else 0,
            page_start=min(pages),
            page_end=max(pages),
            text=text,
            fragments=useful,
        )
