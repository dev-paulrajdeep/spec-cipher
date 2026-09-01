from __future__ import annotations

from corpus.metadata import MetadataExtractor, StandardNumberParser
from corpus.models import CorpusPage, Section, SectionFragment
from corpus.relationships import RelationshipExtractor


def test_indian_standard_metadata_parser_and_extractor() -> None:
    pages = (
        CorpusPage(1, "", "IS 3370 (Part 1): 2021\nConcrete structures for storage of liquids\nSecond Revision\nReaffirmed"),
    )
    assert StandardNumberParser().parse("IS 2645:2003") == "IS 2645:2003"
    metadata = MetadataExtractor().extract(pages, "is3370.pdf")
    assert metadata.standard_number == "IS 3370 (Part 1):2021"
    assert metadata.year == 2021
    assert metadata.title == "Concrete structures for storage of liquids"
    assert metadata.revision == "Second Revision"
    assert metadata.status == "reaffirmed"


def test_relationships_have_explicit_type_and_evidence() -> None:
    pages = (CorpusPage(1, "", "2 References\nIS 2645:2003\nThis standard supersedes IS 9999:1990."),)
    sections = (
        Section("2", "References", 1, 1, 1, pages[0].cleaned_text, (SectionFragment(1, pages[0].cleaned_text),)),
    )
    relationships = RelationshipExtractor().extract(pages, "IS 3370 (Part 1):2021", sections)
    assert {item.relationship_type for item in relationships} == {"references", "supersedes"}
    assert all(item.evidence_text for item in relationships)
