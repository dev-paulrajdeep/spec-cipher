"""Unit tests for SpecCipher pure domain kernel entities (Layer 7)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from speccipher.domain.candidates import Candidate, RequirementFeatures
from speccipher.domain.identity import StandardId, StandardStub
from speccipher.domain.models import DocumentChunk, LogicalDocument, Section, SourceFile
from speccipher.domain.recommendation import Confidence, Recommendation
from speccipher.domain.relationships import RelationshipRole, StandardRelationship


class TestStandardId:
    def test_canonical_parsing_and_formatting(self) -> None:
        sid1 = StandardId.parse("IS 456:2000")
        assert sid1.family == "IS"
        assert sid1.number == "456"
        assert sid1.part is None
        assert sid1.year == 2000
        assert sid1.code == "IS 456:2000"

        sid2 = StandardId.parse("IS 875 (Part 3):2015")
        assert sid2.family == "IS"
        assert sid2.number == "875"
        assert sid2.part == "3"
        assert sid2.year == 2015
        assert sid2.code == "IS 875 (Part 3):2015"

        sid3 = StandardId.parse("SP 7:2026")
        assert sid3.family == "SP"
        assert sid3.number == "7"
        assert sid3.year == 2026
        assert sid3.code == "SP 7:2026"

        sid4 = StandardId.parse("ASTM C150:2020")
        assert sid4.family == "ASTM"
        assert sid4.number == "C150"
        assert sid4.year == 2020
        assert sid4.code == "ASTM C150:2020"

        sid5 = StandardId.parse("BOOK-18-ELECTRONICS-AND")
        assert sid5.family == "GENERIC"
        assert sid5.code == "BOOK-18-ELECTRONICS-AND"

    def test_immutability(self) -> None:
        sid = StandardId(family="IS", number="456", year=2000)
        with pytest.raises(FrozenInstanceError):
            sid.number = "875"  # type: ignore

    def test_serialization_round_trip(self) -> None:
        sid = StandardId.parse("IS 456:2000")
        d = sid.to_dict()
        assert d == {
            "family": "IS",
            "number": "456",
            "part": None,
            "year": 2000,
            "code": "IS 456:2000",
        }
        restored = StandardId.from_dict(d)
        assert restored == sid

    def test_invalid_parsing_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            StandardId.parse("")
        with pytest.raises(ValueError):
            StandardId.parse("invalid standard string !@#$")


class TestStandardStub:
    def test_stub_invariants(self) -> None:
        sid = StandardId.parse("IS 1234:2020")
        stub = StandardStub(standard_id=sid, title="Referenced Standard", is_eligible=True)
        assert stub.is_stub is True
        assert stub.is_eligible is False
        assert stub.standard_id == sid

        d = stub.to_dict()
        assert d["is_stub"] is True
        assert d["is_eligible"] is False
        restored = StandardStub.from_dict(d)
        assert restored.is_stub is True
        assert restored.is_eligible is False


class TestModels:
    def test_source_file_validation(self) -> None:
        sf = SourceFile(source_file_id="src-1", filename="IS456.pdf", page_count=100)
        assert sf.page_count == 100
        d = sf.to_dict()
        restored = SourceFile.from_dict(d)
        assert restored == sf

        with pytest.raises(ValueError):
            SourceFile(source_file_id="", filename="test.pdf", page_count=5)

    def test_logical_document(self) -> None:
        sid = StandardId.parse("IS 456:2000")
        doc = LogicalDocument(
            logical_doc_id="is-456",
            standard_id=sid,
            title="Plain and Reinforced Concrete",
            edition="Fourth",
            is_multipart=False,
            total_pages=114,
        )
        assert doc.recommendation_eligible is True
        assert doc.title == "Plain and Reinforced Concrete"
        d = doc.to_dict()
        restored = LogicalDocument.from_dict(d)
        assert restored == doc

    def test_section(self) -> None:
        sec = Section(
            section_number="5.1",
            title="Materials",
            level=2,
            page_start=13,
            page_end=15,
            text="Cement content",
        )
        assert sec.level == 2
        d = sec.to_dict()
        assert Section.from_dict(d) == sec

        with pytest.raises(ValueError):
            Section(page_start=10, page_end=5)

    def test_document_chunk_backward_compatible_properties(self) -> None:
        sid = StandardId.parse("IS 456:2000")
        chunk = DocumentChunk(
            chunk_id="chk-001",
            standard_id=sid,
            logical_doc_id="doc-1",
            source_file_id="IS456.pdf",
            absolute_page_start=14,
            absolute_page_end=15,
            section_number="5.1.1",
            section_title="Cement",
            text="The cement used shall comply with...",
            is_eligible=True,
        )
        # Legacy property access
        assert chunk.standard_number == "IS 456:2000"
        assert chunk.source_file == "IS456.pdf"
        assert chunk.page_start == 14
        assert chunk.page_end == 15
        assert chunk.recommendation_eligible is True

        d = chunk.to_dict()
        assert d["chunk_id"] == "chk-001"
        assert d["absolute_page_start"] == 14
        restored = DocumentChunk.from_dict(d)
        assert restored.chunk_id == chunk.chunk_id
        assert restored.standard_id == sid


class TestRelationships:
    def test_relationship_role_taxonomy(self) -> None:
        assert RelationshipRole.from_string("primary standard") == RelationshipRole.PRIMARY
        assert RelationshipRole.from_string("testing method") == RelationshipRole.TEST_STANDARD
        assert RelationshipRole.from_string("safety clause") == RelationshipRole.SAFETY_STANDARD
        assert (
            RelationshipRole.from_string("normative reference")
            == RelationshipRole.NORMATIVE_REFERENCE
        )
        assert RelationshipRole.from_string("unknown type") == RelationshipRole.ALLIED_STANDARD

    def test_standard_relationship_properties_and_round_trip(self) -> None:
        src_id = StandardId.parse("IS 456:2000")
        tgt_id = StandardId.parse("IS 2645:2003")
        rel = StandardRelationship(
            source_standard_id=src_id,
            target_standard_id=tgt_id,
            role=RelationshipRole.NORMATIVE_REFERENCE,
            page_number=14,
            evidence_text="Integral cement waterproofing compounds complying with IS 2645...",
        )
        assert rel.standard_number == "IS 2645:2003"
        assert rel.target_standard_number == "IS 2645:2003"
        assert rel.source_standard_number == "IS 456:2000"
        assert rel.relationship_type == "Normative Reference"

        d = rel.to_dict()
        restored = StandardRelationship.from_dict(d)
        assert restored.target_standard_id == tgt_id
        assert restored.source_standard_id == src_id
        assert restored.role == RelationshipRole.NORMATIVE_REFERENCE


class TestCandidates:
    def test_candidate_scoring(self) -> None:
        sid = StandardId.parse("IS 456:2000")
        chunk = DocumentChunk(
            chunk_id="c1",
            standard_id=sid,
            logical_doc_id="doc1",
            source_file_id="is456.pdf",
            absolute_page_start=1,
            absolute_page_end=1,
            text="Concrete chunk",
        )
        cand = Candidate(chunk=chunk, dense_score=0.85, rrf_score=0.016, rerank_score=0.92)
        assert cand.similarity_score == 0.85
        assert cand.score == 0.92

        cand2 = Candidate(chunk=chunk, dense_score=0.85, rrf_score=0.016, rerank_score=None)
        assert cand2.score == 0.85

        d = cand.to_dict()
        assert Candidate.from_dict(d).rerank_score == 0.92

    def test_requirement_features(self) -> None:
        rf = RequirementFeatures(
            requirement_tokens=frozenset(["concrete", "waterproofing"]),
            product_terms=frozenset(["admixture"]),
        )
        d = rf.to_dict()
        assert d["product_terms"] == ["admixture"]
        restored = RequirementFeatures.from_dict(d)
        assert restored.product_terms == frozenset(["admixture"])


class TestRecommendation:
    def test_confidence_validation(self) -> None:
        c = Confidence(level="High", score=0.95, reasons=("direct_scope_match", "recent_revision"))
        assert c.level == "High"
        assert c.score == 0.95

        with pytest.raises(ValueError):
            Confidence(level="InvalidLevel", score=0.5)
        with pytest.raises(ValueError):
            Confidence(level="High", score=1.5)

    def test_recommendation_dual_contract_serialization(self) -> None:
        sid = StandardId.parse("IS 2645:2003")
        chunk = DocumentChunk(
            chunk_id="chk-1",
            standard_id=sid,
            logical_doc_id="doc-1",
            source_file_id="IS2645.pdf",
            absolute_page_start=4,
            absolute_page_end=5,
            section_number="2.1",
            section_title="Scope",
            text="Integral waterproofing compound requirements.",
        )
        conf = Confidence(level="High", score=0.92, reasons=("scope_match",))
        rec = Recommendation(
            standard_id=sid,
            title="Waterproofing compound",
            rank=1,
            confidence=conf,
            similarity_score=0.82,
            applicability_score=0.90,
            evidence=(chunk,),
            status="current",
            version="2003",
        )

        assert rec.standard_number == "IS 2645:2003"
        assert rec.is_code == "IS 2645:2003"
        assert rec.score == 0.82

        d = rec.to_dict()
        # Modern domain keys
        assert d["standard_id"]["code"] == "IS 2645:2003"
        assert d["confidence"]["level"] == "High"
        assert d["similarity_score"] == 0.82
        assert d["applicability_score"] == 0.90

        # Legacy frontend / test compatibility keys
        assert d["is_code"] == "IS 2645:2003"
        assert d["standard_number"] == "IS 2645:2003"
        assert d["title"] == "Waterproofing compound"
        assert d["score"] == 0.82
        assert d["rank"] == 1
        assert d["source"]["source_file"] == "IS2645.pdf"
        assert d["source"]["page_start"] == 4
        assert d["source_evidence"]["document_title"] == "Waterproofing compound"
        assert d["category_breadcrumb"] == "2.1 → Scope"

        # Round-trip deserialization
        restored = Recommendation.from_dict(d)
        assert restored.standard_id == sid
        assert restored.title == "Waterproofing compound"
        assert restored.confidence.level == "High"
        assert len(restored.evidence) == 1
        assert restored.evidence[0].chunk_id == "chk-1"
