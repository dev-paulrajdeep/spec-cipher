"""Deterministic sample datasets, queries, and standard specifications for E2E tests."""

SAMPLE_STANDARDS = [
    {
        "id": "SP_7_2026",
        "family": "SP",
        "number": "7",
        "part": None,
        "year": 2026,
        "code": "SP 7:2026",
        "title": "National Building Code of India 2026",
        "status": "current",
        "edition": "Fourth Revision",
        "identity_source": "extracted",
        "is_eligible": 1,
        "is_stub": 0,
        "page_count": 2999,
        "volumes": [
            {"file_name": "SP 7_2026-1-400.pdf", "page_count": 400, "page_offset": 0, "part_index": 1},
            {"file_name": "SP 7_2026-401-800.pdf", "page_count": 400, "page_offset": 400, "part_index": 2},
            {"file_name": "SP 7_2026-801-1200.pdf", "page_count": 400, "page_offset": 800, "part_index": 3},
            {"file_name": "SP 7_2026-1201-1600.pdf", "page_count": 400, "page_offset": 1200, "part_index": 4},
            {"file_name": "SP 7_2026-1601-2000.pdf", "page_count": 400, "page_offset": 1600, "part_index": 5},
            {"file_name": "SP 7_2026-2001-2400.pdf", "page_count": 400, "page_offset": 2000, "part_index": 6},
            {"file_name": "SP 7_2026-2401-2800.pdf", "page_count": 400, "page_offset": 2400, "part_index": 7},
            {"file_name": "SP 7_2026-2801-2999.pdf", "page_count": 199, "page_offset": 2800, "part_index": 8},
        ],
    },
    {
        "id": "IS_456_2000",
        "family": "IS",
        "number": "456",
        "part": None,
        "year": 2000,
        "code": "IS 456:2000",
        "title": "Plain and Reinforced Concrete - Code of Practice",
        "status": "current",
        "edition": "Fourth Revision",
        "identity_source": "extracted",
        "is_eligible": 1,
        "is_stub": 0,
        "page_count": 114,
        "volumes": [
            {"file_name": "is.456.2000.pdf", "page_count": 114, "page_offset": 0, "part_index": 1},
        ],
    },
    {
        "id": "IS_3370_1_2021",
        "family": "IS",
        "number": "3370",
        "part": "1",
        "year": 2021,
        "code": "IS 3370 (Part 1):2021",
        "title": "Concrete Structures for the Storage of Liquids - Code of Practice: Part 1 General Requirements",
        "status": "current",
        "edition": "Second Revision",
        "identity_source": "extracted",
        "is_eligible": 1,
        "is_stub": 0,
        "page_count": 42,
        "volumes": [
            {"file_name": "is.3370.1.2021.pdf", "page_count": 42, "page_offset": 0, "part_index": 1},
        ],
    },
    {
        "id": "IS_2645_2003",
        "family": "IS",
        "number": "2645",
        "part": None,
        "year": 2003,
        "code": "IS 2645:2003",
        "title": "Integral Waterproofing Compounds for Cement Mortar and Concrete - Specification",
        "status": "current",
        "edition": "Second Revision",
        "identity_source": "extracted",
        "is_eligible": 1,
        "is_stub": 0,
        "page_count": 18,
        "volumes": [
            {"file_name": "is.2645.2003.pdf", "page_count": 18, "page_offset": 0, "part_index": 1},
        ],
    },
    {
        "id": "IS_9999_2010",
        "family": "IS",
        "number": "9999",
        "part": None,
        "year": 2010,
        "code": "IS 9999:2010",
        "title": "Withdrawn Specification for Legacy Testing",
        "status": "withdrawn",
        "edition": "First Revision",
        "identity_source": "extracted",
        "is_eligible": 0,
        "is_stub": 0,
        "page_count": 10,
        "volumes": [
            {"file_name": "is.9999.2010.pdf", "page_count": 10, "page_offset": 0, "part_index": 1},
        ],
    },
    {
        "id": "IS_1786_2008",
        "family": "IS",
        "number": "1786",
        "part": None,
        "year": 2008,
        "code": "IS 1786:2008",
        "title": "High Strength Deformed Steel Bars and Wires for Concrete Reinforcement",
        "status": "current",
        "edition": "Fourth Revision",
        "identity_source": "unverified_reference",
        "is_eligible": 0,
        "is_stub": 1,
        "page_count": 24,
        "volumes": [],
    },
]

SAMPLE_CHUNKS = [
    {
        "chunk_id": "chk-sp7-01",
        "logical_doc_id": "SP_7_2026",
        "standard_code": "SP 7:2026",
        "source_file_id": "SP 7_2026-1-400.pdf",
        "page_start": 1,
        "page_end": 2,
        "absolute_page_start": 1,
        "absolute_page_end": 2,
        "section_number": "1",
        "section_title": "Scope",
        "text": "This standard covers the detailed requirements for building construction, structural safety, fire and life safety, building services, and plumbing services across residential and commercial occupancies.",
        "is_eligible": 1,
    },
    {
        "chunk_id": "chk-sp7-02",
        "logical_doc_id": "SP_7_2026",
        "standard_code": "SP 7:2026",
        "source_file_id": "SP 7_2026-1-400.pdf",
        "page_start": 150,
        "page_end": 151,
        "absolute_page_start": 150,
        "absolute_page_end": 151,
        "section_number": "4.2",
        "section_title": "Fire and Life Safety Exits",
        "text": "Fire exit door width for cinema halls, auditoriums, and assembly buildings shall have minimum clear width of 2.0 meters. Pressurization of escape routes shall prevent smoke ingress.",
        "is_eligible": 1,
    },
    {
        "chunk_id": "chk-sp7-03",
        "logical_doc_id": "SP_7_2026",
        "standard_code": "SP 7:2026",
        "source_file_id": "SP 7_2026-401-800.pdf",
        "page_start": 10,
        "page_end": 11,
        "absolute_page_start": 410,
        "absolute_page_end": 411,
        "section_number": "5.3",
        "section_title": "Steel Reinforcement Materials",
        "text": "High strength deformed steel bars and wires for concrete reinforcement shall strictly conform to IS 1786:2008.",
        "is_eligible": 1,
    },
    {
        "chunk_id": "chk-sp7-04",
        "logical_doc_id": "SP_7_2026",
        "standard_code": "SP 7:2026",
        "source_file_id": "SP 7_2026-2001-2400.pdf",
        "page_start": 120,
        "page_end": 121,
        "absolute_page_start": 2120,
        "absolute_page_end": 2121,
        "section_number": "8.4",
        "section_title": "Pressurization and Smoke Control",
        "text": "Emergency pressurization of escape routes in high-rise buildings shall maintain a minimum differential pressure of 50 Pa against adjacent spaces.",
        "is_eligible": 1,
    },
    {
        "chunk_id": "chk-is456-01",
        "logical_doc_id": "IS_456_2000",
        "standard_code": "IS 456:2000",
        "source_file_id": "is.456.2000.pdf",
        "page_start": 1,
        "page_end": 2,
        "absolute_page_start": 1,
        "absolute_page_end": 2,
        "section_number": "1",
        "section_title": "Scope",
        "text": "This standard deals with the general structural use of plain and reinforced concrete in construction, specifying materials, design criteria, durability, and workmanship.",
        "is_eligible": 1,
    },
    {
        "chunk_id": "chk-is456-02",
        "logical_doc_id": "IS_456_2000",
        "standard_code": "IS 456:2000",
        "source_file_id": "is.456.2000.pdf",
        "page_start": 19,
        "page_end": 20,
        "absolute_page_start": 19,
        "absolute_page_end": 20,
        "section_number": "6.1",
        "section_title": "Concrete Mix Proportioning",
        "text": "For reinforced concrete under severe environmental exposure, the minimum cement content for M30 grade concrete shall be 320 kg/m3 with maximum water-cement ratio 0.45.",
        "is_eligible": 1,
    },
    {
        "chunk_id": "chk-is3370-01",
        "logical_doc_id": "IS_3370_1_2021",
        "standard_code": "IS 3370 (Part 1):2021",
        "source_file_id": "is.3370.1.2021.pdf",
        "page_start": 1,
        "page_end": 2,
        "absolute_page_start": 1,
        "absolute_page_end": 2,
        "section_number": "1",
        "section_title": "Scope",
        "text": "This standard lays down general requirements for the design and construction of underground water storage reservoir and reinforced concrete structures for the storage of liquids.",
        "is_eligible": 1,
    },
    {
        "chunk_id": "chk-is3370-02",
        "logical_doc_id": "IS_3370_1_2021",
        "standard_code": "IS 3370 (Part 1):2021",
        "source_file_id": "is.3370.1.2021.pdf",
        "page_start": 14,
        "page_end": 15,
        "absolute_page_start": 14,
        "absolute_page_end": 15,
        "section_number": "4.3",
        "section_title": "Crack Width and Liquid Tightness",
        "text": "Liquid storage structures using grade M30 reinforced concrete must maintain maximum crack width under 0.2 mm to ensure liquid tightness, incorporating waterproofing admixture conforming to IS 2645.",
        "is_eligible": 1,
    },
    {
        "chunk_id": "chk-is2645-01",
        "logical_doc_id": "IS_2645_2003",
        "standard_code": "IS 2645:2003",
        "source_file_id": "is.2645.2003.pdf",
        "page_start": 1,
        "page_end": 2,
        "absolute_page_start": 1,
        "absolute_page_end": 2,
        "section_number": "1",
        "section_title": "Scope",
        "text": "This standard specifies requirements for integral waterproofing compounds for cement mortar and concrete used in liquid retaining structures and basements.",
        "is_eligible": 1,
    },
    {
        "chunk_id": "chk-is9999-01",
        "logical_doc_id": "IS_9999_2010",
        "standard_code": "IS 9999:2010",
        "source_file_id": "is.9999.2010.pdf",
        "page_start": 1,
        "page_end": 2,
        "absolute_page_start": 1,
        "absolute_page_end": 2,
        "section_number": "1",
        "section_title": "Scope",
        "text": "This standard is obsolete and withdrawn. It should not be recommended for modern civil infrastructure.",
        "is_eligible": 0,
    },
]

SAMPLE_RELATIONSHIPS = [
    {
        "source_code": "IS 3370 (Part 1):2021",
        "target_code": "IS 456:2000",
        "relationship_type": "normative_reference",
        "page_number": 2,
        "evidence_text": "General structural concrete design shall conform to IS 456.",
    },
    {
        "source_code": "IS 3370 (Part 1):2021",
        "target_code": "IS 2645:2003",
        "relationship_type": "allied_standard",
        "page_number": 15,
        "evidence_text": "Waterproofing admixtures shall conform to IS 2645.",
    },
    {
        "source_code": "SP 7:2026",
        "target_code": "IS 1786:2008",
        "relationship_type": "normative_reference",
        "page_number": 410,
        "evidence_text": "Steel reinforcement shall conform to IS 1786:2008.",
    },
]

SAMPLE_SCOPE_TEXTS = {
    "SP 7:2026": "This standard covers the detailed requirements for building construction, structural safety, fire and life safety.",
    "IS 456:2000": "This standard deals with the general structural use of plain and reinforced concrete in construction.",
    "IS 3370 (Part 1):2021": "This standard lays down general requirements for the design and construction of underground water storage reservoir and concrete structures.",
    "IS 2645:2003": "This standard specifies requirements for integral waterproofing compounds for cement mortar and concrete.",
    "IS 9999:2010": "Withdrawn legacy standard.",
    "IS 1786:2008": "High strength deformed steel bars and wires for concrete reinforcement.",
}

SAMPLE_OPERATOR_OVERRIDES = {
    "overrides": [
        {
            "source_filename": "SP 7_2026-401-800.pdf",
            "file_sha256": "4a7b9c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b",
            "asserted_standard": {
                "family": "SP",
                "number": "7",
                "part": None,
                "year": 2026,
                "code": "SP 7:2026",
            },
            "part_index": 2,
            "page_offset": 400,
            "operator": "lead-architect",
            "timestamp": "2026-11-11T09:00:00Z",
            "rationale": "SP 7:2026 Volume 2 containing Part 3 Structural Design and Part 4 Fire Protection",
        }
    ]
}

# Queries for Tier 1, 2, 3, 4
QUERY_UNDERGROUND_RESERVOIR = (
    "Design and construction of an underground water storage reservoir using grade M30 reinforced "
    "concrete, requiring minimum cement content of 320 kg/m3, maximum crack width under 0.2 mm, "
    "and waterproofing admixture conforming to BIS specifications."
)

QUERY_FIRE_EXIT = "Fire exit door width for cinema halls and assembly buildings"

QUERY_PRESSURIZATION = "pressurization of escape routes"

QUERY_STEEL_BARS_STUB = "Specification for cold-worked high strength deformed steel bars for concrete reinforcement as per IS 1786"

QUERY_WIND_EFFECTS = "building design considering wind effects on occupant comfort inside and outside"

QUERY_CENTRIFUGAL_PUMP = "general-purpose centrifugal pumps for industrial water service"

QUERY_ABSTENTION_RANDOM = "xyzzy qwerty random unmatched strings for non-existent domain 123456789"

QUERY_EXTREME_LENGTH = "structural reinforced concrete " * 250  # > 4000 chars

QUERY_WHITESPACE = "   \t\n   "
