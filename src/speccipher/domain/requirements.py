"""Pure requirement terms used by deterministic serving decisions."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .candidates import RequirementFeatures

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "by",
        "for",
        "from",
        "in",
        "into",
        "is",
        "of",
        "on",
        "or",
        "shall",
        "the",
        "to",
        "with",
    }
)
_MATERIAL_TERMS = frozenset(
    {
        "asphalt",
        "bitumen",
        "bronze",
        "cement",
        "concrete",
        "iron",
        "masonry",
        "polymer",
        "stainless",
        "steel",
        "timber",
        "water",
    }
)
_APPLICATION_TERMS = frozenset(
    {
        "building",
        "commercial",
        "firefighting",
        "foundation",
        "industrial",
        "irrigation",
        "municipal",
        "pipeline",
        "reservoir",
        "roof",
        "storage",
        "tank",
        "waterproofing",
    }
)
_PRODUCT_TERMS = frozenset(
    {
        "boiler",
        "centrifugal",
        "compressor",
        "efficiency",
        "impeller",
        "induction",
        "motor",
        "pump",
        "transformer",
        "valve",
        "volute",
        "waterproofing",
    }
)


def tokenize(text: str) -> frozenset[str]:
    """Return normalized, non-stopword terms from text."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return frozenset(token for token in _TOKEN_RE.findall(text.lower()) if token not in _STOP_WORDS)


@dataclass(frozen=True)
class Requirement:
    """Original requirement and its reproducible lexical feature representation."""

    text: str
    features: RequirementFeatures

    @classmethod
    def from_text(cls, text: str) -> Requirement:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("requirement must be a non-empty string")
        tokens = tokenize(text)
        materials = tokens & _MATERIAL_TERMS
        applications = tokens & _APPLICATION_TERMS
        products = tokens & _PRODUCT_TERMS
        if not products:
            products = tokens - materials - applications
        return cls(
            text=text,
            features=RequirementFeatures(
                requirement_tokens=tokens,
                product_terms=products,
                material_terms=materials,
                application_terms=applications,
            ),
        )
