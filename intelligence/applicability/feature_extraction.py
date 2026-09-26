"""Small deterministic requirement interpreter plus an optional structured adapter boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from intelligence.text import tokenize


_MATERIAL_TERMS = frozenset({
    "cement", "concrete", "steel", "timber", "water", "bitumen", "polymer", "masonry", "asphalt",
    "cast", "iron", "bronze", "stainless",
})
_APPLICATION_TERMS = frozenset({
    "reservoir", "storage", "tank", "bridge", "building", "roof", "foundation", "pipeline",
    "waterproofing", "industrial", "commercial", "municipal", "irrigation", "firefighting",
})
_PRODUCT_TERMS = frozenset({
    "pump", "pumps", "centrifugal", "impeller", "volute", "motor", "motors", "induction",
    "efficiency", "valve", "valves", "compressor", "boiler", "transformer", "waterproofing",
})


@dataclass(frozen=True)
class RequirementFeatures:
    requirement_tokens: frozenset[str]
    product_terms: frozenset[str]
    material_terms: frozenset[str]
    application_terms: frozenset[str]


class RequirementInterpreter(Protocol):
    """Replaceable boundary for an optional structured LLM interpreter."""

    def interpret(self, requirement: str) -> RequirementFeatures: ...


class DeterministicRequirementInterpreter:
    """Default interpreter: transparent token extraction with no external model calls."""

    def interpret(self, requirement: str) -> RequirementFeatures:
        if not isinstance(requirement, str) or not requirement.strip():
            raise ValueError("Requirement must be a non-empty string")
        tokens = tokenize(requirement)
        materials = tokens.intersection(_MATERIAL_TERMS)
        applications = tokens.intersection(_APPLICATION_TERMS)
        products = tokens.intersection(_PRODUCT_TERMS)
        # Prefer explicit product lexicon hits; residual tokens remain products
        # so short domain queries like "pump" still carry product signal.
        residual = tokens.difference(materials, applications)
        return RequirementFeatures(
            requirement_tokens=tokens,
            product_terms=products or residual,
            material_terms=materials,
            application_terms=applications,
        )
