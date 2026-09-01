"""Explainable set-overlap scoring primitives."""

from __future__ import annotations


def required_term_coverage(required: frozenset[str], available: frozenset[str]) -> float:
    """Fraction of explicit required terms evidenced by candidate text, or zero if absent."""
    return len(required.intersection(available)) / len(required) if required else 0.0
