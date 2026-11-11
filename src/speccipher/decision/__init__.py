"""Decision & Applicability Engine Layer (Layer 4 Sibling).

Provides deterministic constraint evaluation, applicability scoring, relationship graph
traversal, and categorical confidence assessment with machine-readable reasons.

INVARIANT: Sibling engines in Layer 4 must NEVER import each other!
"""

from __future__ import annotations

from .confidence import calculate_categorical_confidence

__all__ = ["calculate_categorical_confidence"]
