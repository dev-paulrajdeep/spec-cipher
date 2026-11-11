"""Explanation Engine Layer (Layer 4 Sibling).

Provides fail-open, non-blocking LLM reasoning adapters with strict guardrails
ensuring explanations cannot alter finalized recommendations.

INVARIANT: Sibling engines in Layer 4 must NEVER import each other!
"""

from __future__ import annotations

__all__: list[str] = []
