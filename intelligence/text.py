"""Shared deterministic tokenisation used for transparent scoring and constraints."""

from __future__ import annotations

import re


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?", re.I)
_STOP_WORDS = frozenset({"a", "an", "and", "are", "for", "from", "in", "is", "of", "on", "or", "the", "to", "with"})


def tokenize(text: str) -> frozenset[str]:
    return frozenset(token.lower() for token in _TOKEN_RE.findall(text) if token.lower() not in _STOP_WORDS)
