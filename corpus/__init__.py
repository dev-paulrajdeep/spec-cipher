"""Canonical corpus preparation pipeline for SpecCipher."""

from __future__ import annotations

from typing import Any

__all__ = ["CorpusBuilder"]


def __getattr__(name: str) -> Any:
    if name == "CorpusBuilder":
        from .pipeline import CorpusBuilder

        return CorpusBuilder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
