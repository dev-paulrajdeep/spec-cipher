"""Ingestion operator overrides package."""
from .loader import OperatorOverride, validate_operator_override

__all__ = ["OperatorOverride", "validate_operator_override"]
