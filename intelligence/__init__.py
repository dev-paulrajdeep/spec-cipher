"""Explainable decision engine over retrieval candidates and corpus facts."""

from .config import IntelligenceConfig
from .pipeline import IntelligenceEngine

__all__ = ["IntelligenceConfig", "IntelligenceEngine"]
