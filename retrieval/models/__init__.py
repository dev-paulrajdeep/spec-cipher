"""Models that expose retrieval output without exposing FAISS internals."""

from .candidate import StandardCandidate, group_results_by_standard
from .result import RetrievalResult

__all__ = ["RetrievalResult", "StandardCandidate", "group_results_by_standard"]
