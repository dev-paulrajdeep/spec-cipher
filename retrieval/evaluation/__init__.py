"""Objective candidate-retrieval metrics and compact gold-set runner."""

from .metrics import EvaluationCase, EvaluationSummary, evaluate, mrr_at_k, precision_at_k, recall_at_k
from .gold import DEMO_GOLD_CASES

__all__ = ["DEMO_GOLD_CASES", "EvaluationCase", "EvaluationSummary", "evaluate", "mrr_at_k", "precision_at_k", "recall_at_k"]
