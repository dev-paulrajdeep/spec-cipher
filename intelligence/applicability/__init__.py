"""Deterministic, decomposed degree-of-fit assessment for admissible candidates."""

from .assessor import ApplicabilityAssessor
from .feature_extraction import DeterministicRequirementInterpreter, RequirementFeatures, RequirementInterpreter

__all__ = ["ApplicabilityAssessor", "DeterministicRequirementInterpreter", "RequirementFeatures", "RequirementInterpreter"]
