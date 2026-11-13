"""SpecCipher Pure Domain Kernel (Layer 7).

All entities in this package depend exclusively on the Python standard library.
No third-party packages (pydantic, faiss, numpy, flask, etc.) are imported.
"""

from .candidates import Candidate, RequirementFeatures, RerankScore
from .errors import (
    DomainError,
    MultipartAssemblyError,
    OverrideValidationError,
    ProvenanceValidationError,
)
from .identity import StandardId, StandardStub, format_standard_code
from .models import DocumentChunk, LogicalDocument, MissingVolume, Section, SourceFile
from .recommendation import Confidence, ConfidenceLevel, Recommendation
from .relationships import Relationship, RelationshipRole, StandardRelationship
from .requirements import Requirement

__all__ = [
    # Errors
    "DomainError",
    "MultipartAssemblyError",
    "OverrideValidationError",
    "ProvenanceValidationError",
    # Identity
    "StandardId",
    "StandardStub",
    "format_standard_code",
    # Models
    "SourceFile",
    "LogicalDocument",
    "MissingVolume",
    "Section",
    "DocumentChunk",
    # Relationships
    "RelationshipRole",
    "StandardRelationship",
    "Relationship",
    # Candidates
    "Candidate",
    "RerankScore",
    "RequirementFeatures",
    "Requirement",
    # Recommendation
    "ConfidenceLevel",
    "Confidence",
    "Recommendation",
]
