"""SpecCipher Pure Domain Kernel (Layer 7).

All entities in this package depend exclusively on the Python standard library.
No third-party packages (pydantic, faiss, numpy, flask, etc.) are imported.
"""

from .candidates import Candidate, RequirementFeatures
from .identity import StandardId, StandardStub, format_standard_code
from .models import DocumentChunk, LogicalDocument, Section, SourceFile
from .recommendation import Confidence, ConfidenceLevel, Recommendation
from .relationships import Relationship, RelationshipRole, StandardRelationship

__all__ = [
    # Identity
    "StandardId",
    "StandardStub",
    "format_standard_code",
    # Models
    "SourceFile",
    "LogicalDocument",
    "Section",
    "DocumentChunk",
    # Relationships
    "RelationshipRole",
    "StandardRelationship",
    "Relationship",
    # Candidates
    "Candidate",
    "RequirementFeatures",
    # Recommendation
    "ConfidenceLevel",
    "Confidence",
    "Recommendation",
]
