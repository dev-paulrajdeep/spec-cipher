"""SpecCipher: Deterministic Indian Standards Discovery & Recommendation Engine.

Modular Monolith architecture with strict 7-layer boundary enforcement:
- Layer 1: Presentation & Delivery (speccipher.api, speccipher.cli)
- Layer 2: Application Services & Composition Root (speccipher.services)
- Layer 3: Corpus Release & Lifecycle Management (speccipher.release)
- Layer 4: Core Domain Engines (retrieval, decision, ingestion, understanding, explanation)
- Layer 5: Infrastructure & Persistence (speccipher.catalog, speccipher.pdfio)
- Layer 6: Configuration (speccipher.config)
- Layer 7: Pure Domain Kernel (speccipher.domain - standard library only)
"""

from __future__ import annotations

__version__ = "0.2.0"

__all__ = ["__version__"]
