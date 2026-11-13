# ADR-0008: Optional, Fail-Open Explanations

- **Status**: Accepted
- **Date**: 2026-11-13
- **Deciders**: SpecCipher Architecture Team

## Context

Recommendation selection is deterministic and must remain usable when the optional Ollama explanation service is unavailable or slow. A generative explanation must not change the selected standards, ranking, or recommendation confidence.

## Decision

Keep explanation generation outside the recommendation decision. The versioned API exposes `POST /v1/explain` as a separate on-demand operation. It revalidates the requested standard and evidence against the mounted release before generating an explanation. The explanation service uses a configured timeout and returns a deterministic explanation when generative output is unavailable; failures do not revise a finalized recommendation.

## Consequences

Clients can request narrative output separately, and the recommendation path does not depend on successful generation. Explanation requests still perform validation and retrieval synchronously; the API is on-demand, not an asynchronous job system. Timeout and fallback behavior remain configuration and implementation details that tests should keep aligned.

## Alternatives considered

- Generate explanations inline while ranking: rejected because optional model availability would couple recommendation delivery to generation.
- Queue asynchronous explanation jobs: deferred because the current API provides a synchronous on-demand endpoint and no job lifecycle.
