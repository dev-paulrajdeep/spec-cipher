# ADR-0009: Versioned API and Legacy Compatibility

- **Status**: Accepted
- **Date**: 2026-11-13
- **Deciders**: SpecCipher Architecture Team

## Context

SpecCipher 2.0 provides structured versioned endpoints while the existing static frontend and integrations still use the legacy `POST /get_result` response contract. Replacing the legacy contract outright would force those clients to migrate in lockstep.

## Decision

Expose the current analysis API at `POST /v1/analyze` and preserve `POST /get_result` as a compatibility endpoint. Keep the legacy response serialization in the API layer so existing frontend fields continue to be provided while clients can migrate to the versioned response. The versioned API also exposes `/health/live`, `/health/ready`, and `POST /v1/explain`.

## Consequences

The API supports incremental client migration, and the legacy serializer remains a compatibility surface that must be covered by contract tests. Both endpoints should delegate through the configured application container and preserve release-backed decision behavior.

## Alternatives considered

- Remove `/get_result` as part of the v1 rollout: rejected because the existing frontend depends on that contract.
- Make the legacy endpoint redirect clients to v1: rejected because the response schemas are different and existing clients require the legacy payload shape.
