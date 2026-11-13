# ADR-0002: `src/` Layout and `speccipher` Namespace

- **Status**: Accepted
- **Date**: 2026-11-13
- **Deciders**: SpecCipher Architecture Team

## Context

The legacy packages lived at repository root, where tests and scripts could import them directly without exercising an installed package. This made package boundaries and import behavior depend on the working directory. SpecCipher 2.0 now keeps its application package under `src/speccipher` and declares package discovery in `pyproject.toml`.

## Decision

Use the `src/` layout with `speccipher` as the distribution's import namespace. Build and install the project through its PEP 517/621 `pyproject.toml` configuration. The current console entry point is `speccipher.cli.manage:main`.

## Consequences

Imports resolve from the installed package rather than from repository-root modules, making packaging boundaries explicit. Local development and tests need the project installed or an equivalent configured source path. The CLI's current entry point remains authoritative until intentionally changed.

## Alternatives considered

- Keep root-level import packages: rejected because it permits accidental imports that do not represent the installed distribution.
- Change the CLI entry point to `speccipher.cli.main:main`: deferred because that module is not the entry point declared by the current project metadata.
