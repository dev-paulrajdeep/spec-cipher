"""Atomic promotion and CURRENT pointer updates for immutable corpus releases."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from speccipher.release.manifest import ReleaseIntegrityError, verify_release


class ReleasePublishError(RuntimeError):
    """Raised when a release cannot be promoted or selected safely."""


class ReleasePublisher:
    """Promote verified staging releases and atomically select a release by ID."""

    def __init__(self, staging_root: Path | str, releases_root: Path | str):
        self.staging_root = Path(staging_root).resolve()
        self.releases_root = Path(releases_root).resolve()
        self.current_path = self.releases_root / "CURRENT"

    def publish(self, staging_release: Path | str) -> Path:
        """Verify and atomically promote one direct child of the staging root."""
        stage_path = Path(staging_release).resolve()
        if stage_path.parent != self.staging_root:
            raise ReleasePublishError("Staging release must be a direct child of staging_root")
        if not stage_path.is_dir():
            raise ReleasePublishError(f"Staging release does not exist: {stage_path}")

        try:
            manifest = verify_release(stage_path, check_fts_integrity=True)
        except (ReleaseIntegrityError, FileNotFoundError) as exc:
            raise ReleasePublishError(f"Staging release failed verification: {exc}") from exc

        release_id = manifest["release_id"]
        target_path = self.releases_root / release_id
        if target_path.exists() or target_path.is_symlink():
            raise ReleasePublishError(f"Release already exists and is immutable: {release_id}")

        self.releases_root.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(stage_path, target_path)
        except OSError as exc:
            raise ReleasePublishError(
                f"Could not atomically promote release; staging and releases must share a filesystem: {exc}"
            ) from exc

        self._write_current(release_id)
        return target_path

    def rollback(self, release_id: str) -> Path:
        """Switch CURRENT to a previously published, verified release."""
        if (
            not isinstance(release_id, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", release_id) is None
        ):
            raise ReleasePublishError(f"Invalid release ID: {release_id!r}")
        target_path = self.releases_root / release_id
        if target_path.is_symlink() or target_path.resolve().parent != self.releases_root:
            raise ReleasePublishError("Rollback target must be a direct release directory")
        try:
            manifest = verify_release(target_path)
        except (ReleaseIntegrityError, FileNotFoundError) as exc:
            raise ReleasePublishError(f"Rollback target failed verification: {exc}") from exc
        if manifest["release_id"] != release_id:
            raise ReleasePublishError(
                "Rollback target manifest does not match requested release ID"
            )

        self._write_current(release_id)
        return target_path

    def _write_current(self, release_id: str) -> None:
        """Atomically replace CURRENT from a temporary file on the same filesystem."""
        self.releases_root.mkdir(parents=True, exist_ok=True)
        manifest_path = self.releases_root / release_id / "release.json"
        try:
            release_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        except OSError as exc:
            raise ReleasePublishError(
                f"Could not read release manifest for CURRENT pointer: {exc}"
            ) from exc
        pointer_payload = json.dumps(
            {"release_id": release_id, "release_sha256": f"sha256:{release_digest}"},
            separators=(",", ":"),
            sort_keys=True,
        )
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                prefix=".CURRENT.",
                suffix=".tmp",
                dir=self.releases_root,
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                stream.write(pointer_payload + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, self.current_path)
        except OSError as exc:
            raise ReleasePublishError(f"Could not atomically update CURRENT: {exc}") from exc
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


def publish_release(
    staging_root: Path | str,
    releases_root: Path | str,
    staging_release: Path | str,
) -> dict[str, Any]:
    """Convenience function returning manifest metadata after promotion."""
    publisher = ReleasePublisher(staging_root, releases_root)
    promoted_path = publisher.publish(staging_release)
    return verify_release(promoted_path)
