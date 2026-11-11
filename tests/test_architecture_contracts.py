"""Automated architecture contract tests for SpecCipher.

Validates the 7-layer modular monolith boundaries, Layer 4 sibling engine
independence, offline/serving plane isolation, pure domain stdlib kernel,
and zero heavy initialization on import.
"""

from __future__ import annotations

import ast
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "speccipher"

# Layer mapping (higher number = lower in dependency hierarchy)
LAYER_MAP: dict[str, int] = {
    "api": 1,
    "cli": 1,
    "services": 2,
    "release": 3,
    "retrieval": 4,
    "decision": 4,
    "ingestion": 4,
    "understanding": 4,
    "explanation": 4,
    "catalog": 5,
    "pdfio": 5,
    "config": 6,
    "domain": 7,
}

LAYER4_SIBLINGS = {"retrieval", "decision", "ingestion", "understanding", "explanation"}
SERVING_PLANE = {"api", "services", "retrieval", "decision", "understanding", "explanation"}


def _get_python_files(directory: Path) -> Iterator[Path]:
    """Yield all Python files within directory, skipping __pycache__ and test files."""
    if not directory.exists():
        return
    for path in directory.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        yield path


def _get_module_package(file_path: Path) -> str | None:
    """Return the speccipher subpackage name for a given source file."""
    try:
        rel = file_path.relative_to(SRC_ROOT)
        return rel.parts[0] if rel.parts else None
    except ValueError:
        return None


def _extract_imported_modules(file_path: Path) -> list[tuple[str, int]]:
    """Extract list of (imported_module_name, line_number) from a python file."""
    text = file_path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(file_path))
    imports: list[tuple[str, int]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                pkg = _get_module_package(file_path)
                if pkg:
                    rel_parts = list(file_path.relative_to(SRC_ROOT).parts[:-1])
                    for _ in range(node.level - 1):
                        if rel_parts:
                            rel_parts.pop()
                    prefix = "speccipher." + ".".join(rel_parts) if rel_parts else "speccipher"
                    target = f"{prefix}.{node.module}" if node.module else prefix
                    imports.append((target, node.lineno))
            elif node.module:
                imports.append((node.module, node.lineno))

    return imports


@pytest.mark.contract
class TestArchitectureContracts:
    """Enforces architectural constraints and boundary invariants."""

    def test_domain_purity_stdlib_only(self) -> None:
        """speccipher.domain MUST depend ONLY on Python standard library and itself."""
        domain_dir = SRC_ROOT / "domain"
        if not domain_dir.exists():
            pytest.skip("src/speccipher/domain not yet present")

        stdlib_modules = set(sys.stdlib_module_names) | {"__future__"}
        violations: list[str] = []

        for py_file in _get_python_files(domain_dir):
            for imported_mod, lineno in _extract_imported_modules(py_file):
                top_pkg = imported_mod.split(".")[0]
                if top_pkg in stdlib_modules:
                    continue
                if imported_mod.startswith("speccipher.domain"):
                    continue
                violations.append(
                    f"{py_file.name}:{lineno} imports '{imported_mod}' which is not in stdlib or speccipher.domain"
                )

        assert not violations, (
            "Domain purity contract violated! speccipher.domain must depend ONLY on stdlib:\n"
            + "\n".join(violations)
        )

    def test_no_upward_layer_imports(self) -> None:
        """Enforces the 7-layer strict downward dependency rule."""
        if not SRC_ROOT.exists():
            pytest.skip("src/speccipher not yet present")

        violations: list[str] = []

        for py_file in _get_python_files(SRC_ROOT):
            src_pkg = _get_module_package(py_file)
            if not src_pkg or src_pkg not in LAYER_MAP:
                continue
            src_layer = LAYER_MAP[src_pkg]

            for imported_mod, lineno in _extract_imported_modules(py_file):
                if not imported_mod.startswith("speccipher."):
                    continue
                parts = imported_mod.split(".")
                if len(parts) < 2:
                    continue
                target_pkg = parts[1]
                if target_pkg not in LAYER_MAP:
                    continue
                target_layer = LAYER_MAP[target_pkg]

                if target_layer < src_layer:
                    violations.append(
                        f"Layer {src_layer} ({src_pkg}) {py_file.name}:{lineno} imports "
                        f"Layer {target_layer} ({target_pkg}) '{imported_mod}'"
                    )

        assert not violations, "Upward layer import violations detected:\n" + "\n".join(violations)

    def test_layer4_engine_independence(self) -> None:
        """Layer 4 sibling engines MUST NEVER import each other."""
        if not SRC_ROOT.exists():
            pytest.skip("src/speccipher not yet present")

        violations: list[str] = []

        for py_file in _get_python_files(SRC_ROOT):
            src_pkg = _get_module_package(py_file)
            if src_pkg not in LAYER4_SIBLINGS:
                continue

            for imported_mod, lineno in _extract_imported_modules(py_file):
                if not imported_mod.startswith("speccipher."):
                    continue
                parts = imported_mod.split(".")
                if len(parts) < 2:
                    continue
                target_pkg = parts[1]
                if target_pkg in LAYER4_SIBLINGS and target_pkg != src_pkg:
                    violations.append(
                        f"Layer 4 sibling breach: {src_pkg}/{py_file.name}:{lineno} imports sibling '{imported_mod}'"
                    )

        assert not violations, "Layer 4 sibling independence contract violated:\n" + "\n".join(
            violations
        )

    def test_serving_plane_does_not_import_ingestion(self) -> None:
        """Online serving plane modules MUST NEVER import speccipher.ingestion."""
        if not SRC_ROOT.exists():
            pytest.skip("src/speccipher not yet present")

        violations: list[str] = []

        for py_file in _get_python_files(SRC_ROOT):
            src_pkg = _get_module_package(py_file)
            if src_pkg not in SERVING_PLANE:
                continue

            for imported_mod, lineno in _extract_imported_modules(py_file):
                if imported_mod == "speccipher.ingestion" or imported_mod.startswith(
                    "speccipher.ingestion."
                ):
                    violations.append(
                        f"Serving plane boundary violation: {src_pkg}/{py_file.name}:{lineno} imports '{imported_mod}'"
                    )

        assert not violations, (
            "Serving plane boundary violated! Serving plane must never import ingestion plane:\n"
            + "\n".join(violations)
        )

    def test_lint_imports_cli_execution(self) -> None:
        """Runs the external import-linter tool if installed."""
        lint_imports_bin = shutil.which("lint-imports")
        if not lint_imports_bin:
            candidate = Path(sys.executable).parent / (
                "lint-imports.exe" if sys.platform == "win32" else "lint-imports"
            )
            if candidate.exists():
                lint_imports_bin = str(candidate)
        if not lint_imports_bin:
            pytest.skip("lint-imports CLI not installed in current environment")

        result = subprocess.run(
            [lint_imports_bin],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"lint-imports failed:\n{result.stdout}\n{result.stderr}"

    def test_zero_heavy_init_on_module_import(self) -> None:
        """Application and service modules must import quickly with zero heavy resource loading."""
        if not (SRC_ROOT / "api").exists() or not (SRC_ROOT / "services").exists():
            pytest.skip("speccipher.api or speccipher.services not yet present")

        script = (
            "import time; t0 = time.perf_counter(); "
            "import speccipher.services.container; "
            "import speccipher.api.app; "
            "t1 = time.perf_counter(); "
            "print(f'IMPORT_TIME={t1 - t0:.4f}')"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"Import failed: {proc.stderr}"
        for line in proc.stdout.splitlines():
            if line.startswith("IMPORT_TIME="):
                elapsed = float(line.split("=")[1])
                assert elapsed < 1.0, f"Import took {elapsed:.2f}s; violates zero heavy init rule"
