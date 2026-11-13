import json
import subprocess
import sys
from pathlib import Path


def test_metadata_validation_reports_bundle_coverage(tmp_path) -> None:
    """The validator counts identified and unresolved rows in a bundle."""
    root = Path(__file__).parents[1]
    bundle = tmp_path / "retrieval-index"
    bundle.mkdir()
    (bundle / "chunks.json").write_text(
        json.dumps(
            [
                {"chunk_id": "chunk-1", "standard_number": "IS 456:2000"},
                {"chunk_id": "chunk-2", "standard_number": None},
            ]
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "scripts/validate_retrieval_metadata.py", str(bundle)],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Total chunks: 2" in result.stdout
    assert "TEMP-" not in result.stdout
    assert "Chunks with standard_number: 1" in result.stdout
    assert "Chunks without standard_number: 1" in result.stdout
    assert "Unique IS standards: 1" in result.stdout


def test_metadata_validation_uses_parent_identity_for_canonical_json(tmp_path) -> None:
    source = tmp_path / "standard.json"
    source.write_text(
        '{"metadata":{"standard_number":"IS 456:2000","source_file":"is456.pdf","page_count":1},"chunks":[{"chunk_id":"c1",'
        '"standard_number":null,"source_file":"is456.pdf","page_start":1,"page_end":1,'
        '"section_number":null,"section_title":null,"text":"concrete"}]}',
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "scripts/validate_retrieval_metadata.py", str(source)],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Chunks with standard_number: 1" in result.stdout
    assert "Unique IS standards: 1" in result.stdout
