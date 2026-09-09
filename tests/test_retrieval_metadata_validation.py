from pathlib import Path
import subprocess
import sys


def test_metadata_validation_reports_actual_production_coverage() -> None:
    root = Path(__file__).parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/validate_retrieval_metadata.py", "retrieval-index"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Total chunks: 13077" in result.stdout
    assert "Chunks without standard_number: 13077" in result.stdout
    assert "Unique IS standards: 0" in result.stdout


def test_metadata_validation_uses_parent_identity_for_canonical_json(tmp_path) -> None:
    source = tmp_path / "standard.json"
    source.write_text(
        '{"metadata":{"standard_number":"IS 456:2000"},"chunks":[{"chunk_id":"c1",'
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