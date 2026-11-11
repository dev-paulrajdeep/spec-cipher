import json
import subprocess
import sys
from pathlib import Path


def test_metadata_validation_reports_actual_production_coverage() -> None:
    """Recommendation index coverage reflects eligible chunks only.

    With catalog/list-only officialBIS sources, the recommendation FAISS bundle
    correctly contains zero vectors.
    """
    root = Path(__file__).parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/validate_retrieval_metadata.py", "retrieval-index"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )

    manifest = json.loads((root / "corpus" / "data" / "manifest.json").read_text(encoding="utf-8"))
    expected_eligible = int(manifest.get("chunk_count_eligible", 0))
    assert f"Total chunks: {expected_eligible}" in result.stdout
    assert "TEMP-" not in result.stdout
    if expected_eligible == 0:
        assert "Chunks with standard_number: 0" in result.stdout
        assert "Unique IS standards: 0" in result.stdout
    else:
        assert "Chunks without standard_number: 0" in result.stdout


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
