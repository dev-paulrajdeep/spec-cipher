from pathlib import Path


RESULTS_SCRIPT = Path(__file__).parents[1] / "frontend" / "js" / "results.js"


def test_frontend_renders_bounded_recommendations_field() -> None:
    source = RESULTS_SCRIPT.read_text(encoding="utf-8")
    assert "response?.recommendations" in source
    assert ".slice(0, 5)" in source
    assert "recommendation?.is_code" in source


def test_frontend_labels_retrieved_candidates_and_threshold_no_match() -> None:
    source = RESULTS_SCRIPT.read_text(encoding="utf-8")
    assert "retrieved_candidate_count" in source
    assert "Retrieved candidates" in source
    assert "Retrieved candidates did not meet the" in source
    assert "did not return any matching document chunks" not in source