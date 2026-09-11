from pathlib import Path


RESULTS_SCRIPT = Path(__file__).parents[1] / "frontend" / "js" / "results.js"
ANALYZE_SCRIPT = Path(__file__).parents[1] / "frontend" / "js" / "analyze.js"
INDEX_HTML = Path(__file__).parents[1] / "frontend" / "index.html"


def test_frontend_renders_bounded_recommendations_field() -> None:
    source = RESULTS_SCRIPT.read_text(encoding="utf-8")
    assert "response?.recommendations" in source
    assert ".slice(0, 5)" in source
    assert "recommendation?.is_code" in source


def test_frontend_labels_retrieved_candidates_and_threshold_no_match() -> None:
    source = RESULTS_SCRIPT.read_text(encoding="utf-8")
    assert "retrieved_candidate_count" in source
    assert "Retrieved candidates" in source
    assert "No sufficiently supported standard found" in source
    assert "applicability/evidence thresholds" in source
    assert "did not return any matching document chunks" not in source


def test_frontend_refine_focuses_requirement_input() -> None:
    source = RESULTS_SCRIPT.read_text(encoding="utf-8")
    assert "btn-refine-spec" in source
    assert "textarea.focus()" in source
    assert "scrollIntoView" in source


def test_frontend_error_panel_and_retry_exist() -> None:
    analyze = ANALYZE_SCRIPT.read_text(encoding="utf-8")
    index = INDEX_HTML.read_text(encoding="utf-8")
    assert "analysis-error-panel" in analyze
    assert "btn-retry-analysis" in analyze
    assert 'maxlength="4000"' in index
    assert "IS 5120" in index


def test_frontend_renders_explanation_mode_labels() -> None:
    source = RESULTS_SCRIPT.read_text(encoding="utf-8")
    assert "explanation_label" in source
    assert "Generative evidence-grounded summary" in source
    assert "Evidence-grounded summary" in source
    assert "explanation-note" in source
    assert "AI explanation unavailable" not in source
