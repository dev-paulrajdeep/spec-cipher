"""Browser acceptance test for the complete SpecCipher application path.

The script starts only the Flask process it owns, drives the real UI with
Playwright, and writes machine-readable evidence plus screenshots under the
selected artifacts directory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUERY = "general-purpose centrifugal pumps for industrial water service"
DEFAULT_NO_RESULT_QUERY = "fire safety requirements for a commercial building"
DEFAULT_MODEL = "qwen3-4b-thinking:latest"
DEFAULT_PORT = 5000


class SmokeFailure(RuntimeError):
    """A failure that should make the acceptance command exit non-zero."""


@dataclass
class StartedServer:
    process: subprocess.Popen[bytes]
    log_handle: Any
    log_path: Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.getenv("SPEC_CIPHER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("SPEC_CIPHER_PORT", DEFAULT_PORT)))
    parser.add_argument("--timeout", type=float, default=180.0, help="Per-operation timeout in seconds")
    parser.add_argument("--headed", action="store_true", help="Show Chromium while running")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--no-result-query", default=DEFAULT_NO_RESULT_QUERY)
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / ".tmp" / "e2e_smoke")
    return parser.parse_args(argv)


def http_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 5.0) -> tuple[int, Any]:
    if payload is None:
        request = Request(url, method="GET")
    else:
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return response.status, json.loads(body) if body else None
    except HTTPError as error:
        body = error.read().decode("utf-8")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = body
        return error.code, parsed


def http_text(url: str, timeout: float = 5.0) -> tuple[int, str]:
    request = Request(url, method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8")
    except HTTPError as error:
        return error.code, error.read().decode("utf-8", errors="replace")


def wait_for_ready(base_url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error = "no response"
    while time.monotonic() < deadline:
        try:
            status, body = http_text(f"{base_url}/")
            if status == 200 and "spec-analysis-form" in body:
                return
            last_error = f"GET / returned HTTP {status}"
        except (OSError, URLError, ValueError) as error:
            last_error = str(error)
        time.sleep(0.25)
    raise SmokeFailure(f"Flask application did not become ready within {timeout:.0f}s: {last_error}")


def verify_index() -> None:
    required = ("index.faiss", "chunks.json", "chunk_ids.json", "metadata.json")
    missing = [name for name in required if not (ROOT / "retrieval-index" / name).is_file()]
    if missing:
        raise SmokeFailure(f"Persisted retrieval index is incomplete; missing: {', '.join(missing)}")


def verify_ollama(model_name: str, timeout: float = 5.0) -> None:
    host = os.getenv("QWEN_OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    try:
        status, payload = http_json(f"{host}/api/tags", timeout=timeout)
    except (OSError, URLError, ValueError) as error:
        raise SmokeFailure(f"Ollama is unavailable at {host}: {error}") from error
    if status != 200 or not isinstance(payload, dict):
        raise SmokeFailure(f"Ollama health check failed at {host}: HTTP {status}")
    models = {item.get("name") for item in payload.get("models", []) if isinstance(item, dict)}
    if model_name not in models:
        raise SmokeFailure(f"Required Qwen model {model_name!r} is not installed in Ollama")


def verify_prerequisites(model_name: str) -> None:
    if not sys.executable:
        raise SmokeFailure("Python executable is not available")
    if not (ROOT / "backend" / "app.py").is_file():
        raise SmokeFailure("Backend entrypoint backend/app.py is missing")
    if not (ROOT / "frontend" / "index.html").is_file():
        raise SmokeFailure("Frontend entrypoint frontend/index.html is missing")
    verify_index()
    try:
        import playwright  # noqa: F401
    except ImportError as error:
        raise SmokeFailure("Playwright is not installed in the active Python environment") from error
    verify_ollama(model_name)


def start_server(host: str, port: int, artifacts_dir: Path, qwen_host: str | None = None) -> StartedServer:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    suffix = "failure" if qwen_host else "success"
    log_path = artifacts_dir / f"backend-{suffix}.log"
    log_handle = log_path.open("wb")
    environment = os.environ.copy()
    environment.update({
        "PYTHONUNBUFFERED": "1",
        "SPEC_CIPHER_HOST": host,
        "SPEC_CIPHER_PORT": str(port),
    })
    if qwen_host is not None:
        environment["QWEN_OLLAMA_HOST"] = qwen_host
        environment["QWEN_TIMEOUT_SECONDS"] = "1"
    process = subprocess.Popen(
        [sys.executable, "backend/app.py"],
        cwd=ROOT,
        env=environment,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )
    return StartedServer(process, log_handle, log_path)


def stop_server(server: StartedServer) -> None:
    if server.process.poll() is None:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(server.process.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            os.killpg(os.getpgid(server.process.pid), signal.SIGTERM)
        try:
            server.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.process.kill()
            server.process.wait(timeout=5)
    server.log_handle.close()


def stable_recommendation(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key not in {"explanation", "warnings"}}


def rendered_score(score: Any) -> str:
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        return "-"
    if 0 <= score <= 1:
        return f"{score:.3f}"
    return str(score)


def response_for_ui(page: Any, query: str, timeout_ms: int) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    with page.expect_response(
        lambda response: response.url.endswith("/get_result") and response.request.method == "POST",
        timeout=timeout_ms,
    ) as response_info:
        page.locator("#spec-raw-text").fill(query)
        page.locator("#btn-submit-analysis").click()
    response = response_info.value
    elapsed = time.perf_counter() - started
    if response.status != 200:
        raise SmokeFailure(f"Browser request to /get_result returned HTTP {response.status}")
    return response.json(), elapsed


def assert_success_page(page: Any, payload: dict[str, Any], query: str, timeout_ms: int) -> dict[str, Any]:
    if payload.get("status") != "success" or payload.get("user_input") != query:
        raise SmokeFailure("Browser response did not contain the expected successful query payload")
    recommendations = payload.get("recommendations")
    if not isinstance(recommendations, list) or not recommendations:
        raise SmokeFailure("Successful browser query produced no recommendation")
    first = recommendations[0]
    standard_number = first.get("standard_number")
    title = first.get("title") or first.get("section_title")
    evidence = first.get("evidence") or []
    primary_evidence = evidence[0] if evidence else {}
    evidence_text = first.get("extracted_content") or primary_evidence.get("text")
    explanation = first.get("explanation")
    if not standard_number or not title or not evidence_text:
        raise SmokeFailure("Recommendation response is missing standard number, title, or evidence")
    if not isinstance(explanation, dict) or not explanation.get("summary"):
        raise SmokeFailure("Explanation is missing from the successful response")

    page.locator("#results-content-area").wait_for(state="visible", timeout=timeout_ms)
    page.locator(".standard-record-card").first.wait_for(state="visible", timeout=timeout_ms)
    text = page.locator("#results-content-area").inner_text()
    reasoning_panel = page.locator(".standard-record-card").first.locator(".record-reasoning-panel")
    reasoning_panel.wait_for(state="visible", timeout=timeout_ms)
    reasoning_text = reasoning_panel.inner_text()
    score = first.get("score", first.get("final_score"))
    for required in (standard_number, title, "Relevance", "Applicability", rendered_score(score), f"#{first.get('rank')}" ):
        if str(required) not in text:
            raise SmokeFailure(f"Rendered recommendation page is missing {required!r}")
    for required in ("Extracted content", evidence_text[:40], "Explanation"):
        if str(required).casefold() not in reasoning_text.casefold():
            raise SmokeFailure(
                f"Rendered recommendation page is missing {required!r}; "
                f"reasoning text was {reasoning_text!r}"
            )
    if explanation["summary"] not in reasoning_text:
        raise SmokeFailure("Rendered recommendation page does not contain the explanation summary")
    related_topics = first.get("related_topics") or []
    if related_topics:
        related_node = page.locator('[data-testid="related-topics"]').first
        related_node.wait_for(state="visible", timeout=timeout_ms)
        if related_topics[0] not in related_node.inner_text():
            raise SmokeFailure("Rendered recommendation page does not contain related topics")
    return {
        "standard_number": standard_number,
        "title": title,
        "score": first.get("score", first.get("final_score")),
        "rank": first.get("rank"),
        "evidence_count": len(first.get("evidence", [])) or 1,
        "explanation": explanation,
        "related_standards_count": len(first.get("related_standards") or []),
    }


def run_browser_flow(args: argparse.Namespace, base_url: str, model_name: str) -> dict[str, Any]:
    from playwright.sync_api import sync_playwright

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    server = start_server(args.host, args.port, args.artifacts_dir)
    result: dict[str, Any] = {"query": args.query, "qwen_latency_seconds": None}
    try:
        wait_for_ready(base_url, args.timeout)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=not args.headed)
            context = browser.new_context()
            page = context.new_page()
            page.goto(f"{base_url}/", wait_until="domcontentloaded", timeout=int(args.timeout * 1000))
            result_payload, latency = response_for_ui(page, args.query, int(args.timeout * 1000))
            result["http_status"] = 200
            result["total_user_visible_latency_seconds"] = latency
            result["recommendation"] = assert_success_page(page, result_payload, args.query, int(args.timeout * 1000))
            page.screenshot(path=str(args.artifacts_dir / "successful-result.png"), full_page=True)
            result["recommendation_integrity_baseline"] = stable_recommendation(result_payload["recommendations"][0])

            stop_server(server)
            server = start_server(args.host, args.port, args.artifacts_dir, qwen_host="http://127.0.0.1:1")
            wait_for_ready(base_url, args.timeout)
            page.goto(f"{base_url}/", wait_until="domcontentloaded", timeout=int(args.timeout * 1000))
            failure_payload, failure_latency = response_for_ui(page, args.query, int(args.timeout * 1000))
            if failure_payload.get("status") != "success" or not failure_payload.get("recommendations"):
                raise SmokeFailure("Qwen failure isolation did not preserve the recommendation")
            failed_recommendation = failure_payload["recommendations"][0]
            explanation = failed_recommendation.get("explanation")
            if not isinstance(explanation, dict) or not explanation.get("summary"):
                raise SmokeFailure("Qwen failure isolation did not provide a deterministic explanation")
            if not any("AI explanation unavailable" in warning for warning in failed_recommendation.get("warnings", [])):
                raise SmokeFailure("Qwen failure isolation did not expose the existing warning")
            if stable_recommendation(failed_recommendation) != result["recommendation_integrity_baseline"]:
                raise SmokeFailure("Qwen failure changed recommendation fields")
            page.locator("#results-content-area").wait_for(state="visible", timeout=int(args.timeout * 1000))
            if "AI explanation unavailable" not in page.locator("#results-global-warnings").inner_text():
                raise SmokeFailure("Qwen failure warning was not rendered in the UI")
            result["failure_isolation"] = {"status": "passed", "latency_seconds": failure_latency}

            page.goto(f"{base_url}/", wait_until="domcontentloaded", timeout=int(args.timeout * 1000))
            no_result_payload, no_result_latency = response_for_ui(page, args.no_result_query, int(args.timeout * 1000))
            if no_result_payload.get("status") != "success" or no_result_payload.get("recommendations") != []:
                raise SmokeFailure("No-result query did not return the existing empty recommendation contract")
            if not any("No sufficiently supported standard" in warning for warning in no_result_payload.get("warnings", [])):
                raise SmokeFailure("No-result query did not return the existing no-result warning")
            page.locator("#results-content-area").wait_for(state="visible", timeout=int(args.timeout * 1000))
            if "No sufficiently supported standard found" not in page.locator("#ranked-standards-list").inner_text():
                raise SmokeFailure("No-result UI did not render the existing no-match behavior")
            if "AI explanation unavailable" in page.locator("#results-global-warnings").inner_text():
                raise SmokeFailure("No-result path attempted Qwen explanation")
            result["no_result"] = {"status": "passed", "latency_seconds": no_result_latency}
            browser.close()
    finally:
        stop_server(server)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model_name = os.getenv("QWEN_MODEL_NAME", DEFAULT_MODEL)
    summary: dict[str, Any] = {"query": args.query, "model": model_name, "status": "failed"}
    try:
        verify_prerequisites(model_name)
        base_url = f"http://{args.host}:{args.port}"
        summary.update(run_browser_flow(args, base_url, model_name))
        summary["status"] = "passed"
        print(json.dumps(summary, indent=2, sort_keys=True))
        (args.artifacts_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        return 0
    except SmokeFailure as error:
        summary["failure"] = str(error)
        args.artifacts_dir.mkdir(parents=True, exist_ok=True)
        (args.artifacts_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(f"E2E FAIL: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        summary["failure"] = f"Unexpected {error.__class__.__name__}: {error}"
        args.artifacts_dir.mkdir(parents=True, exist_ok=True)
        (args.artifacts_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(f"E2E FAIL: {summary['failure']}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
