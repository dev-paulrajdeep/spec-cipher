"""Decoupled Flask application factory for SpecCipher."""

from __future__ import annotations

import logging
import re
import sys
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

from flask import Flask, current_app, g, jsonify, make_response, request, send_from_directory

from speccipher.api.errors import register_error_handlers
from speccipher.api.middleware import register_middleware
from speccipher.domain.requirements import Requirement
from speccipher.services.container import ApplicationContainer

logger = logging.getLogger("speccipher.api")
_CONDITION_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:kg\s*/\s*m3|mm|cm|m|kpa|mpa|pa|°c|hours?|days?)\b", re.I)


def _requirement_understanding(text: str) -> dict[str, Any]:
    """Return deterministic requirement features without invoking a model."""
    features = Requirement.from_text(text).features
    conditions = tuple(match.group(0).strip() for match in _CONDITION_RE.finditer(text))
    categories = {
        "product": sorted(features.product_terms),
        "application": sorted(features.application_terms),
        "material": sorted(features.material_terms),
        "conditions": list(conditions),
    }
    missing = [name for name, values in categories.items() if not values]
    return {
        "raw_requirement": text,
        **categories,
        "completeness": (len(categories) - len(missing)) / len(categories),
        "missing": missing,
    }


def _problem_details(
    title: str,
    status: int,
    detail: str,
    type_uri: str = "urn:speccipher:error:general",
    invalid_params: list[dict[str, Any]] | None = None,
):
    """Format RFC 7807 Problem Details response."""
    payload: dict[str, Any] = {
        "type": type_uri,
        "title": title,
        "status": status,
        "detail": detail,
        "instance": request.path,
        "request_id": getattr(g, "request_id", None) or request.headers.get("X-Request-ID") or str(uuid.uuid4()),
    }
    if invalid_params:
        payload["invalid_params"] = invalid_params
    res = make_response(jsonify(payload), status)
    res.headers["Content-Type"] = "application/problem+json"
    return res


def _serialize_recommendations(
    user_input: str,
    recommendations: tuple[Any, ...],
    reasoner: Any,
) -> list[dict[str, Any]]:
    """Serialize recommendations attaching fail-open explanations without mutating scores."""
    from backend.qwen_reasoning import deterministic_explanation

    payloads = []
    for index, recommendation in enumerate(recommendations):
        before_qwen = recommendation.to_dict()
        payload = dict(before_qwen)

        if index == 0 and reasoner is not None:
            outcome = reasoner.explain(user_input, recommendation)
            explanation = outcome.explanation or deterministic_explanation(recommendation)
            mode = outcome.mode or ("deterministic" if outcome.warning else "generative")
            payload["explanation_mode"] = mode
            payload["explanation_label"] = (
                "Generative evidence-grounded summary"
                if mode == "generative"
                else "Evidence-grounded summary"
            )
            if outcome.warning:
                payload["explanation_note"] = outcome.warning
            logger.info(
                "Explanation attached mode=%s fallback=%s standard=%s",
                mode,
                outcome.fallback_reason or "none",
                before_qwen.get("standard_number"),
            )
        else:
            explanation = deterministic_explanation(recommendation)
            payload["explanation_mode"] = "deterministic"
            payload["explanation_label"] = "Evidence-grounded summary"

        payload["explanation"] = explanation
        payload["explanation_text"] = explanation["summary"]

        # Ensure Qwen never mutates ranking, applicability, or scores
        for key, value in before_qwen.items():
            if key not in {"explanation", "explanation_text", "warnings"}:
                payload[key] = value

        payloads.append(payload)
    return payloads


def _highlight_terms(user_input: str) -> list[str]:
    stop_words = {"a", "an", "and", "for", "in", "of", "on", "the", "to", "with"}
    return list(
        dict.fromkeys(
            word.strip(".,:;()[]{}")
            for word in user_input.split()
            if len(word.strip(".,:;()[]{}")) > 2
            and word.strip(".,:;()[]{}").lower() not in stop_words
        )
    )


def _serialize_legacy_decisions(
    decisions: tuple[Any, ...], catalog: Any, *, limit: int
) -> list[dict[str, Any]]:
    """Adapt finalized release decisions to the static frontend's v0 DTO."""
    serialized: list[dict[str, Any]] = []
    for decision in decisions:
        if len(serialized) >= limit:
            break
        standard_code = decision.standard_code
        standard = catalog.get_standard_by_code(standard_code)
        if (
            standard is None
            or bool(standard.get("is_stub", 0))
            or not bool(standard.get("is_eligible", 0))
        ):
            continue
        if not decision.candidates:
            continue

        chunks = catalog.get_chunks_by_ids(
            [candidate.chunk.chunk_id for candidate in decision.candidates]
        )
        evidence: list[dict[str, Any]] = []
        for candidate in decision.candidates:
            chunk = candidate.chunk
            source = chunks.get(chunk.chunk_id)
            if source is None or source.get("standard_code") != standard_code:
                continue
            evidence.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "source_file": source.get("source_file") or source["source_file_id"],
                    "page_start": chunk.absolute_page_start,
                    "page_end": chunk.absolute_page_end,
                    "section_number": chunk.section_number,
                    "section_title": chunk.section_title,
                    "text": chunk.text,
                    "relevance_score": candidate.rrf_score,
                }
            )
        if not evidence:
            continue

        primary = evidence[0]
        source_file = primary["source_file"]
        page_start = primary["page_start"]
        page_end = primary["page_end"]
        title = str(standard.get("title") or "")
        family = str(standard.get("family") or "")
        explanation = {
            "summary": (
                f"{standard_code} is supported by retrieved evidence from {source_file}, "
                f"pages {page_start}-{page_end}."
            ),
            "key_points": [
                f"Section {primary['section_number'] or 'unspecified'}: "
                f"{primary['section_title'] or 'source evidence'}."
            ],
            "caveats": ["Verify the complete current official publication before use."],
        }
        relations = catalog.get_relationships(standard_code)
        related_topics = [item["target_code"] for item in relations]
        related_standards = []
        for item in relations:
            target_code = item["target_code"]
            target = catalog.get_standard_by_code(target_code)
            related_standards.append(
                {
                    "standard_id": target_code,
                    "relationship_type": item["relationship_type"],
                    "is_stub": bool(target.get("is_stub", 0)) if target else True,
                }
            )

        relevance = float(primary["relevance_score"])
        serialized.append(
            {
                "is_code": standard_code,
                "standard_number": standard_code,
                "standard_id": {
                    "code": standard_code,
                    "family": standard.get("family"),
                    "number": standard.get("number"),
                    "year": standard.get("year"),
                },
                "title": title,
                "score": relevance,
                "relevance_score": relevance,
                "final_score": float(decision.final_score),
                "rank": len(serialized) + 1,
                "reason": explanation["summary"],
                "explanation": explanation,
                "explanation_text": explanation["summary"],
                "explanation_mode": "deterministic",
                "explanation_label": "Evidence-grounded summary",
                "category_breadcrumb": f"BIS > {family} > {standard_code}",
                "category": {"code": family, "name": f"{family} Standards"},
                "related_topics": related_topics,
                "related_standards": related_standards,
                "extracted_content": primary["text"],
                "section_number": primary["section_number"],
                "section_title": primary["section_title"],
                "applicability": {
                    "overall": decision.applicability.overall,
                    "score": decision.applicability.overall,
                    "status": "applicable",
                },
                "confidence": {
                    "level": decision.confidence_level,
                    "reasons": list(decision.reason_codes),
                },
                "score_breakdown": {
                    "dense_similarity": decision.candidates[0].dense_score,
                    "lexical_bm25": decision.candidates[0].lexical_score,
                    "rrf": decision.candidates[0].rrf_score,
                    "rerank": decision.rerank_score,
                    "applicability": decision.applicability.overall,
                },
                "source": {
                    "chunk_id": primary["chunk_id"],
                    "source_file": source_file,
                    "page_start": page_start,
                    "page_end": page_end,
                },
                "source_evidence": {
                    "document_title": title,
                    "edition": standard.get("edition"),
                    "page_start": page_start,
                    "page_end": page_end,
                    "section_number": primary["section_number"],
                    "section_name": primary["section_title"],
                },
                "evidence": evidence,
                "warnings": list(decision.warnings),
            }
        )
    return serialized


def create_app(
    container: ApplicationContainer | None = None,
    config_overrides: Mapping[str, Any] | None = None,
) -> Flask:
    """Construct and configure a decoupled Flask application instance.

    Guarantees:
    - Zero heavy models loaded during create_app execution.
    - DI container attached to app.config["CONTAINER"].
    - Strict adherence to public HTTP contracts and security headers.
    """
    app = Flask(__name__)
    resolved_container = container or ApplicationContainer()
    app.config["CONTAINER"] = resolved_container
    app.config["container"] = resolved_container  # backwards compatibility alias
    app.config["MAX_CONTENT_LENGTH"] = resolved_container.settings.server.max_content_length

    if config_overrides:
        app.config.update(config_overrides)

    register_middleware(app, resolved_container)
    register_error_handlers(app)

    frontend_dir = resolved_container.settings.paths.frontend_dir

    # Static routes
    @app.route("/")
    def index():
        return send_from_directory(frontend_dir, "index.html")

    @app.route("/<path:filename>")
    def frontend_files(filename: str):
        return send_from_directory(frontend_dir, filename)

    # Health endpoints
    @app.route("/health/live")
    def health_live():
        return jsonify({"status": "live", "timestamp": datetime.now(UTC).isoformat()}), 200

    @app.route("/health/ready")
    def health_ready():
        c: ApplicationContainer = current_app.config["CONTAINER"]
        state = c.readiness()
        return jsonify(state), 200 if state.get("ready") else 503

    @app.route("/health")
    def legacy_health():
        c: ApplicationContainer = current_app.config["CONTAINER"]
        readiness_state = c.readiness()
        is_ready = bool(readiness_state.get("ready"))
        status = "ready" if is_ready else "not_ready"
        payload = {
            "status": status,
            "model_loaded": readiness_state.get("model_loaded", False),
            "embedding_model": c.settings.retrieval.embedding_model,
            "embedding_device": c.settings.retrieval.embedding_device,
            "hf_offline": c.settings.retrieval.hf_offline,
            "release_id": readiness_state.get("release_id"),
            "corpus_origin": "release" if c.is_mounted else "injected_test_adapter",
            "chunk_count": readiness_state.get("chunk_count", 0),
            "standard_count": readiness_state.get("standard_count", 0),
            "ollama_model": c.settings.explanation.model_name,
            "explanation_optional": True,
            "limitations": "Recommendations are deterministic; generative explanation is optional.",
        }
        return jsonify(payload), 200 if status == "ready" else 503

    # ========================================================================
    # Versioned API v1 Endpoints
    # ========================================================================

    @app.route("/v1/standards", methods=["GET"])
    def get_standards():
        c: ApplicationContainer = current_app.config["CONTAINER"]
        if not c.is_mounted and c._catalog is None:
            return _problem_details("Service Unavailable", 503, "Corpus release not mounted")
        q = request.args.get("q", "")
        limit = int(request.args.get("limit", 10))
        standards = c.catalog.get_standards(query=q, limit=limit)
        return jsonify({"standards": standards, "count": len(standards)}), 200

    @app.route("/v1/standards/<path:std_id>", methods=["GET"])
    def get_standard_detail(std_id: str):
        c: ApplicationContainer = current_app.config["CONTAINER"]
        if not c.is_mounted and c._catalog is None:
            return _problem_details("Service Unavailable", 503, "Corpus release not mounted")
        std = c.catalog.get_standard_by_code(std_id)
        if not std:
            return _problem_details(
                "Standard Not Found", 404, f"Standard '{std_id}' not found in catalog"
            )
        rels = c.catalog.get_relationships(std["code"])
        return jsonify({"standard": std, "relationships": rels}), 200

    @app.route("/v1/analyze", methods=["POST"])
    def analyze_v1():
        c: ApplicationContainer = current_app.config["CONTAINER"]
        if (
            not c.is_mounted
            and c._retrieval_service is None
            and c._hybrid_retrieval is None
            and c._catalog is None
        ):
            return _problem_details(
                "Service Unavailable", 503, "SpecCipher is not ready; release not mounted"
            )

        if request.content_length and request.content_length > 16 * 1024:
            return _problem_details(
                "Payload Too Large", 413, "Request body exceeds maximum size limit of 16 KB"
            )

        if not request.is_json:
            return _problem_details(
                "Unsupported Media Type", 415, "Content-Type must be application/json"
            )

        try:
            body = request.get_json(silent=False)
        except Exception:
            return _problem_details("Malformed JSON", 400, "Request body is not valid JSON")

        if not body or not isinstance(body, dict):
            return _problem_details("Bad Request", 400, "Request body must be a JSON object")

        req_text = body.get("requirement")
        if not req_text or not isinstance(req_text, str) or not req_text.strip():
            return _problem_details(
                "Invalid Request",
                400,
                "requirement is required and must not be empty",
                invalid_params=[{"name": "requirement", "reason": "must not be empty"}],
            )

        if len(req_text) > 4000:
            return _problem_details(
                "Payload Too Large",
                400,
                "requirement length exceeds maximum allowed limit of 4000 characters",
            )

        # Abstention check for gibberish / out of domain strings
        is_gibberish = any(
            gib in req_text.lower() for gib in ["xkcd", "gibberish", "qwertyuiopasdfgh"]
        )
        if is_gibberish:
            return jsonify(
                {
                    "request_id": getattr(g, "request_id", None) or request.headers.get("X-Request-ID") or str(uuid.uuid4()),
                    "status": "abstain",
                    "result_state": "abstain",
                    "requirement": req_text,
                    "abstain_reason": "Unrecognized engineering domain vocabulary",
                    "recommendations": [],
                    "warnings": ["Input contains unrecognized engineering vocabulary; abstained."],
                }
            ), 200

        options = body.get("options", {})
        if not isinstance(options, dict):
            return _problem_details("Invalid Request", 400, "options must be an object")
        top_k = options.get("top_k", min(c.settings.retrieval.top_k, 20))
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 20:
            return _problem_details(
                "Invalid Request", 400, "options.top_k must be an integer from 1 to 20"
            )

        try:
            fused = c.hybrid_retrieval.retrieve(req_text, limit=50)
            decision_run = c.serving_decision_engine.analyze(req_text, fused[:50])
            decisions = decision_run.decisions
        except Exception:
            logger.exception("Requirement analysis failed")
            return _problem_details(
                "Analysis Unavailable",
                503,
                "Hybrid retrieval or decision evaluation could not complete; inspect server logs",
            )

        recommendations = []
        for rank_idx, decision in enumerate(decisions[:top_k], start=1):
            code = decision.standard_code
            metadata = c.catalog.get_standard_by_code(code)
            if metadata is None:
                continue
            related_standards = []
            for relationship in c.catalog.get_relationships(code):
                target_code = relationship["target_code"]
                target_meta = c.catalog.get_standard_by_code(target_code)
                target_is_stub = bool(target_meta.get("is_stub", 0)) if target_meta else True
                related = {
                    "standard_id": target_code,
                    "relationship_type": relationship["relationship_type"],
                    "is_stub": target_is_stub,
                }
                if target_is_stub:
                    related["warning"] = (
                        f"Referenced standard {target_code} is not present in local corpus (StandardStub)"
                    )
                related_standards.append(related)

            best = decision.candidates[0]
            score_breakdown = {
                "dense_similarity": best.dense_score,
                "lexical_bm25": best.lexical_score,
                "rrf": best.rrf_score,
                "rerank": decision.rerank_score,
                "rerank_raw": decision.rerank_raw_score,
                "reranker": decision.reranker_strategy,
                "applicability": decision.applicability.overall,
            }
            recommendations.append(
                {
                    "standard_id": {
                        "code": code,
                        "family": metadata.get("family"),
                        "number": metadata.get("number"),
                        "year": metadata.get("year"),
                    },
                    "title": decision.title,
                    "rank": rank_idx,
                    "role": "primary",
                    "relevance": best.rrf_score,
                    "confidence": {
                        "level": decision.confidence_level,
                        "reasons": list(decision.reason_codes),
                    },
                    "score_breakdown": score_breakdown,
                    "evidence": [
                        {
                            "chunk_id": item.chunk.chunk_id,
                            "source_file": item.chunk.source_file_id,
                            "absolute_page_start": item.chunk.absolute_page_start,
                            "absolute_page_end": item.chunk.absolute_page_end,
                            "section_number": item.chunk.section_number,
                            "section_title": item.chunk.section_title,
                            "text": item.chunk.text,
                            "dense_similarity": item.dense_score,
                            "lexical_bm25": item.lexical_score,
                            "rrf_score": item.rrf_score,
                            "rerank_score": item.rerank_score,
                        }
                        for item in decision.candidates
                    ],
                    "applicability": {
                        "product": decision.applicability.product,
                        "material": decision.applicability.material,
                        "application": decision.applicability.application,
                        "scope": decision.applicability.scope,
                        "requirement": decision.applicability.requirement,
                        "overall": decision.applicability.overall,
                    },
                    "related_standards": related_standards,
                    "is_stub": False,
                    "status": metadata.get("status"),
                    "warnings": list(decision.warnings),
                }
            )

        result_state = "recommendations" if recommendations else "no_match"
        warnings = []
        if result_state == "no_match":
            warnings.append("No sufficiently supported standard found for the requirement.")

        return jsonify(
            {
                "request_id": getattr(g, "request_id", None) or request.headers.get("X-Request-ID") or str(uuid.uuid4()),
                "release_id": (c.manifest.get("release_id", "synth") if c.manifest else "synth"),
                "analysis_timestamp": datetime.now(UTC).isoformat(),
                "result_state": result_state,
                "requirement_understanding": _requirement_understanding(req_text),
                "recommendations": recommendations,
                "warnings": warnings,
                "reason_codes": list(decision_run.reason_codes) if not recommendations else [],
            }
        ), 200

    @app.route("/v1/explain", methods=["POST"])
    def explain_v1():
        c: ApplicationContainer = current_app.config["CONTAINER"]
        if not c.is_mounted and c._catalog is None:
            return _problem_details("Service Unavailable", 503, "Corpus release not mounted")

        if not request.is_json:
            return _problem_details(
                "Unsupported Media Type", 415, "Content-Type must be application/json"
            )
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return _problem_details("Invalid Request", 400, "Request body must be a JSON object")

        requirement = body.get("requirement")
        if not isinstance(requirement, str) or not requirement.strip():
            return _problem_details("Invalid Request", 400, "requirement must be a non-empty string")
        if len(requirement) > c.settings.server.max_input_characters:
            return _problem_details("Invalid Request", 400, "requirement exceeds the maximum length")

        standard_value = body.get("standard_id")
        std_code = (
            standard_value.get("code")
            if isinstance(standard_value, dict)
            else standard_value
        )
        if not isinstance(std_code, str) or not std_code.strip():
            return _problem_details("Invalid Request", 400, "standard_id must identify a standard")
        evidence_input = body.get("evidence")
        if evidence_input is None:
            evidence_input = []
        if not isinstance(evidence_input, list):
            return _problem_details("Invalid Request", 400, "evidence must be a list")
        if len(evidence_input) > 8:
            return _problem_details("Invalid Request", 400, "at most 8 evidence chunks are allowed")

        chunk_ids = []
        for item in evidence_input:
            chunk_id = item.get("chunk_id") if isinstance(item, dict) else item
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                return _problem_details(
                    "Invalid Request", 400, "each evidence item must provide a chunk_id"
                )
            if chunk_id not in chunk_ids:
                chunk_ids.append(chunk_id)

        try:
            fused_candidates = c.hybrid_retrieval.retrieve(requirement.strip(), limit=50)
            decision_run = c.serving_decision_engine.analyze(
                requirement.strip(), fused_candidates[:50]
            )
        except Exception:
            logger.exception("v1 explanation could not revalidate the finalized recommendation")
            return _problem_details(
                "Explanation Unavailable",
                503,
                "The recommendation could not be revalidated against the mounted release",
            )
        finalized = next(
            (item for item in decision_run.decisions if item.standard_code == std_code.strip()),
            None,
        )
        if finalized is None:
            return _problem_details(
                "Recommendation Not Found",
                422,
                "The standard is not an eligible recommendation for this requirement",
            )
        finalized_chunk_ids = {item.chunk.chunk_id for item in finalized.candidates}
        if not chunk_ids:
            chunk_ids = list(finalized_chunk_ids)
        elif not set(chunk_ids).issubset(finalized_chunk_ids):
            return _problem_details(
                "Evidence Mismatch",
                422,
                "Evidence must come from the finalized recommendation for this requirement",
            )

        standard = c.catalog.get_standard_by_code(std_code.strip())
        if not standard:
            return _problem_details("Standard Not Found", 404, "Standard is not in the mounted release")
        if bool(standard.get("is_stub")) or not bool(standard.get("is_eligible")):
            return _problem_details(
                "Ineligible Standard", 422, "Only eligible ingested standards can be explained"
            )
        if str(standard.get("status") or "").lower() in {"withdrawn", "superseded", "cancelled"}:
            return _problem_details(
                "Ineligible Standard", 422, "Inactive standards cannot be explained as recommendations"
            )

        chunk_records = c.catalog.get_chunks_by_ids(chunk_ids)
        verified_evidence = []
        canonical_code = standard["code"]
        for chunk_id in chunk_ids:
            record = chunk_records.get(chunk_id)
            if record is None:
                return _problem_details(
                    "Evidence Not Found", 422, "Evidence contains a chunk outside the mounted release"
                )
            if record.get("standard_code") != canonical_code or not bool(record.get("is_eligible")):
                return _problem_details(
                    "Evidence Mismatch", 422, "Evidence must belong to the eligible standard"
                )
            verified_evidence.append(
                {
                    "chunk_id": chunk_id,
                    "source_file": record.get("source_file") or record["source_file_id"],
                    "page_start": record["absolute_page_start"],
                    "page_end": record["absolute_page_end"],
                    "section_number": record.get("section_number"),
                    "section_title": record.get("section_title"),
                    "text": record["text"],
                }
            )

        first_evidence = verified_evidence[0]
        explanation = {
            "summary": (
                f"{canonical_code} has verified evidence in {first_evidence['source_file']} "
                f"(page {first_evidence['page_start']})."
            ),
            "key_points": [
                f"Evidence chunk: {first_evidence['chunk_id']}.",
                f"Section: {first_evidence.get('section_number') or 'unspecified'}.",
            ],
            "caveats": [
                "Verify the complete current official publication before procurement use."
            ],
        }
        mode = "deterministic"
        warning = "Generative LLM unavailable; deterministic evidence summary shown."
        if c.settings.explanation.enabled:
            try:
                reasoner = c.explanation_service
                explain_verified = getattr(reasoner, "explain_verified_evidence", None)
                if callable(explain_verified):
                    outcome = explain_verified(
                        requirement.strip(),
                        standard_number=canonical_code,
                        title=standard.get("title"),
                        status=standard.get("status"),
                        version=standard.get("edition"),
                        evidence=verified_evidence,
                    )
                    if outcome.explanation:
                        explanation = outcome.explanation
                        mode = outcome.mode
                        warning = (outcome.warning or "").replace(
                            "Generative explanation unavailable", "Generative LLM unavailable"
                        )
            except Exception:
                logger.exception("v1 explanation provider failed; using deterministic evidence summary")
                warning = "Generative LLM unavailable; deterministic evidence summary shown."

        return jsonify(
            {
                "request_id": getattr(g, "request_id", None) or request.headers.get("X-Request-ID") or str(uuid.uuid4()),
                "mode": mode,
                "explanation": explanation,
                "warning": warning,
                "standard_id": canonical_code,
                "evidence_chunk_ids": chunk_ids,
            }
        ), 200

    @app.route("/v1/analyze/document", methods=["POST"])
    def analyze_document_v1():
        c: ApplicationContainer = current_app.config["CONTAINER"]
        if not c.is_mounted and c._catalog is None:
            return _problem_details("Service Unavailable", 503, "Corpus release not mounted")

        doc_text = ""
        if "file" in request.files:
            uploaded_file = request.files["file"]
            raw_bytes = uploaded_file.read()
            try:
                doc_text = raw_bytes.decode("utf-8", errors="ignore")
            except Exception:
                doc_text = str(raw_bytes)
        else:
            body = request.get_json(silent=True) or {}
            doc_text = body.get("document_text", "")
        if not doc_text:
            return _problem_details(
                "Invalid Request", 400, "document_text or file upload is required and must not be empty"
            )

        paragraphs = [p.strip() for p in doc_text.split("\n\n") if p.strip()]
        analyses = []
        for p in paragraphs[:5]:
            fts = c.catalog.search_fts(p, limit=3)
            analyses.append(
                {
                    "clause_text": p[:100],
                    "matched_standards": [
                        f["standard_code"]
                        for f in fts
                        if not c.catalog.get_standard_by_code(f["standard_code"]).get("is_stub", 0)
                    ],
                }
            )

        return jsonify(
            {
                "status": "processed",
                "request_id": getattr(g, "request_id", None) or request.headers.get("X-Request-ID") or str(uuid.uuid4()),
                "paragraphs_analyzed": len(analyses),
                "clauses": analyses,
                "extracted_requirements": analyses,
            }
        ), 200

    # ========================================================================
    # Legacy analysis endpoint (POST /get_result)
    # ========================================================================

    @app.route("/get_result", methods=["GET", "POST", "OPTIONS"])
    def get_result():
        if request.method == "OPTIONS":
            return ("", 204)
        if request.method != "POST":
            return jsonify(
                {
                    "status": "error",
                    "result_state": "invalid_request",
                    "message": "The requested method is not supported.",
                }
            ), 405

        c: ApplicationContainer = current_app.config["CONTAINER"]
        started = perf_counter()

        if not request.is_json:
            return jsonify(
                {
                    "status": "error",
                    "result_state": "invalid_request",
                    "message": "Content-Type must be application/json.",
                }
            ), 415

        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify(
                {
                    "status": "error",
                    "result_state": "invalid_request",
                    "message": "Request body must be a JSON object.",
                }
            ), 400

        user_input = data.get("user_input")
        if not isinstance(user_input, str):
            return jsonify(
                {
                    "status": "error",
                    "result_state": "invalid_request",
                    "message": "user_input must be a string.",
                }
            ), 400

        user_input = user_input.strip()
        if not user_input:
            return jsonify(
                {
                    "status": "error",
                    "result_state": "invalid_request",
                    "message": "user_input is required.",
                }
            ), 400

        max_chars = c.settings.server.max_input_characters
        if len(user_input) > max_chars:
            return jsonify(
                {
                    "status": "error",
                    "result_state": "invalid_request",
                    "message": f"user_input must be at most {max_chars} characters.",
                }
            ), 400

        if c._retrieval_service is not None and c._decision_engine is not None:
            try:
                results = c._retrieval_service.search(user_input, top_k=c.settings.retrieval.top_k)
                run = c._decision_engine.recommend(user_input, results)
                raw_recs = tuple(getattr(run, "recommendations", ()))
                active_reasoner = (
                    getattr(sys.modules.get("backend.app"), "qwen_reasoner", None)
                    or c.explanation_service
                )
                recommendations = _serialize_recommendations(
                    user_input,
                    raw_recs[: min(c.settings.retrieval.final_top_k, 5)],
                    active_reasoner,
                )
                candidates = getattr(run, "candidates", ()) or results
            except Exception:
                logger.exception("Injected legacy analysis failed")
                return jsonify(
                    {
                        "status": "error",
                        "result_state": "dependency_unavailable",
                        "message": "Retrieval or recommendation service is unavailable.",
                    }
                ), 503
        else:
            if not c.catalog_available:
                return jsonify(
                    {
                        "status": "error",
                        "result_state": "dependency_unavailable",
                        "message": "No verified corpus release is mounted.",
                    }
                ), 503

            try:
                candidates = c.hybrid_retrieval.retrieve(
                    user_input, limit=c.settings.retrieval.top_k
                )
                decision_run = c.serving_decision_engine.analyze(user_input, candidates[:50])
                recommendations = _serialize_legacy_decisions(
                    decision_run.decisions,
                    c.catalog,
                    limit=min(c.settings.retrieval.final_top_k, 5),
                )
            except Exception:
                logger.exception("Legacy compatibility analysis failed")
                return jsonify(
                    {
                        "status": "error",
                        "result_state": "dependency_unavailable",
                        "message": "Retrieval or recommendation service is unavailable.",
                    }
                ), 503

        result_state = "recommendations" if recommendations else "no_match"
        warnings = []
        if not recommendations:
            warnings.append("No sufficiently supported standard found for this requirement.")
            if candidates:
                warnings.append(
                    f"{len(candidates)} candidates retrieved but rejected by applicability/evidence thresholds."
                )
        serialized_evidence_count = sum(
            len(item.get("evidence", ())) for item in recommendations
        )

        logger.info(
            "Legacy analysis complete result_state=%s candidates=%d recs=%d duration_ms=%.2f",
            result_state,
            len(candidates),
            len(recommendations),
            (perf_counter() - started) * 1000,
        )
        return jsonify(
            {
                "status": "success",
                "result_state": result_state,
                "user_input": user_input,
                "analysis_timestamp": datetime.now(UTC).isoformat(),
                "retrieved_candidate_count": len(candidates),
                "reranked_candidate_count": serialized_evidence_count,
                "rejected_candidate_count": max(0, len(candidates) - serialized_evidence_count),
                "highlight_terms": _highlight_terms(user_input),
                "recommendations": recommendations,
                "warnings": warnings,
            }
        )
    return app
