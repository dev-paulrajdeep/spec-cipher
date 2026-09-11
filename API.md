# Backend API

## `POST /get_result`

Submit one procurement requirement as JSON:

```json
{"user_input":"industrial pump construction requirements"}
```

The response is grouped at standard level and contains at most five recommendations. Retrieval candidates are retained internally; they are not returned as one top-level result per chunk.

### Recommendations found: `200 OK`

```json
{
  "status": "success",
  "result_state": "recommendations",
  "user_input": "industrial pump construction requirements",
  "retrieved_candidate_count": 1,
  "reranked_candidate_count": 1,
  "rejected_candidate_count": 0,
  "recommendations": [
    {
      "standard_number": "IS 5120",
      "title": "Technical requirements for general-purpose centrifugal pumps",
      "score": 0.437,
      "rank": 1,
      "source": {
        "chunk_id": "chunk_abc123",
        "source_file": "standard.pdf",
        "page_start": 1,
        "page_end": 1
      },
      "source_evidence": {
        "page_start": 1,
        "page_end": 1,
        "section_number": null,
        "section_name": "Document preamble"
      },
      "evidence": [
        {
          "standard_number": "IS 5120",
          "chunk_id": "chunk_abc123",
          "source_file": "standard.pdf",
          "page_start": 1,
          "page_end": 1,
          "section_number": null,
          "section_title": "Document preamble",
          "text": "Evidence excerpt from the source...",
          "relevance_score": 0.437
        }
      ],
      "explanation": null,
      "explanation_text": "Recommendation is supported by retrieved source evidence.",
      "warnings": ["Explanation unavailable: Ollama request failed"]
    }
  ],
  "warnings": []
}
```

`score` and `relevance_score` are bounded relevance signals used for ordering. They are not calibrated confidence, probability, compliance, or similarity percentages. The UI displays them to three decimal places.

### No sufficiently supported match: `200 OK`

```json
{
  "status": "success",
  "result_state": "no_match",
  "recommendations": [],
  "warnings": ["No sufficiently supported standard was found for this requirement."]
}
```

This state means the service completed but the retrieved evidence did not pass the recommendation gates. It is distinct from an unavailable system.

### Invalid request: `400`, `413`, or `415`

```json
{"status":"error","result_state":"invalid_request","message":"user_input is required."}
```

### Retrieval or recommendation service unavailable: `503`

```json
{
  "status": "error",
  "result_state": "dependency_unavailable",
  "message": "Retrieval dependency is unavailable."
}
```

This state indicates a missing/incompatible bundle, unavailable model, or serving failure. It must not be rendered as a no-match result.

`GET /health` reports the loaded model/device and returns `200` only when the
process has an initialized retrieval model. It returns `503` with
`"status":"not_ready"` otherwise. The endpoint deliberately reports the
canonical corpus filename rather than an absolute workstation path.

The current bundled MVP has one independently indexed `IS 5120` chunk. A
successful recommendation is therefore not evidence of multi-standard or
BIS-wide ranking coverage. The source PDF is not distributed; the canonical
extracted corpus JSON is the available provenance artifact.

## Preflight

Run the static bundle and corpus identity check before starting Flask:

```powershell
.\.venv\Scripts\python.exe scripts\preflight.py
```

Rebuild the exact declared corpus when the check reports a fingerprint mismatch:

```powershell
.\.venv\Scripts\python.exe scripts\build_retrieval_index.py --corpus corpus\data\IS_Codes_for_Industrial_Pumps_Engineer_Checklist.json --output retrieval-index --device cpu --allow-cpu-fallback
```
