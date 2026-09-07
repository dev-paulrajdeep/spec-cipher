# Backend API — Frontend Integration

This document describes how the frontend communicates with the Flask backend.

## `POST /get_result`

The frontend sends the user's procurement specification to the backend.

### What the frontend sends

The request body must be JSON:

```json
{
  "user_input": "Steel pipes for high pressure water supply"
}
```

The important field is:

```text
user_input
```

It contains the text entered by the user.

---

# What the frontend receives from the backend

## Successful response — `200 OK`

The backend returns JSON in this format:

```json
{
  "status": "success",
  "user_input": "Steel pipes for high pressure water supply",
  "recommendations": [
    {
      "standard_number": "IS XXXX",
      "title": null,
      "section_number": "5.2",
      "section_title": "Steel Pipes",
      "text": "Actual retrieved chunk text...",
      "score": 0.91,
      "rank": 1,
      "source": {
        "chunk_id": "chunk_abc123",
        "source_file": "SP_7_2026.json",
        "page_start": 12,
        "page_end": 14,
        "element_ids": []
      }
    }
  ],
  "warnings": []
}
```

> ## **IMPORTANT — `recommendations` is the main output**
>
> The frontend should read:
>
> ```javascript
> result.recommendations
> ```
>
> This is an **array containing up to 5 retrieved results**.

---

## `recommendations[]`

Each item inside `recommendations` represents **one retrieved document chunk**.

```json
{
  "standard_number": "IS XXXX",
  "title": null,
  "section_number": "5.2",
  "section_title": "Steel Pipes",
  "text": "Actual retrieved chunk text...",
  "score": 0.91,
  "rank": 1,
  "source": {
    "chunk_id": "chunk_abc123",
    "source_file": "SP_7_2026.json",
    "page_start": 12,
    "page_end": 14,
    "element_ids": []
  }
}
```

### Fields received by the frontend

| Field             | Type          | What it contains                                                  |
| ----------------- | ------------- | ----------------------------------------------------------------- |
| `standard_number` | string        | Indian Standard number associated with the retrieved chunk        |
| `title`           | string / null | Standard title; currently returned as `null`                      |
| `section_number`  | string / null | Section number of the retrieved chunk                             |
| `section_title`   | string / null | Section title of the retrieved chunk                              |
| `text`            | string        | **The actual text retrieved from the Indian Standard document**   |
| `score`           | number        | Similarity score between the user's query and the retrieved chunk |
| `rank`            | integer       | Position of the result (`1` = highest-ranked)                     |
| `source`          | object        | Information about the original document/chunk                     |

---

# `source`

The `source` object contains the location/provenance of the retrieved chunk.

```json
"source": {
  "chunk_id": "chunk_abc123",
  "source_file": "SP_7_2026.json",
  "page_start": 12,
  "page_end": 14,
  "element_ids": []
}
```

| Field         | Type    | What it contains                             |
| ------------- | ------- | -------------------------------------------- |
| `chunk_id`    | string  | Unique ID of the retrieved chunk             |
| `source_file` | string  | Source document/file                         |
| `page_start`  | integer | First page of the chunk                      |
| `page_end`    | integer | Last page of the chunk                       |
| `element_ids` | array   | Source element IDs associated with the chunk |

---

# What the frontend will normally use

For displaying a recommendation, the most useful fields are:

```javascript
recommendation.standard_number
recommendation.section_number
recommendation.section_title
recommendation.text
recommendation.score
recommendation.rank
recommendation.source
```

For example:

```javascript
const recommendation = result.recommendations[0];

console.log(recommendation.standard_number);
console.log(recommendation.text);
console.log(recommendation.score);
```

---

# Example frontend request

```javascript
const response = await fetch("/get_result", {
    method: "POST",
    headers: {
        "Content-Type": "application/json"
    },
    body: JSON.stringify({
        user_input: "Steel pipes for high pressure water supply"
    })
});

const result = await response.json();
```

After this:

```javascript
result
```

contains the complete backend response, and:

```javascript
result.recommendations
```

contains the retrieved results.

---

# Error responses

## `400 Bad Request`

The request was invalid.

Example:

```json
{
  "status": "error",
  "message": "user_input is required."
}
```

This usually means the frontend sent no input or an empty input.

---

## `503 Service Unavailable`

The retrieval system is unavailable or the persisted retrieval index cannot be used.

Example:

```json
{
  "status": "error",
  "message": "Retrieval service unavailable."
}
```

---

# Simple request → response flow

```text
FRONTEND
    |
    |  POST /get_result
    |  {"user_input": "..."}
    ↓
FLASK BACKEND
    |
    |  sends input to RetrievalService
    ↓
RETRIEVAL SYSTEM
    |
    |  returns up to 5 results
    ↓
FLASK BACKEND
    |
    |  converts results to JSON
    ↓
FRONTEND
    |
    └── result.recommendations
```

## In short

**Frontend sends:**

```json
{
  "user_input": "..."
}
```

**Backend returns:**

```text
status
user_input
recommendations[]
warnings
```

**The important part is `recommendations[]`:**

```text
standard_number
title
section_number
section_title
text
score
rank
source
```

That is the data the frontend receives from the retrieval backend.
