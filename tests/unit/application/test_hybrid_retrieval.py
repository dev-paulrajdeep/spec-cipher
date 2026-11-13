from __future__ import annotations

import pytest

from speccipher.application.use_cases.hybrid_retrieval import (
    HybridRetrieval,
    fuse_ranked_candidates,
)
from speccipher.domain.candidates import Candidate
from speccipher.domain.models import DocumentChunk


def _candidate(
    chunk_id: str, *, dense: float | None = None, lexical: float | None = None
) -> Candidate:
    return Candidate(
        chunk=DocumentChunk(
            chunk_id=chunk_id,
            standard_id=None,
            logical_doc_id="doc-1",
            source_file_id="source-1",
            absolute_page_start=1,
            absolute_page_end=1,
            text=f"text for {chunk_id}",
        ),
        dense_score=dense,
        lexical_score=lexical,
    )


def test_rrf_merges_rankings_deterministically_and_keeps_score_dimensions() -> None:
    dense = [_candidate("b", dense=0.91), _candidate("a", dense=0.82), _candidate("c", dense=0.73)]
    lexical = [
        _candidate("a", lexical=-0.2),
        _candidate("b", lexical=-0.1),
        _candidate("d", lexical=-0.4),
    ]

    result = fuse_ranked_candidates(dense, lexical)

    assert [candidate.chunk.chunk_id for candidate in result] == ["a", "b", "c", "d"]
    assert result[0].dense_score == 0.82
    assert result[0].lexical_score == -0.2
    assert result[0].rrf_score == pytest.approx(1 / 61 + 1 / 62)
    assert result[2].dense_score == 0.73
    assert result[2].lexical_score is None
    assert result[3].dense_score is None
    assert result[3].lexical_score == -0.4
    assert result[3].rrf_score == pytest.approx(1 / 63)


@pytest.mark.parametrize(
    ("dense", "lexical"),
    [([], []), ([], [_candidate("lex", lexical=-0.2)]), ([_candidate("vec", dense=0.8)], [])],
)
def test_rrf_handles_empty_or_single_search_results(dense, lexical) -> None:
    result = fuse_ranked_candidates(dense, lexical)

    assert len(result) == len(dense) + len(lexical)
    if not dense and not lexical:
        assert result == ()


def test_duplicate_candidate_within_one_ranked_list_counts_only_its_first_rank() -> None:
    result = fuse_ranked_candidates(
        [
            _candidate("same", dense=0.8),
            _candidate("same", dense=0.7),
            _candidate("later", dense=0.6),
        ],
        [],
    )

    assert len(result) == 2
    by_id = {candidate.chunk.chunk_id: candidate for candidate in result}
    assert by_id["same"].rrf_score == pytest.approx(1 / 61)
    assert by_id["same"].dense_score == 0.8
    assert by_id["later"].rrf_score == pytest.approx(1 / 63)


class _VectorIndex:
    def __init__(self, results: list[Candidate]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, limit: int) -> list[Candidate]:
        self.calls.append((query, limit))
        return self.results


class _LexicalIndex(_VectorIndex):
    pass


def test_hybrid_use_case_queries_each_port_and_fuses_results() -> None:
    vector = _VectorIndex([_candidate("dense", dense=0.9)])
    lexical = _LexicalIndex([_candidate("lexical", lexical=-0.2)])
    use_case = HybridRetrieval(vector, lexical)

    result = use_case.retrieve("  concrete  ", limit=25)

    assert vector.calls == [("  concrete  ", 25)]
    assert lexical.calls == [("  concrete  ", 25)]
    assert [candidate.chunk.chunk_id for candidate in result] == ["dense", "lexical"]


@pytest.mark.parametrize("query", ["", "   ", None])
def test_hybrid_use_case_rejects_empty_query(query) -> None:
    use_case = HybridRetrieval(_VectorIndex([]), _LexicalIndex([]))

    with pytest.raises(ValueError, match="non-empty string"):
        use_case.retrieve(query)
