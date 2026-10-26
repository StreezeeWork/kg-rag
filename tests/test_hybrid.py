"""Тесты слияния выдач (RRF, weighted) и HybridRetriever."""

from __future__ import annotations

import pytest

from kbrag.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    LSAEmbedder,
    Retriever,
    rrf_fuse,
    weighted_fuse,
)
from kbrag.types import Chunk, Document, ScoredChunk


def paragraph_chunks(docs: list[Document]) -> list[Chunk]:
    """Фрагменты «по абзацам» без чанкера из ingest: строки-заголовки задают section."""
    chunks: list[Chunk] = []
    for doc in docs:
        section, pos, number = "", 0, 0
        for para in (p.strip() for p in doc.text.split("\n\n")):
            if not para:
                continue
            start = doc.text.index(para, pos)
            pos = start + len(para)
            if para.startswith("#"):
                section = para.lstrip("#").strip()
                continue
            chunks.append(
                Chunk(f"{doc.doc_id}::{number:03d}", doc.doc_id, para, start, pos, doc.title, section)
            )
            number += 1
    return chunks


def chunk(chunk_id: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id=chunk_id.split("::")[0], text=f"текст {chunk_id}", start=0, end=10)


def scored(chunk_id: str, score: float, source: str) -> ScoredChunk:
    return ScoredChunk(chunk=chunk(chunk_id), score=score, source=source, details={source: score})


def ids(results: list[ScoredChunk]) -> list[str]:
    return [r.chunk.chunk_id for r in results]


A, B, C, D = "a::000", "b::000", "c::000", "d::000"


@pytest.fixture()
def two_lists() -> dict[str, list[ScoredChunk]]:
    return {
        "bm25": [scored(A, 9.0, "bm25"), scored(B, 5.0, "bm25"), scored(C, 1.0, "bm25")],
        "dense": [scored(B, 0.9, "dense"), scored(D, 0.4, "dense")],
    }


# --- RRF ---


def test_rrf_arithmetic(two_lists: dict[str, list[ScoredChunk]]) -> None:
    fused = rrf_fuse(two_lists, k=60)
    expected = {A: 1 / 61, B: 1 / 62 + 1 / 61, C: 1 / 63, D: 1 / 62}
    assert ids(fused) == [B, A, D, C]
    for item in fused:
        assert item.score == pytest.approx(expected[item.chunk.chunk_id])
        assert item.source == "hybrid"
        assert item.details["hybrid"] == item.score


def test_rrf_details_keep_component_scores_and_ranks(two_lists: dict[str, list[ScoredChunk]]) -> None:
    by_id = {r.chunk.chunk_id: r for r in rrf_fuse(two_lists)}
    assert by_id[B].details == {
        "bm25": 5.0,
        "bm25_rank": 2,
        "dense": 0.9,
        "dense_rank": 1,
        "hybrid": pytest.approx(1 / 62 + 1 / 61),
    }
    # Фрагмент, которого нет в выдаче dense, не получает ключей dense.
    assert "dense" not in by_id[A].details
    assert by_id[D].details["dense_rank"] == 2


def test_rrf_weights_and_default_weight(two_lists: dict[str, list[ScoredChunk]]) -> None:
    fused = rrf_fuse(two_lists, k=60, weights={"bm25": 2.0})  # dense → вес по умолчанию 1.0
    expected = {A: 2 / 61, B: 2 / 62 + 1 / 61, C: 2 / 63, D: 1 / 62}
    assert ids(fused) == [B, A, C, D]
    for item in fused:
        assert item.score == pytest.approx(expected[item.chunk.chunk_id])


def test_rrf_k_parameter(two_lists: dict[str, list[ScoredChunk]]) -> None:
    by_id = {r.chunk.chunk_id: r.score for r in rrf_fuse(two_lists, k=0)}
    assert by_id[A] == pytest.approx(1.0)
    assert by_id[B] == pytest.approx(1 / 2 + 1)
    with pytest.raises(ValueError):
        rrf_fuse(two_lists, k=-1)


def test_rrf_tie_break_by_chunk_id_regardless_of_order() -> None:
    first = rrf_fuse({"bm25": [scored("z::000", 3.0, "bm25")], "dense": [scored(A, 0.5, "dense")]})
    second = rrf_fuse({"dense": [scored(A, 0.5, "dense")], "bm25": [scored("z::000", 3.0, "bm25")]})
    assert ids(first) == ids(second) == [A, "z::000"]
    assert first[0].score == first[1].score


def test_rrf_counts_duplicate_once_at_best_rank() -> None:
    fused = rrf_fuse({"bm25": [scored(A, 5.0, "bm25"), scored(A, 4.0, "bm25"), scored(B, 3.0, "bm25")]})
    by_id = {r.chunk.chunk_id: r for r in fused}
    assert by_id[A].score == pytest.approx(1 / 61)
    assert by_id[A].details["bm25"] == 5.0
    assert by_id[B].details["bm25_rank"] == 2


def test_fusion_of_empty_inputs() -> None:
    assert rrf_fuse({}) == []
    assert rrf_fuse({"bm25": [], "dense": []}) == []
    assert weighted_fuse({"bm25": []}, weights={"bm25": 1.0}) == []


def test_fusion_does_not_mutate_inputs(two_lists: dict[str, list[ScoredChunk]]) -> None:
    rrf_fuse(two_lists)
    weighted_fuse(two_lists, weights={"bm25": 0.5, "dense": 0.5})
    assert two_lists["bm25"][1].details == {"bm25": 5.0}
    assert two_lists["dense"][0].source == "dense"


# --- weighted ---


def test_weighted_arithmetic() -> None:
    lists = {
        "bm25": [scored(A, 10.0, "bm25"), scored(B, 5.0, "bm25"), scored(C, 0.0, "bm25")],  # → 1, 0.5, 0
        "dense": [scored(B, 0.9, "dense"), scored(D, 0.3, "dense")],  # → 1, 0
    }
    fused = weighted_fuse(lists, weights={"bm25": 0.7, "dense": 0.3})
    expected = {A: 0.7, B: 0.7 * 0.5 + 0.3 * 1.0, C: 0.0, D: 0.0}
    # C и D набрали по 0 — порядок между ними решает chunk_id.
    assert ids(fused) == [A, B, C, D]
    for item in fused:
        assert item.score == pytest.approx(expected[item.chunk.chunk_id])
    by_id = {r.chunk.chunk_id: r for r in fused}
    assert by_id[B].details == {
        "bm25": 5.0,
        "bm25_rank": 2,
        "dense": 0.9,
        "dense_rank": 1,
        "hybrid": pytest.approx(0.65),
    }


def test_weighted_single_or_equal_scores_normalize_to_one() -> None:
    single = weighted_fuse({"bm25": [scored(A, 3.0, "bm25")]}, weights={"bm25": 0.5})
    assert single[0].score == pytest.approx(0.5)
    equal = weighted_fuse({"bm25": [scored(B, 2.0, "bm25"), scored(A, 2.0, "bm25")]}, weights={"bm25": 1.0})
    assert [r.score for r in equal] == [1.0, 1.0]
    assert ids(equal) == [A, B]


def test_weighted_handles_negative_cosine() -> None:
    fused = weighted_fuse(
        {"dense": [scored(A, 0.2, "dense"), scored(B, -0.4, "dense")]}, weights={"dense": 1.0}
    )
    assert [r.score for r in fused] == pytest.approx([1.0, 0.0])


def test_weighted_float_tie_break_is_stable() -> None:
    lists = {
        "bm25": [scored(B, 1.0, "bm25"), scored(A, 0.5, "bm25"), scored(C, 0.0, "bm25")],  # → 1, 0.5, 0
        "dense": [scored(A, 0.9, "dense"), scored(C, 0.1, "dense")],  # → 1, 0
        "extra": [scored("z::000", 7.0, "extra")],  # единственный элемент → 1
    }
    fused = weighted_fuse(lists, weights={"bm25": 0.7, "dense": 0.3, "extra": 0.65})
    by_id = {r.chunk.chunk_id: r.score for r in fused}
    # A = 0.7·0.5 + 0.3·1 = 0.6499999999999999 во float, z = 0.65: математически равны.
    assert by_id[A] != by_id["z::000"]
    assert by_id[A] == pytest.approx(by_id["z::000"])
    # Равенство с точностью до погрешности float → решает chunk_id, а не последний бит.
    assert ids(fused) == [B, A, "z::000", C]


def test_weighted_requires_all_weights(two_lists: dict[str, list[ScoredChunk]]) -> None:
    with pytest.raises(ValueError, match="dense"):
        weighted_fuse(two_lists, weights={"bm25": 1.0})


# --- HybridRetriever с управляемыми компонентами ---


class StaticRetriever(Retriever):
    """Компонент с заранее заданным порядком выдачи — чтобы проверять только логику гибрида."""

    def __init__(self, name: str, ranking: list[str]) -> None:
        super().__init__()
        self.name = name
        self.ranking = ranking
        self.requested_k: list[int] = []

    def index(self, chunks: list[Chunk]) -> None:
        self._set_chunks(chunks)

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        self.requested_k.append(k)
        by_id = {c.chunk_id: c for c in self.chunks}
        return [
            ScoredChunk(by_id[cid], float(10 - i), self.name, {self.name: float(10 - i)})
            for i, cid in enumerate(self.ranking[:k])
        ]


@pytest.fixture()
def corpus() -> list[Chunk]:
    return [chunk(cid) for cid in (A, B, C, D)]


def make_hybrid(corpus: list[Chunk], **kwargs) -> tuple[HybridRetriever, StaticRetriever, StaticRetriever]:
    bm25 = StaticRetriever("bm25", [A, B, C])
    dense = StaticRetriever("dense", [B, D])
    hybrid = HybridRetriever([bm25, dense], **kwargs)
    hybrid.index(corpus)
    return hybrid, bm25, dense


def test_hybrid_indexes_all_components(corpus: list[Chunk]) -> None:
    hybrid, bm25, dense = make_hybrid(corpus)
    assert hybrid.chunks == bm25.chunks == dense.chunks == corpus


@pytest.mark.parametrize(("k", "candidates_k", "expected"), [(2, 30, 30), (50, 30, 50), (3, 3, 3)])
def test_hybrid_asks_components_for_candidates(
    corpus: list[Chunk], k: int, candidates_k: int, expected: int
) -> None:
    hybrid, bm25, dense = make_hybrid(corpus, candidates_k=candidates_k)
    results = hybrid.search("вопрос", k)
    assert bm25.requested_k == dense.requested_k == [expected]
    assert len(results) == min(k, 4)


def test_hybrid_rrf_matches_rrf_fuse(corpus: list[Chunk]) -> None:
    hybrid, _, _ = make_hybrid(corpus, rrf_k=10)
    results = hybrid.search("вопрос", 2)
    assert ids(results) == [B, A]  # B: 1/12 + 1/11, A: 1/11
    assert results[0].score == pytest.approx(1 / 12 + 1 / 11)
    assert all(r.source == "hybrid" for r in results)


def test_hybrid_weighted_defaults_to_equal_weights(corpus: list[Chunk]) -> None:
    hybrid, _, _ = make_hybrid(corpus, fusion="weighted")
    assert hybrid.weights == {"bm25": 1.0, "dense": 1.0}
    results = hybrid.search("вопрос", 4)
    # bm25: A=1, B=0.5, C=0; dense: B=1, D=0 → B=1.5, A=1, C=0, D=0.
    assert ids(results) == [B, A, C, D]
    assert [r.score for r in results] == pytest.approx([1.5, 1.0, 0.0, 0.0])


def test_hybrid_search_batch_equals_search(corpus: list[Chunk]) -> None:
    hybrid, _, _ = make_hybrid(corpus)
    queries = ["первый", "второй"]
    assert hybrid.search_batch(queries, 3) == [hybrid.search(q, 3) for q in queries]
    assert hybrid.search_batch(queries, 0) == [[], []]


def test_hybrid_non_positive_k_and_search_before_index(corpus: list[Chunk]) -> None:
    hybrid, _, _ = make_hybrid(corpus)
    assert hybrid.search("вопрос", 0) == []
    with pytest.raises(RuntimeError, match="index"):
        HybridRetriever([StaticRetriever("bm25", [])]).search("вопрос", 3)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"retrievers": []},
        {"retrievers": [StaticRetriever("bm25", []), StaticRetriever("bm25", [])]},
        {"retrievers": [StaticRetriever("bm25", [])], "fusion": "max"},
        {
            "retrievers": [StaticRetriever("bm25", []), StaticRetriever("dense", [])],
            "fusion": "weighted",
            "weights": {"bm25": 1.0},
        },
        {"retrievers": [StaticRetriever("bm25", [])], "candidates_k": 0},
    ],
)
def test_hybrid_rejects_invalid_configuration(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        HybridRetriever(**kwargs)


# --- HybridRetriever на реальных компонентах и мини-корпусе ---


@pytest.fixture()
def real_hybrid(mini_docs: list[Document]) -> HybridRetriever:
    hybrid = HybridRetriever([BM25Retriever(), DenseRetriever(LSAEmbedder())], candidates_k=10)
    hybrid.index(paragraph_chunks(mini_docs))
    return hybrid


def test_real_hybrid_finds_vpn_client_with_both_signals(real_hybrid: HybridRetriever) -> None:
    top = real_hybrid.search("Какой клиент нужен для VPN?", k=3)[0]
    assert top.chunk.chunk_id == "it-vpn::000"
    assert top.details["bm25_rank"] == top.details["dense_rank"] == 1
    assert top.details["bm25"] > 0 and 0 < top.details["dense"] <= 1.0 + 1e-6


def test_real_hybrid_unanswerable_has_no_bm25_signal(
    real_hybrid: HybridRetriever, mini_gold_records: list[dict]
) -> None:
    question = next(item["question"] for item in mini_gold_records if item["type"] == "unanswerable")
    results = real_hybrid.search(question, k=3)
    # BM25 не нашёл ни одного общего терма → в details есть только dense; по нему пайплайн решит об отказе.
    assert results
    assert all("bm25" not in r.details and "dense" in r.details for r in results)


def test_real_hybrid_unknown_query_returns_empty(real_hybrid: HybridRetriever) -> None:
    assert real_hybrid.search("zzzz qqqq", k=5) == []
