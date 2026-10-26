"""Тесты BM25 и общих утилит retrieval.base (top_k_indices, контракт Retriever)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from kbrag.retrieval import BM25Retriever, top_k_indices
from kbrag.text import stem
from kbrag.types import Chunk, Document


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


def plain_chunk(chunk_id: str, text: str, title: str = "") -> Chunk:
    return Chunk(
        chunk_id=chunk_id, doc_id=chunk_id.split("::")[0], text=text, start=0, end=len(text), title=title
    )


@pytest.fixture()
def mini_chunks(mini_docs: list[Document]) -> list[Chunk]:
    return paragraph_chunks(mini_docs)


@pytest.fixture()
def mini_bm25(mini_chunks: list[Chunk]) -> BM25Retriever:
    retriever = BM25Retriever()
    retriever.index(mini_chunks)
    return retriever


@pytest.fixture()
def tiny_chunks() -> list[Chunk]:
    # Длины после анализа: 2, 3, 2 термина → avgdl = 7/3.
    return [
        plain_chunk("a::000", "кот спит"),
        plain_chunk("b::000", "кот ест рыбу"),
        plain_chunk("c::000", "собака лает"),
    ]


def test_paragraph_chunks_respect_offsets(mini_docs: list[Document], mini_chunks: list[Chunk]) -> None:
    # Самопроверка тестового помощника: смещения обязаны указывать на текст фрагмента.
    docs_by_id = {d.doc_id: d for d in mini_docs}
    assert len(mini_chunks) == 8
    for chunk in mini_chunks:
        assert docs_by_id[chunk.doc_id].text[chunk.start : chunk.end] == chunk.text


# --- top_k_indices ---


def test_top_k_indices_sorted_descending() -> None:
    assert top_k_indices(np.array([0.1, 0.9, 0.5, 0.7]), 3).tolist() == [1, 3, 2]


def test_top_k_indices_ties_prefer_smaller_index() -> None:
    assert top_k_indices(np.array([0.5, 1.0, 0.5, 1.0, 0.5]), 4).tolist() == [1, 3, 0, 2]


@pytest.mark.parametrize("k", [0, -1])
def test_top_k_indices_non_positive_k(k: int) -> None:
    assert top_k_indices(np.array([1.0, 2.0]), k).tolist() == []


def test_top_k_indices_k_larger_than_size_and_empty() -> None:
    assert top_k_indices(np.array([2.0, 3.0]), 10).tolist() == [1, 0]
    assert top_k_indices(np.array([]), 3).tolist() == []


# --- формула BM25 ---


def test_idf_matches_formula(tiny_chunks: list[Chunk]) -> None:
    retriever = BM25Retriever()
    retriever.index(tiny_chunks)
    # N = 3; «кот» встречается в 2 фрагментах, «собака» — в 1.
    assert retriever.idf[stem("кот")] == pytest.approx(math.log(1 + (3 - 2 + 0.5) / (2 + 0.5)))
    assert retriever.idf[stem("собака")] == pytest.approx(math.log(1 + (3 - 1 + 0.5) / (1 + 0.5)))
    assert retriever.doc_len.tolist() == [2, 3, 2]
    assert retriever.avgdl == pytest.approx(7 / 3)


def test_score_matches_formula_by_hand(tiny_chunks: list[Chunk]) -> None:
    k1, b = 1.5, 0.75
    retriever = BM25Retriever(k1=k1, b=b)
    retriever.index(tiny_chunks)
    results = retriever.search("собака", k=5)

    avgdl = 7 / 3
    idf = math.log(1 + (3 - 1 + 0.5) / (1 + 0.5))
    tf, doc_len = 1, 2
    expected = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / avgdl))
    assert [r.chunk.chunk_id for r in results] == ["c::000"]
    assert results[0].score == pytest.approx(expected)


def test_length_normalization_controlled_by_b(tiny_chunks: list[Chunk]) -> None:
    # «кот» встречается по одному разу во фрагментах длиной 2 и 3.
    with_norm = BM25Retriever(b=0.75)
    with_norm.index(tiny_chunks)
    short, long = with_norm.search("кот", k=5)
    assert (short.chunk.chunk_id, long.chunk.chunk_id) == ("a::000", "b::000")
    assert short.score > long.score

    without_norm = BM25Retriever(b=0.0)
    without_norm.index(tiny_chunks)
    first, second = without_norm.search("кот", k=5)
    assert first.score == pytest.approx(second.score)


def test_k1_zero_ignores_term_frequency() -> None:
    chunks = [
        plain_chunk("a::000", "кот кот кот"),
        plain_chunk("b::000", "кот спит"),
        plain_chunk("c::000", "пёс"),
    ]
    saturated = BM25Retriever(k1=1.5, b=0.0)
    saturated.index(chunks)
    assert saturated.search("кот", 1)[0].chunk.chunk_id == "a::000"

    binary = BM25Retriever(k1=0.0, b=0.0)
    binary.index(chunks)
    scores = [r.score for r in binary.search("кот", 5)]
    assert scores == pytest.approx([binary.idf[stem("кот")]] * 2)


def test_repeated_query_term_counts_twice(tiny_chunks: list[Chunk]) -> None:
    retriever = BM25Retriever()
    retriever.index(tiny_chunks)
    single = retriever.score_all("кот")
    double = retriever.score_all("кот кот")
    np.testing.assert_allclose(double, 2 * single)


# --- поведение на мини-корпусе ---


def test_word_forms_are_matched(mini_bm25: BM25Retriever) -> None:
    assert stem("отпуском") == stem("отпуск")
    results = mini_bm25.search("отпуском", k=5)
    assert {r.chunk.doc_id for r in results} == {"hr-vacation"}
    assert len(results) == 2


def test_yo_and_case_are_normalized(mini_bm25: BM25Retriever) -> None:
    upper = mini_bm25.search("ПОДАЁТСЯ", k=3)
    plain = mini_bm25.search("подается", k=3)
    assert [r.chunk.chunk_id for r in upper] == [r.chunk.chunk_id for r in plain] == ["hr-vacation::001"]


@pytest.mark.parametrize("query", ["квантовый звездолёт", "и в на для", "", "!!! ???"])
def test_query_without_known_terms_returns_empty(mini_bm25: BM25Retriever, query: str) -> None:
    assert mini_bm25.search(query, k=5) == []


def test_only_positive_scores_are_returned(mini_bm25: BM25Retriever) -> None:
    results = mini_bm25.search("пароль", k=10)
    assert [r.chunk.chunk_id for r in results] == ["it-vpn::001"]
    assert all(r.score > 0 for r in results)


def test_results_follow_retriever_contract(mini_bm25: BM25Retriever) -> None:
    results = mini_bm25.search("Сколько дней длится отпуск?", k=3)
    assert 0 < len(results) <= 3
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    for item in results:
        assert item.source == "bm25"
        assert item.details == {"bm25": item.score}


@pytest.mark.parametrize("k", [0, -3])
def test_non_positive_k_returns_empty(mini_bm25: BM25Retriever, k: int) -> None:
    assert mini_bm25.search("отпуск", k=k) == []


def test_k_limits_number_of_results(mini_bm25: BM25Retriever) -> None:
    assert len(mini_bm25.search("дней", k=1)) == 1


def test_gold_questions_find_evidence_document(
    mini_bm25: BM25Retriever, mini_gold_records: list[dict]
) -> None:
    by_id = {item["id"]: item for item in mini_gold_records}
    for qid in ("t-001", "t-002", "t-004"):
        top = mini_bm25.search(by_id[qid]["question"], k=1)
        assert top[0].chunk.doc_id == by_id[qid]["evidence"][0]["doc_id"], qid
    # Вопрос без ответа в базе: ни одного общего терма → пустая выдача.
    assert mini_bm25.search(by_id["t-005"]["question"], k=5) == []


def test_with_header_indexes_document_title(mini_chunks: list[Chunk]) -> None:
    # Слово «корпоративному» есть только в заголовке документа it-vpn.
    with_header = BM25Retriever(with_header=True)
    with_header.index(mini_chunks)
    assert {r.chunk.chunk_id for r in with_header.search("корпоративный", k=5)} == {
        "it-vpn::000",
        "it-vpn::001",
    }

    without_header = BM25Retriever(with_header=False)
    without_header.index(mini_chunks)
    assert without_header.search("корпоративный", k=5) == []


def test_ties_are_broken_by_index_order() -> None:
    chunks = [
        plain_chunk("a::000", "отпуск"),
        plain_chunk("b::000", "отпуск"),
        plain_chunk("c::000", "пароль"),
    ]
    retriever = BM25Retriever()
    retriever.index(chunks)
    assert [r.chunk.chunk_id for r in retriever.search("отпуск", 5)] == ["a::000", "b::000"]


def test_search_batch_equals_individual_searches(mini_bm25: BM25Retriever) -> None:
    queries = ["отпуск", "VPN клиент", "неизвестное слово"]
    assert mini_bm25.search_batch(queries, k=3) == [mini_bm25.search(q, k=3) for q in queries]


# --- граничные случаи индекса ---


def test_search_before_index_raises() -> None:
    with pytest.raises(RuntimeError, match="index"):
        BM25Retriever().search("отпуск", 3)


def test_empty_index_returns_empty() -> None:
    retriever = BM25Retriever()
    retriever.index([])
    assert retriever.chunks == []
    assert retriever.search("отпуск", 3) == []


def test_stopword_only_chunks_do_not_break_index() -> None:
    retriever = BM25Retriever()
    retriever.index([plain_chunk("a::000", "и в на"), plain_chunk("b::000", "и")])
    assert retriever.avgdl == 0.0
    assert retriever.search("отпуск", 3) == []


def test_reindex_replaces_previous_index(tiny_chunks: list[Chunk]) -> None:
    retriever = BM25Retriever()
    retriever.index(tiny_chunks)
    retriever.index([plain_chunk("z::000", "пароль")])
    assert retriever.search("кот", 3) == []
    assert [r.chunk.chunk_id for r in retriever.search("пароль", 3)] == ["z::000"]


def test_duplicate_chunk_ids_rejected() -> None:
    with pytest.raises(ValueError, match="chunk_id"):
        BM25Retriever().index([plain_chunk("a::000", "кот"), plain_chunk("a::000", "пёс")])


@pytest.mark.parametrize("params", [{"k1": -0.1}, {"b": 1.5}, {"b": -0.1}])
def test_invalid_parameters_rejected(params: dict) -> None:
    with pytest.raises(ValueError):
        BM25Retriever(**params)
