"""Тесты разбора ссылок [n] в ответе."""

from __future__ import annotations

import re

import pytest

from kbrag.generation.citations import parse_citations
from kbrag.types import Chunk, Citation, Document, ScoredChunk

_PARAGRAPH_RE = re.compile(r"[^\n]+(?:\n[^\n]+)*")


def make_context(docs: list[Document], doc_id: str, needle: str) -> ScoredChunk:
    """Фрагмент = абзац мини-документа, содержащий ``needle`` (чанкер здесь не нужен)."""
    doc = next(d for d in docs if d.doc_id == doc_id)
    for i, match in enumerate(_PARAGRAPH_RE.finditer(doc.text)):
        if needle in match.group():
            chunk = Chunk(
                chunk_id=f"{doc_id}::{i:03d}", doc_id=doc_id, text=match.group(),
                start=match.start(), end=match.end(), title=doc.title, metadata=dict(doc.metadata),
            )
            return ScoredChunk(chunk=chunk, score=1.0, source="bm25")
    raise AssertionError(f"абзац с {needle!r} не найден в {doc_id}")


@pytest.fixture()
def contexts(mini_docs: list[Document]) -> list[ScoredChunk]:
    return [
        make_context(mini_docs, "hr-vacation", "28 календарных"),  # [1]
        make_context(mini_docs, "hr-remote-2025", "1500 рублей"),  # [2]
        make_context(mini_docs, "it-vpn", "клиент «Шлюз»"),  # [3]
    ]


def indices(citations: list[Citation]) -> list[int]:
    return [c.index for c in citations]


def test_single_citation_is_bound_to_its_fragment(contexts: list[ScoredChunk]) -> None:
    citations, invalid = parse_citations("Отпуск длится 28 календарных дней [1].", contexts)

    assert invalid == 0
    assert citations == [
        Citation(index=1, chunk_id="hr-vacation::002", doc_id="hr-vacation", title="Ежегодный оплачиваемый отпуск")
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Ответ [1, 2].", [1, 2]),
        ("Ответ [1,2].", [1, 2]),
        ("Ответ [3; 1].", [3, 1]),
        ("Ответ [ 2 ].", [2]),
        ("Ответ [1][3].", [1, 3]),
        ("Первое [2]. Второе [1]. Третье [2][1].", [2, 1]),  # повторы — один раз, в порядке появления
    ],
)
def test_citation_formats(contexts: list[ScoredChunk], text: str, expected: list[int]) -> None:
    citations, invalid = parse_citations(text, contexts)
    assert indices(citations) == expected
    assert invalid == 0


def test_chunk_fields_follow_context_order(contexts: list[ScoredChunk]) -> None:
    citations, _ = parse_citations("[3] и [2]", contexts)
    assert [c.chunk_id for c in citations] == [contexts[2].chunk.chunk_id, contexts[1].chunk.chunk_id]
    assert [c.doc_id for c in citations] == ["it-vpn", "hr-remote-2025"]


def test_out_of_range_numbers_are_counted_as_invalid(contexts: list[ScoredChunk]) -> None:
    citations, invalid = parse_citations("Факт [1]. Выдумка [4]. Ещё [0]. И [2, 9].", contexts)
    assert indices(citations) == [1, 2]
    assert invalid == 3  # 4, 0 и 9


def test_repeated_invalid_number_is_counted_once(contexts: list[ScoredChunk]) -> None:
    citations, invalid = parse_citations("Выдумка [7]. Снова [7].", contexts)
    assert citations == []
    assert invalid == 1


def test_text_without_citations(contexts: list[ScoredChunk]) -> None:
    assert parse_citations("В базе знаний нет ответа на этот вопрос.", contexts) == ([], 0)
    assert parse_citations("", contexts) == ([], 0)


def test_other_brackets_are_not_citations(contexts: list[ScoredChunk]) -> None:
    text = "Смотри [раздел 2], [a], [1-2] и [ ] — ссылок нет; а вот [2] есть."
    citations, invalid = parse_citations(text, contexts)
    assert indices(citations) == [2]
    assert invalid == 0


def test_any_number_is_invalid_without_contexts() -> None:
    assert parse_citations("Ответ [1].", []) == ([], 1)
