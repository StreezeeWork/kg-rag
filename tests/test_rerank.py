"""Тесты реранкера с фейковым CrossEncoder (без скачивания моделей)."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from kbrag.config import RerankConfig
from kbrag.retrieval import CrossEncoderReranker, Reranker, make_reranker
from kbrag.types import Chunk, ScoredChunk


class FakeCrossEncoder:
    """Подмена sentence_transformers.CrossEncoder.

    Релевантность = сколько раз слово запроса встречается в пассаже: результат
    предсказуем, и по нему легко проверить новый порядок.
    """

    def __init__(self, model_name: str, max_length: int | None = None, device: str | None = None) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.device = device
        self.predict_calls: list[dict] = []

    def predict(self, pairs, batch_size=32, show_progress_bar=None):
        pairs = list(pairs)
        self.predict_calls.append({"pairs": pairs, "batch_size": batch_size})
        return np.array([float(passage.count(query)) for query, passage in pairs])


class ColumnCrossEncoder(FakeCrossEncoder):
    """Модель, которая возвращает score столбцом формы (n, 1)."""

    def predict(self, pairs, batch_size=32, show_progress_bar=None):
        return super().predict(pairs, batch_size, show_progress_bar).reshape(-1, 1)


class MultiLabelCrossEncoder(FakeCrossEncoder):
    """Классификатор на 3 класса — для реранкинга не подходит."""

    def predict(self, pairs, batch_size=32, show_progress_bar=None):
        return np.zeros((len(list(pairs)), 3))


@pytest.fixture()
def fake_st(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    module = types.ModuleType("sentence_transformers")
    module.CrossEncoder = FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return module


def candidate(chunk_id: str, text: str, score: float, title: str = "") -> ScoredChunk:
    chunk = Chunk(
        chunk_id=chunk_id, doc_id=chunk_id.split("::")[0], text=text, start=0, end=len(text), title=title
    )
    return ScoredChunk(
        chunk=chunk,
        score=score,
        source="hybrid",
        details={"bm25": score * 10, "bm25_rank": 1, "hybrid": score},
    )


@pytest.fixture()
def candidates() -> list[ScoredChunk]:
    # Порядок поиска: c1, c3, c4, c2. Реранкер должен поставить c2 (3 «кот») первым.
    return [
        candidate("c1::000", "кот", 0.9),
        candidate("c3::000", "собака", 0.8),
        candidate("c4::000", "кот кот", 0.7),
        candidate("c2::000", "кот кот кот", 0.6),
    ]


def test_rerank_sorts_by_new_score(fake_st: types.ModuleType, candidates: list[ScoredChunk]) -> None:
    reranker = CrossEncoderReranker("fake-model", device="cpu")
    results = reranker.rerank("кот", candidates, top_k=4)
    assert [r.chunk.chunk_id for r in results] == ["c2::000", "c4::000", "c1::000", "c3::000"]
    assert [r.score for r in results] == [3.0, 2.0, 1.0, 0.0]
    assert all(r.source == "rerank" for r in results)


def test_rerank_respects_top_k(fake_st: types.ModuleType, candidates: list[ScoredChunk]) -> None:
    reranker = CrossEncoderReranker("fake-model", device="cpu")
    assert [r.chunk.chunk_id for r in reranker.rerank("кот", candidates, top_k=2)] == ["c2::000", "c4::000"]
    assert len(reranker.rerank("кот", candidates, top_k=10)) == 4


def test_rerank_details_keep_history(fake_st: types.ModuleType, candidates: list[ScoredChunk]) -> None:
    reranker = CrossEncoderReranker("fake-model", device="cpu")
    top = reranker.rerank("кот", candidates, top_k=1)[0]
    assert top.details == {"bm25": 6.0, "bm25_rank": 1, "hybrid": 0.6, "rerank": 3.0, "pre_rerank": 0.6}


def test_rerank_does_not_mutate_candidates(fake_st: types.ModuleType, candidates: list[ScoredChunk]) -> None:
    before = [(c.chunk.chunk_id, c.score, c.source, dict(c.details)) for c in candidates]
    CrossEncoderReranker("fake-model", device="cpu").rerank("кот", candidates, top_k=2)
    assert [(c.chunk.chunk_id, c.score, c.source, c.details) for c in candidates] == before


def test_rerank_tie_break_by_chunk_id(fake_st: types.ModuleType) -> None:
    tied = [candidate("b::000", "кот", 0.9), candidate("a::000", "кот", 0.1)]
    results = CrossEncoderReranker("fake-model", device="cpu").rerank("кот", tied, top_k=2)
    assert [r.chunk.chunk_id for r in results] == ["a::000", "b::000"]


@pytest.mark.parametrize("top_k", [0, -1])
def test_rerank_empty_inputs_skip_model(
    fake_st: types.ModuleType, candidates: list[ScoredChunk], top_k: int
) -> None:
    reranker = CrossEncoderReranker("fake-model", device="cpu")
    assert reranker.rerank("кот", [], top_k=3) == []
    assert reranker.rerank("кот", candidates, top_k=top_k) == []
    assert reranker.model.predict_calls == []


def test_rerank_passes_model_settings(fake_st: types.ModuleType, candidates: list[ScoredChunk]) -> None:
    reranker = CrossEncoderReranker("my-cross-encoder", batch_size=7, max_length=128, device="cpu")
    assert isinstance(reranker, Reranker)
    assert (reranker.model.model_name, reranker.model.max_length, reranker.model.device) == (
        "my-cross-encoder",
        128,
        "cpu",
    )
    reranker.rerank("кот", candidates, top_k=1)
    assert reranker.model.predict_calls[0]["batch_size"] == 7


def test_rerank_uses_header_text(fake_st: types.ModuleType) -> None:
    item = candidate("a::000", "Срок — 5 дней.", 1.0, title="Отпуск")
    with_header = CrossEncoderReranker("fake-model", device="cpu", with_header=True)
    with_header.rerank("срок", [item], top_k=1)
    assert with_header.model.predict_calls[0]["pairs"] == [("срок", "Отпуск\nСрок — 5 дней.")]

    without_header = CrossEncoderReranker("fake-model", device="cpu", with_header=False)
    without_header.rerank("срок", [item], top_k=1)
    assert without_header.model.predict_calls[0]["pairs"] == [("срок", "Срок — 5 дней.")]


def test_rerank_accepts_column_output(fake_st: types.ModuleType, candidates: list[ScoredChunk]) -> None:
    fake_st.CrossEncoder = ColumnCrossEncoder
    results = CrossEncoderReranker("fake-model", device="cpu").rerank("кот", candidates, top_k=2)
    assert [r.score for r in results] == [3.0, 2.0]


def test_rerank_rejects_multi_label_model(fake_st: types.ModuleType, candidates: list[ScoredChunk]) -> None:
    fake_st.CrossEncoder = MultiLabelCrossEncoder
    with pytest.raises(ValueError, match="одним выходом"):
        CrossEncoderReranker("fake-model", device="cpu").rerank("кот", candidates, top_k=2)


def test_auto_device_resolved_lazily(fake_st: types.ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert CrossEncoderReranker("fake-model").model.device == "cpu"


def test_missing_sentence_transformers_gives_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ImportError, match=r"pip install -e \.\[neural\]"):
        CrossEncoderReranker("fake-model", device="cpu")


def test_make_reranker_disabled_needs_no_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    assert make_reranker(RerankConfig(enabled=False)) is None


def test_make_reranker_enabled(fake_st: types.ModuleType) -> None:
    cfg = RerankConfig(enabled=True, model="my-cross-encoder", batch_size=4, max_length=256, device="cpu")
    reranker = make_reranker(cfg)
    assert isinstance(reranker, CrossEncoderReranker)
    assert (reranker.model_name, reranker.batch_size, reranker.max_length, reranker.device) == (
        "my-cross-encoder",
        4,
        256,
        "cpu",
    )
    assert reranker.name == "rerank"
