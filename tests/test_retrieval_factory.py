"""Тесты сборки поисковика из AppConfig (build_retriever)."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from kbrag.config import AppConfig
from kbrag.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    LSAEmbedder,
    SentenceTransformerEmbedder,
    build_retriever,
)
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


def config(**retrieval) -> AppConfig:
    return AppConfig.model_validate({"retrieval": retrieval})


def test_default_config_builds_rrf_hybrid() -> None:
    cfg = AppConfig()
    retriever = build_retriever(cfg)
    assert isinstance(retriever, HybridRetriever)
    bm25, dense = retriever.retrievers
    assert isinstance(bm25, BM25Retriever)
    assert isinstance(dense, DenseRetriever)
    assert isinstance(dense.embedder, LSAEmbedder)
    assert (retriever.fusion, retriever.rrf_k, retriever.candidates_k) == (
        "rrf",
        60,
        cfg.retrieval.candidates_k,
    )
    assert retriever.weights == {"bm25": 0.5, "dense": 0.5}


def test_bm25_mode_uses_parameters_from_config() -> None:
    cfg = AppConfig.model_validate(
        {"retrieval": {"mode": "bm25", "bm25": {"k1": 0.9, "b": 0.4}}, "chunking": {"with_header": False}}
    )
    retriever = build_retriever(cfg)
    assert isinstance(retriever, BM25Retriever)
    assert (retriever.k1, retriever.b, retriever.with_header) == (0.9, 0.4, False)


def test_dense_mode_builds_lsa_with_configured_dim() -> None:
    retriever = build_retriever(config(mode="dense", dense={"embedder": "lsa", "lsa_dim": 32}))
    assert isinstance(retriever, DenseRetriever)
    assert isinstance(retriever.embedder, LSAEmbedder)
    assert retriever.embedder.dim == 32
    assert retriever.with_header is True


def test_hybrid_weighted_settings_are_passed() -> None:
    cfg = config(
        mode="hybrid",
        candidates_k=12,
        hybrid={"fusion": "weighted", "rrf_k": 5, "weights": {"bm25": 0.3, "dense": 0.7}},
    )
    retriever = build_retriever(cfg)
    assert isinstance(retriever, HybridRetriever)
    assert (retriever.fusion, retriever.rrf_k, retriever.candidates_k) == ("weighted", 5, 12)
    assert retriever.weights == {"bm25": 0.3, "dense": 0.7}
    # Словарь весов скопирован: правка ретривера не меняет конфиг (и наоборот).
    retriever.weights["bm25"] = 1.0
    assert cfg.retrieval.hybrid.weights["bm25"] == 0.3


def test_hybrid_with_missing_weight_fails_fast() -> None:
    with pytest.raises(ValueError, match="dense"):
        build_retriever(config(mode="hybrid", hybrid={"fusion": "weighted", "weights": {"bm25": 1.0}}))


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_every_mode_retrieves_from_mini_corpus(
    mode: str, mini_docs: list[Document], mini_gold_records: list[dict]
) -> None:
    retriever = build_retriever(config(mode=mode))
    retriever.index(paragraph_chunks(mini_docs))
    item = next(r for r in mini_gold_records if r["id"] == "t-004")
    results = retriever.search(item["question"], k=3)
    assert results[0].chunk.doc_id == item["evidence"][0]["doc_id"]
    assert len(results) <= 3
    assert all(r.source == retriever.name for r in results)


class FakeSentenceTransformer:
    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device

    def encode(
        self, texts, batch_size=32, normalize_embeddings=False, convert_to_numpy=True, show_progress_bar=None
    ):
        return np.array([[len(t), 1.0] for t in texts], dtype=np.float64)


def test_sentence_transformers_embedder_gets_cache_dir(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    cfg = AppConfig.model_validate(
        {
            "cache_dir": str(tmp_path),
            "retrieval": {
                "mode": "dense",
                "dense": {"embedder": "sentence-transformers", "model": "m-e5", "device": "cpu"},
            },
        }
    )
    retriever = build_retriever(cfg)
    assert isinstance(retriever.embedder, SentenceTransformerEmbedder)
    assert retriever.embedder.cache_dir == str(tmp_path)
    assert retriever.embedder.model.model_name == "m-e5"


def test_unknown_mode_rejected() -> None:
    # Pydantic не пропустит неизвестный режим из YAML, но присваивание атрибута он не проверяет —
    # фабрика должна упасть с понятной ошибкой, а не вернуть None.
    cfg = AppConfig()
    cfg.retrieval.mode = "magic"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="magic"):
        build_retriever(cfg)
