"""Интеграционные тесты с настоящими нейросетевыми моделями.

По умолчанию пропускаются: они скачивают веса (сотни мегабайт) и идут десятки секунд.
Запуск: ``KBRAG_RUN_NEURAL=1 pytest -m neural`` (нужен ``pip install -e .[neural]``).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from kbrag.retrieval import BM25Retriever, CrossEncoderReranker, DenseRetriever, SentenceTransformerEmbedder
from kbrag.types import Chunk, Document

pytestmark = [
    pytest.mark.neural,
    pytest.mark.skipif(
        os.environ.get("KBRAG_RUN_NEURAL") != "1",
        reason="нейросетевые тесты включаются переменной KBRAG_RUN_NEURAL=1 (скачивают модели)",
    ),
]

E5_MODEL = "intfloat/multilingual-e5-small"
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


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


@pytest.fixture(scope="module")
def e5_embedder(tmp_path_factory: pytest.TempPathFactory) -> SentenceTransformerEmbedder:
    pytest.importorskip("sentence_transformers")
    cache_dir = tmp_path_factory.mktemp("neural-cache")
    return SentenceTransformerEmbedder(E5_MODEL, device="cpu", cache_dir=str(cache_dir))


@pytest.fixture()
def mini_chunks(mini_docs: list[Document]) -> list[Chunk]:
    return paragraph_chunks(mini_docs)


def test_e5_vectors_are_normalized(e5_embedder: SentenceTransformerEmbedder) -> None:
    vectors = e5_embedder.encode_queries(["Сколько дней отпуска?", "Как подключиться к VPN?"])
    assert vectors.dtype == np.float32
    assert vectors.shape == (2, 384)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_e5_dense_retrieval_handles_paraphrase(
    e5_embedder: SentenceTransformerEmbedder, mini_chunks: list[Chunk]
) -> None:
    retriever = DenseRetriever(e5_embedder)
    retriever.index(mini_chunks)
    assert retriever.search("Какой клиент нужен для VPN?", k=1)[0].chunk.chunk_id == "it-vpn::000"
    # Ни одного общего слова с документом, только смысл.
    top2 = retriever.search("Можно ли трудиться из дома?", k=2)
    assert {r.chunk.doc_id for r in top2} <= {"hr-remote-2023", "hr-remote-2025"}


def test_e5_passage_cache_is_reused(
    e5_embedder: SentenceTransformerEmbedder, mini_chunks: list[Chunk]
) -> None:
    texts = [c.index_text() for c in mini_chunks]
    first = e5_embedder.encode_passages(texts)
    path = e5_embedder.cache_path(texts)
    assert path is not None and path.exists()
    assert len(Path(path).name) == len("0123456789abcdef0123.npy")
    np.testing.assert_allclose(e5_embedder.encode_passages(texts), first)


def test_cross_encoder_reranks_bm25_candidates(mini_chunks: list[Chunk]) -> None:
    pytest.importorskip("sentence_transformers")
    bm25 = BM25Retriever()
    bm25.index(mini_chunks)
    question = "Сколько календарных дней длится ежегодный отпуск?"
    candidates = bm25.search(question, k=5)
    reranker = CrossEncoderReranker(RERANK_MODEL, device="cpu")
    results = reranker.rerank(question, candidates, top_k=2)
    assert results[0].chunk.chunk_id == "hr-vacation::000"
    assert results[0].details["pre_rerank"] == pytest.approx(results[0].details["bm25"])
    assert results[0].score >= results[1].score
