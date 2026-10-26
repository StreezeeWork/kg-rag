"""Поиск релевантных фрагментов: BM25, плотные эмбеддинги, гибрид и реранкер.

Типичное использование::

    retriever = build_retriever(cfg)   # режим берётся из cfg.retrieval.mode
    retriever.index(chunks)
    results = retriever.search("Сколько дней отпуска?", k=5)

Тяжёлые нейросетевые зависимости (sentence-transformers, torch) импортируются лениво —
только при создании ``SentenceTransformerEmbedder`` или ``CrossEncoderReranker``.
"""

from kbrag.retrieval.base import Retriever, top_k_indices
from kbrag.retrieval.bm25 import BM25Retriever
from kbrag.retrieval.dense import DenseRetriever
from kbrag.retrieval.embeddings import (
    Embedder,
    LSAEmbedder,
    SentenceTransformerEmbedder,
    make_embedder,
    neural_available,
)
from kbrag.retrieval.factory import build_retriever
from kbrag.retrieval.hybrid import HybridRetriever, rrf_fuse, weighted_fuse
from kbrag.retrieval.rerank import CrossEncoderReranker, Reranker, make_reranker

__all__ = [
    "BM25Retriever",
    "CrossEncoderReranker",
    "DenseRetriever",
    "Embedder",
    "HybridRetriever",
    "LSAEmbedder",
    "Reranker",
    "Retriever",
    "SentenceTransformerEmbedder",
    "build_retriever",
    "make_embedder",
    "make_reranker",
    "neural_available",
    "rrf_fuse",
    "top_k_indices",
    "weighted_fuse",
]
