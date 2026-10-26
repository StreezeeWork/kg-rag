"""Сборка поисковика по конфигу.

Единственное место, которое знает, как из ``retrieval.mode`` получить конкретные классы.
Пайплайн, оценка и API вызывают ``build_retriever(cfg)`` и не зависят от деталей.
"""

from __future__ import annotations

import logging

from kbrag.config import AppConfig
from kbrag.retrieval.base import Retriever
from kbrag.retrieval.bm25 import BM25Retriever
from kbrag.retrieval.dense import DenseRetriever
from kbrag.retrieval.embeddings import make_embedder
from kbrag.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)


def _build_bm25(cfg: AppConfig) -> BM25Retriever:
    params = cfg.retrieval.bm25
    return BM25Retriever(
        k1=params.k1,
        b=params.b,
        with_header=cfg.chunking.with_header,
        split_compounds=params.split_compounds,
    )


def _build_dense(cfg: AppConfig) -> DenseRetriever:
    embedder = make_embedder(cfg.retrieval.dense, cache_dir=cfg.cache_dir)
    return DenseRetriever(embedder, with_header=cfg.chunking.with_header)


def build_retriever(cfg: AppConfig) -> Retriever:
    """``bm25`` → BM25; ``dense`` → плотный поиск; ``hybrid`` → BM25 + dense со слиянием."""
    mode = cfg.retrieval.mode
    logger.info("Режим поиска: %s", mode)
    if mode == "bm25":
        return _build_bm25(cfg)
    if mode == "dense":
        return _build_dense(cfg)
    if mode == "hybrid":
        hybrid = cfg.retrieval.hybrid
        return HybridRetriever(
            [_build_bm25(cfg), _build_dense(cfg)],
            fusion=hybrid.fusion,
            rrf_k=hybrid.rrf_k,
            weights=dict(hybrid.weights),
            candidates_k=cfg.retrieval.candidates_k,
        )
    raise ValueError(f"Неизвестный режим поиска: {mode!r}")
