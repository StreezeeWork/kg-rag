"""Плотный (векторный) поиск: косинусная близость эмбеддингов запроса и фрагментов.

Сильная сторона — перефразировки: «работать из дома» находит «удалённая работа»,
хотя общих слов нет. Слабая — точные редкие токены (номера, коды), где лучше BM25;
поэтому в проекте есть ещё и гибрид.
"""

from __future__ import annotations

import logging

import numpy as np

from kbrag.retrieval.base import Retriever
from kbrag.retrieval.embeddings import Embedder
from kbrag.types import Chunk, ScoredChunk

logger = logging.getLogger(__name__)


class DenseRetriever(Retriever):
    """Ищет по матрице эмбеддингов фрагментов полным перебором (exact search).

    Для базы знаний в тысячи фрагментов перебор через одно матричное умножение
    занимает миллисекунды, а приближённые индексы (FAISS, HNSW) добавили бы
    зависимость и погрешность без заметного выигрыша.
    """

    name = "dense"

    def __init__(self, embedder: Embedder, with_header: bool = True) -> None:
        super().__init__()
        self.embedder = embedder
        self.with_header = with_header
        self.matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)  # (n_chunks, dim)

    def index(self, chunks: list[Chunk]) -> None:
        self._set_chunks(chunks)
        if not chunks:
            # LSA нельзя обучить на пустом корпусе; поиск по пустому индексу просто вернёт [].
            self.matrix = np.zeros((0, 0), dtype=np.float32)
            return
        texts = [chunk.index_text(self.with_header) for chunk in chunks]
        # Обучение и кодирование корпуса одним вызовом: LSA не считает TF-IDF фрагментов дважды.
        matrix = np.asarray(self.embedder.fit_encode_passages(texts), dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(chunks):
            raise ValueError(f"Эмбеддер вернул матрицу формы {matrix.shape}, ожидалось ({len(chunks)}, dim)")
        self.matrix = matrix
        logger.debug("Dense: проиндексировано %d фрагментов, размерность %d", *matrix.shape)

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        return self.search_batch([query], k)[0]

    def search_batch(self, queries: list[str], k: int) -> list[list[ScoredChunk]]:
        self._require_index()
        if not queries:
            return []
        if k <= 0 or not self.chunks:
            return [[] for _ in queries]
        query_vectors = np.asarray(self.embedder.encode_queries(list(queries)), dtype=np.float32)
        # Векторы нормированы, поэтому скалярное произведение = косинус.
        # Одно умножение матриц сразу даёт score всех запросов со всеми фрагментами.
        all_scores = query_vectors @ self.matrix.T  # (n_queries, n_chunks)

        results: list[list[ScoredChunk]] = []
        for query_vector, scores in zip(query_vectors, all_scores, strict=True):
            if not np.any(query_vector):
                # Нулевой вектор: в запросе нет ничего знакомого модели (у LSA — ни одной
                # известной n-граммы). Честнее вернуть пустую выдачу, чем случайный порядок.
                results.append([])
                continue
            results.append(self._results_from_scores(scores, k))
        return results
