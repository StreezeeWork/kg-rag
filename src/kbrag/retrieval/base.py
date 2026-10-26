"""Общий интерфейс всех поисковиков (retriever-ов).

Пайплайн, оценка и API работают с абстрактным ``Retriever`` и не знают, что внутри:
BM25, плотные векторы или их гибрид. Благодаря этому способ поиска меняется одной
строкой в конфиге, а метрики разных способов сравниваются честно — на одном коде оценки.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from kbrag.types import Chunk, ScoredChunk


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Индексы ``k`` наибольших значений по убыванию; при равенстве меньший индекс идёт раньше.

    Стабильный tie-break нужен для воспроизводимости: без него два фрагмента с одинаковым
    score могли бы меняться местами между запусками, и метрики «плавали» бы.
    Полная сортировка — O(n log n); для корпусов в сотни тысяч фрагментов это доли секунды.
    """
    values = np.asarray(scores, dtype=np.float64).ravel()
    if k <= 0 or values.size == 0:
        return np.empty(0, dtype=np.int64)
    # kind="stable" сохраняет исходный порядок равных элементов → меньший индекс раньше.
    order = np.argsort(-values, kind="stable")
    return order[:k].astype(np.int64)


class Retriever(ABC):
    """Базовый класс поисковика: сначала ``index(chunks)``, затем ``search(query, k)``.

    Контракт результата ``search``: не больше ``k`` элементов, по убыванию score,
    ``source == self.name``, ``details == {self.name: score}``. Поле ``details`` потом
    дополняют гибрид и реранкер — по нему пайплайн оценивает «уверенность» поиска.
    """

    name: str = "base"

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self._is_indexed = False

    @abstractmethod
    def index(self, chunks: list[Chunk]) -> None:
        """Строит индекс по фрагментам (повторный вызов полностью заменяет индекс)."""

    @abstractmethod
    def search(self, query: str, k: int) -> list[ScoredChunk]:
        """Возвращает до ``k`` самых релевантных фрагментов по убыванию score."""

    def search_batch(self, queries: list[str], k: int) -> list[list[ScoredChunk]]:
        """Поиск по нескольким запросам; наследники переопределяют, если умеют быстрее.

        Это часть контракта для пакетных запросов (прогрев, офлайн-обработка списка вопросов):
        dense кодирует все запросы одним вызовом модели. Тесты проверяют, что результат
        совпадает с ``search`` по одному запросу. Оценка (``eval.runner``) намеренно ищет
        по одному вопросу: она меряет задержку каждого ответа (p50/p95), а пакет её размазал бы.
        """
        return [self.search(query, k) for query in queries]

    # --- вспомогательные методы для наследников ---

    def _set_chunks(self, chunks: list[Chunk]) -> None:
        """Запоминает фрагменты и проверяет уникальность ``chunk_id``.

        Гибрид и метрики сопоставляют фрагменты по ``chunk_id``: дубликат молча
        исказил бы и слияние, и оценку, поэтому лучше упасть сразу.
        """
        seen: set[str] = set()
        for chunk in chunks:
            if chunk.chunk_id in seen:
                raise ValueError(f"Повторяющийся chunk_id в индексе: {chunk.chunk_id!r}")
            seen.add(chunk.chunk_id)
        self.chunks = list(chunks)
        self._is_indexed = True

    def _require_index(self) -> None:
        if not self._is_indexed:
            raise RuntimeError(f"{type(self).__name__}: сначала вызовите index(chunks)")

    def _results_from_scores(
        self, scores: np.ndarray, k: int, positive_only: bool = False
    ) -> list[ScoredChunk]:
        """Превращает вектор score (по одному на фрагмент) в отсортированную выдачу."""
        results: list[ScoredChunk] = []
        for idx in top_k_indices(scores, k):
            score = float(scores[idx])
            if positive_only and score <= 0.0:
                # Список отсортирован по убыванию — дальше тоже только нули.
                break
            results.append(
                ScoredChunk(chunk=self.chunks[idx], score=score, source=self.name, details={self.name: score})
            )
        return results
