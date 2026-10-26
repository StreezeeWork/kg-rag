"""Гибридный поиск: объединяем выдачи BM25 и плотного поиска в один список.

BM25 и эмбеддинги ошибаются на разных вопросах (точные коды против перефразировок),
поэтому их объединение обычно лучше каждого по отдельности. Сложность в том, что
score у них в разных шкалах (BM25 — от 0 до десятков, косинус — от −1 до 1), и
складывать их напрямую нельзя. Два стандартных способа:

- **RRF** (Reciprocal Rank Fusion): смотрим только на места в выдаче,
  ``score = Σ w / (k + rank)``. Шкалы не важны вообще — надёжный выбор по умолчанию.
- **weighted**: приводим score каждого списка к [0, 1] (min-max) и складываем с весами.
  Учитывает «насколько» один кандидат лучше другого, но чувствителен к выбросам.

В ``details`` результата сохраняются исходные score и ранги компонентов: по ним пайплайн
оценивает уверенность (например, максимальный косинус dense) и решает, не отказаться ли от ответа.
"""

from __future__ import annotations

import numpy as np

from kbrag.retrieval.base import Retriever
from kbrag.types import Chunk, ScoredChunk

FUSION_METHODS = ("rrf", "weighted")


def rrf_fuse(
    result_lists: dict[str, list[ScoredChunk]],
    k: int = 60,
    weights: dict[str, float] | None = None,
) -> list[ScoredChunk]:
    """Reciprocal Rank Fusion: ``score = Σ weight_name / (k + rank)``, rank считается с 1.

    ``k`` сглаживает разницу между соседними местами: при k=60 первое и второе место
    различаются слабо, и один компонент не «перекрикивает» другой. Вес компонента,
    не указанного в ``weights``, равен 1.0. Списки должны быть отсортированы по убыванию
    релевантности — именно так их возвращает ``Retriever.search``.
    """
    if k < 0:
        raise ValueError(f"rrf k должен быть >= 0, получено {k}")
    weights = weights or {}
    fused: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    details: dict[str, dict[str, float]] = {}
    for name, results in result_lists.items():
        weight = weights.get(name, 1.0)
        for rank, item in enumerate(_unique_by_chunk(results), start=1):
            chunk_id = item.chunk.chunk_id
            chunks.setdefault(chunk_id, item.chunk)
            fused[chunk_id] = fused.get(chunk_id, 0.0) + weight / (k + rank)
            _record_component(details, chunk_id, name, item.score, rank)
    return _sorted_hybrid_results(fused, chunks, details)


def weighted_fuse(
    result_lists: dict[str, list[ScoredChunk]],
    weights: dict[str, float],
) -> list[ScoredChunk]:
    """Взвешенная сумма min-max-нормированных score.

    Каждый список приводится к [0, 1]: лучший → 1, худший → 0; если все score в списке
    равны (или элемент один) — все получают 1.0. Фрагмент, которого нет в списке
    компонента, получает от него 0.
    """
    missing = [name for name in result_lists if name not in weights]
    if missing:
        raise ValueError(f"weighted_fuse: не заданы веса для компонентов {missing}")
    fused: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    details: dict[str, dict[str, float]] = {}
    for name, results in result_lists.items():
        unique = _unique_by_chunk(results)
        if not unique:
            continue
        raw = np.array([item.score for item in unique], dtype=np.float64)
        low, high = raw.min(), raw.max()
        normalized = (raw - low) / (high - low) if high > low else np.ones_like(raw)
        for rank, (item, norm_score) in enumerate(zip(unique, normalized, strict=True), start=1):
            chunk_id = item.chunk.chunk_id
            chunks.setdefault(chunk_id, item.chunk)
            fused[chunk_id] = fused.get(chunk_id, 0.0) + weights[name] * float(norm_score)
            _record_component(details, chunk_id, name, item.score, rank)
    return _sorted_hybrid_results(fused, chunks, details)


class HybridRetriever(Retriever):
    """Опрашивает несколько поисковиков и сливает их выдачи (RRF или weighted)."""

    name = "hybrid"

    def __init__(
        self,
        retrievers: list[Retriever],
        fusion: str = "rrf",
        rrf_k: int = 60,
        weights: dict[str, float] | None = None,
        candidates_k: int = 30,
    ) -> None:
        super().__init__()
        if not retrievers:
            raise ValueError("HybridRetriever: нужен хотя бы один компонент")
        names = [r.name for r in retrievers]
        if len(set(names)) != len(names):
            raise ValueError(f"HybridRetriever: имена компонентов должны быть уникальны, получено {names}")
        if fusion not in FUSION_METHODS:
            raise ValueError(f"Неизвестный способ слияния {fusion!r}; доступны: {FUSION_METHODS}")
        if fusion == "weighted":
            weights = weights if weights is not None else {n: 1.0 for n in names}
            missing = [n for n in names if n not in weights]
            if missing:
                # Проверяем при создании, а не при первом запросе: ошибка конфига видна сразу.
                raise ValueError(f"HybridRetriever: не заданы веса для компонентов {missing}")
        if candidates_k <= 0:
            raise ValueError(f"candidates_k должен быть > 0, получено {candidates_k}")
        self.retrievers = list(retrievers)
        self.fusion = fusion
        self.rrf_k = rrf_k
        self.weights = dict(weights) if weights is not None else None
        self.candidates_k = candidates_k

    def index(self, chunks: list[Chunk]) -> None:
        self._set_chunks(chunks)
        for retriever in self.retrievers:
            retriever.index(chunks)

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        self._require_index()
        if k <= 0:
            return []
        # Каждый компонент отдаёт больше кандидатов, чем нужно в итоге: фрагмент с 15-го места
        # в BM25 и 2-го в dense после слияния может оказаться в топ-5.
        n_candidates = max(k, self.candidates_k)
        result_lists = {r.name: r.search(query, n_candidates) for r in self.retrievers}
        return self._fuse(result_lists)[:k]

    def search_batch(self, queries: list[str], k: int) -> list[list[ScoredChunk]]:
        self._require_index()
        if k <= 0:
            return [[] for _ in queries]
        n_candidates = max(k, self.candidates_k)
        # Пакетный поиск компонентов (dense считает все запросы одним умножением матриц).
        per_component = {r.name: r.search_batch(queries, n_candidates) for r in self.retrievers}
        fused_batch: list[list[ScoredChunk]] = []
        for query_idx in range(len(queries)):
            result_lists = {name: batch[query_idx] for name, batch in per_component.items()}
            fused_batch.append(self._fuse(result_lists)[:k])
        return fused_batch

    def _fuse(self, result_lists: dict[str, list[ScoredChunk]]) -> list[ScoredChunk]:
        if self.fusion == "rrf":
            return rrf_fuse(result_lists, k=self.rrf_k, weights=self.weights)
        assert self.weights is not None  # гарантировано проверкой в __init__
        return weighted_fuse(result_lists, self.weights)


# --- внутренние помощники ---


def _unique_by_chunk(results: list[ScoredChunk]) -> list[ScoredChunk]:
    """Убирает повторы одного фрагмента в списке, оставляя первое (лучшее) вхождение."""
    seen: set[str] = set()
    unique: list[ScoredChunk] = []
    for item in results:
        if item.chunk.chunk_id not in seen:
            seen.add(item.chunk.chunk_id)
            unique.append(item)
    return unique


def _record_component(
    details: dict[str, dict[str, float]], chunk_id: str, name: str, score: float, rank: int
) -> None:
    """Сохраняет исходный score и ранг компонента: ``{"bm25": 7.1, "bm25_rank": 2}``."""
    info = details.setdefault(chunk_id, {})
    info[name] = float(score)
    info[f"{name}_rank"] = rank


def _sorted_hybrid_results(
    fused: dict[str, float],
    chunks: dict[str, Chunk],
    details: dict[str, dict[str, float]],
) -> list[ScoredChunk]:
    results = [
        ScoredChunk(
            chunk=chunks[chunk_id],
            score=score,
            source=HybridRetriever.name,
            details={**details[chunk_id], HybridRetriever.name: score},
        )
        for chunk_id, score in fused.items()
    ]
    # Округление в ключе сортировки: суммы вроде 0.35 + 0.3 и 0.65 могут отличаться
    # в последнем бите float, и без него tie-break по chunk_id срабатывал бы непредсказуемо.
    results.sort(key=lambda item: (-round(item.score, 12), item.chunk.chunk_id))
    return results
