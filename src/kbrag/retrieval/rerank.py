"""Реранкинг: точная, но медленная модель пересортировывает несколько десятков кандидатов.

Поисковики (bi-encoder, BM25) кодируют вопрос и фрагмент **по отдельности** — это быстро,
но модель не видит их вместе. Cross-encoder читает пару «вопрос + фрагмент» целиком и
оценивает релевантность заметно точнее. Прогнать его по всей базе слишком дорого, поэтому
схема двухэтапная: дешёвый поиск достаёт ``candidates_k`` кандидатов, реранкер выбирает ``top_k``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

from kbrag.config import RerankConfig
from kbrag.retrieval.embeddings import NEURAL_INSTALL_HINT, resolve_device
from kbrag.types import ScoredChunk

logger = logging.getLogger(__name__)


class Reranker(ABC):
    """Интерфейс реранкера: получает кандидатов, возвращает новый порядок и новые score."""

    name: str = "reranker"

    @abstractmethod
    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        """Возвращает до ``top_k`` кандидатов по убыванию нового score."""


class CrossEncoderReranker(Reranker):
    """Реранкер на ``sentence_transformers.CrossEncoder``.

    Старый score сохраняется в ``details["pre_rerank"]``, а все прежние детали (bm25, dense,
    ранги) — переносятся: так видно, как реранкер изменил порядок, а пайплайн может брать
    уверенность из ``details["rerank"]``.
    """

    name = "rerank"

    def __init__(
        self,
        model_name: str,
        batch_size: int = 16,
        max_length: int = 512,
        device: str = "auto",
        with_header: bool = True,
        revision: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                f"Для реранкера нужны дополнительные зависимости: {NEURAL_INSTALL_HINT}"
            ) from exc

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = resolve_device(device)
        self.with_header = with_header
        self.revision = revision
        # max_length обрезает слишком длинные пары: иначе модель упадёт или будет очень медленной.
        # revision (commit снапшота на HF Hub) передаём, только если он задан в конфиге.
        extra = {"revision": revision} if revision else {}
        self.model = CrossEncoder(model_name, max_length=max_length, device=self.device, **extra)

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        if not candidates or top_k <= 0:
            return []
        # Реранкер видит тот же текст, что и индекс: with_header приходит из chunking.with_header
        # (см. make_reranker). По умолчанию — с заголовком документа/раздела: фрагмент
        # «Срок — 5 дней» без заголовка непонятно к чему относится.
        pairs = [(query, item.chunk.index_text(self.with_header)) for item in candidates]
        raw_scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        scores = np.asarray(raw_scores, dtype=np.float64)
        if scores.ndim == 2 and scores.shape[1] == 1:
            scores = scores[:, 0]  # некоторые модели возвращают столбец (n, 1)
        if scores.shape != (len(candidates),):
            raise ValueError(
                f"CrossEncoder вернул score формы {scores.shape}, ожидалось ({len(candidates)},); "
                "нужна модель с одним выходом (релевантность)"
            )

        reranked = [
            ScoredChunk(
                chunk=item.chunk,
                score=float(score),
                source=self.name,
                details={**item.details, "rerank": float(score), "pre_rerank": float(item.score)},
            )
            for item, score in zip(candidates, scores, strict=True)
        ]
        reranked.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
        return reranked[:top_k]


def make_reranker(cfg: RerankConfig, *, with_header: bool = True) -> Reranker | None:
    """Создаёт реранкер по секции ``rerank`` конфига; ``None``, если реранкинг выключен.

    ``with_header`` передаётся из ``cfg.chunking.with_header`` (секция ``rerank`` о нарезке
    ничего не знает): реранкер должен оценивать тот же текст фрагмента, что индексировали
    поисковики. Иначе абляция «без заголовков» была бы нечестной — поиск без них, реранкер с ними.
    """
    if not cfg.enabled:
        return None
    logger.info("Загружается реранкер %s", cfg.model)
    return CrossEncoderReranker(
        model_name=cfg.model,
        batch_size=cfg.batch_size,
        max_length=cfg.max_length,
        device=cfg.device,
        with_header=with_header,
        revision=cfg.revision,
    )
