"""Лексический поиск Okapi BM25 — собственная реализация на numpy.

Зачем своя, а не библиотека: формула короткая, а написав её сами, мы точно знаем,
как обрабатываются словоформы (общий анализатор ``text.analyze`` со стеммингом),
стоп-слова и заголовки фрагментов. BM25 — сильный и дешёвый baseline: он отлично
находит редкие точные слова (номера, названия систем), где плотные эмбеддинги ошибаются.

Формула для запроса q и фрагмента d::

    score(q, d) = Σ_t  qtf(t) · idf(t) · tf(t,d)·(k1 + 1) / (tf(t,d) + k1·(1 − b + b·|d| / avgdl))
    idf(t)      = ln(1 + (N − df(t) + 0.5) / (df(t) + 0.5))

- ``k1`` — насыщение: десятое упоминание слова добавляет меньше, чем первое;
- ``b`` — поправка на длину: длинный фрагмент не должен выигрывать только за счёт объёма;
- ``qtf`` — сколько раз терм встретился в запросе (повтор слова усиливает его вес,
  как в классических реализациях, где сумма идёт по всем словам запроса).
"""

from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict

import numpy as np

from kbrag.retrieval.base import Retriever
from kbrag.text import SPLIT_COMPOUNDS_MODES, SplitCompounds, analyze
from kbrag.types import Chunk, ScoredChunk

logger = logging.getLogger(__name__)


class BM25Retriever(Retriever):
    """Okapi BM25 с обратным индексом: терм → (номера фрагментов, вклад терма в каждый)."""

    name = "bm25"

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        with_header: bool = True,
        split_compounds: SplitCompounds = "none",
    ) -> None:
        super().__init__()
        if k1 < 0:
            raise ValueError(f"k1 должен быть >= 0, получено {k1}")
        if not 0.0 <= b <= 1.0:
            raise ValueError(f"b должен быть в диапазоне [0, 1], получено {b}")
        if split_compounds not in SPLIT_COMPOUNDS_MODES:
            raise ValueError(
                f"split_compounds: ожидается одно из {SPLIT_COMPOUNDS_MODES}, получено {split_compounds!r}"
            )
        self.k1 = k1
        self.b = b
        self.with_header = with_header
        # Раскладывать ли составные токены («vpn-шлюз» → + «vpn», «шлюз»), см. text.analyze.
        # Один и тот же режим обязан действовать и при индексации, и для запроса.
        self.split_compounds = split_compounds
        self.idf: dict[str, float] = {}
        self.doc_len: np.ndarray = np.zeros(0)
        self.avgdl: float = 0.0
        # Обратный индекс. Вклад tf·(k1+1)/(tf + k1·norm) зависит только от фрагмента и терма,
        # поэтому считаем его один раз при индексации, а при поиске остаётся сложить векторы.
        self._postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def index(self, chunks: list[Chunk]) -> None:
        self._set_chunks(chunks)
        n_docs = len(chunks)

        # 1) Проходим по фрагментам и собираем «сырые» постинги: терм → [(фрагмент, tf)].
        postings_raw: dict[str, list[tuple[int, int]]] = defaultdict(list)
        doc_len = np.zeros(n_docs, dtype=np.float64)
        for doc_idx, chunk in enumerate(chunks):
            terms = analyze(chunk.index_text(self.with_header), split_compounds=self.split_compounds)
            doc_len[doc_idx] = len(terms)
            for term, tf in Counter(terms).items():
                postings_raw[term].append((doc_idx, tf))

        self.doc_len = doc_len
        self.avgdl = float(doc_len.mean()) if n_docs else 0.0
        # Защита от деления на ноль, если все фрагменты состоят только из стоп-слов.
        safe_avgdl = self.avgdl if self.avgdl > 0 else 1.0
        length_norm = self.k1 * (1.0 - self.b + self.b * doc_len / safe_avgdl)

        # 2) Для каждого терма считаем idf и вклад во фрагменты, где он встречается.
        self.idf = {}
        self._postings = {}
        for term, pairs in postings_raw.items():
            doc_ids = np.array([doc_idx for doc_idx, _ in pairs], dtype=np.int64)
            tf = np.array([count for _, count in pairs], dtype=np.float64)
            df = len(pairs)
            self.idf[term] = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            term_weights = tf * (self.k1 + 1.0) / (tf + length_norm[doc_ids])
            self._postings[term] = (doc_ids, term_weights)

        logger.debug("BM25: проиндексировано %d фрагментов, %d термов", n_docs, len(self.idf))

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        self._require_index()
        if k <= 0 or not self.chunks:
            return []
        scores = self.score_all(query)
        # Нулевой score = ни одного общего терма с запросом; такие фрагменты не возвращаем,
        # чтобы на «чужой» вопрос выдача была пустой, а не случайной.
        return self._results_from_scores(scores, k, positive_only=True)

    def score_all(self, query: str) -> np.ndarray:
        """BM25-score запроса для каждого проиндексированного фрагмента (полезно для отладки)."""
        self._require_index()
        scores = np.zeros(len(self.chunks), dtype=np.float64)
        for term, query_tf in Counter(analyze(query, split_compounds=self.split_compounds)).items():
            posting = self._postings.get(term)
            if posting is None:
                continue  # терма нет в корпусе — он ничего не добавляет
            doc_ids, term_weights = posting
            # В одном постинге номера фрагментов уникальны, поэтому обычное += корректно.
            scores[doc_ids] += query_tf * self.idf[term] * term_weights
        return scores
