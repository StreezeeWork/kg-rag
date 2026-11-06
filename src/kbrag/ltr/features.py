"""Признаки кандидата для learning-to-rank: что о фрагменте говорят BM25, dense, реранкер и его документ.

Зачем это нужно. Без LTR финальный порядок кандидатов задаёт одно число — score cross-encoder.
Он точнее остальных на «обычных» вопросах, но ошибается по-своему: на вопросе, ответ на который лежит
в двух документах, он поднимает несколько кусков самого похожего документа и вытесняет второй
документ из топа. BM25 и dense ошибаются иначе. Модель ранжирования (``ltr.models``) учится
складывать эти сигналы на размеченных вопросах.

Правила, на которых держится честность (те же, что у ``abstention.features``):

* **Признаки — только из кандидатов и текста вопроса**, никогда из эталона: в сервисе эталона нет.
* **Не зависят от порядка, в котором пришли кандидаты.** Ранги считаются здесь заново по score
  (при равенстве — по ``chunk_id``): иначе признак «место в выдаче» молча зависел бы от того,
  кто и как отсортировал список до нас.
* **Недоступное значение — NaN, а не 0.** «Фрагмент не попал в топ BM25» и «BM25 дал ему 0» —
  разные вещи. CatBoost обрабатывает NaN сам.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from kbrag.eval.metrics import ARCHIVED_STATUS
from kbrag.text import analyze
from kbrag.types import ScoredChunk

NAN = float("nan")

# Имя признака → почему он может помочь поставить нужный фрагмент выше.
# Порядок ФИКСИРОВАН: в нём идут столбцы матрицы, а обученная модель читает признаки по номеру
# столбца. Новый признак добавляют в конец; файл модели хранит список имён (``RankerModel.feature_names``),
# и модель со старым списком сервис не загрузит.
FEATURE_DESCRIPTIONS: dict[str, str] = {
    "rerank": "Score cross-encoder: он читает вопрос и фрагмент вместе — самый точный одиночный сигнал.",
    "rerank_rank": "Место по реранкеру (1 — лучший): порядок сравним между вопросами, а сырой score — нет.",
    "rerank_gap_top": (
        "Отставание от лучшего score реранкера в этом вопросе: у разных вопросов шкала cross-encoder"
        " сдвинута по-разному, а разность с лидером сравнима."
    ),
    "rerank_z": "z-оценка score реранкера среди кандидатов вопроса: насколько фрагмент выделяется из толпы.",
    "rrf": "Score гибрида RRF (BM25 + dense) до реранкера.",
    "rrf_rank": (
        "Место в гибриде до реранкера. Здесь второй документ multi_hop-вопроса часто стоит выше,"
        " чем у реранкера: BM25 и dense ищут каждое слово вопроса, а не общий смысл."
    ),
    "bm25": "BM25-score: точные термины и коды («OTP-1», «size/XL»), которых не знают эмбеддинги.",
    "bm25_rank": "Место в выдаче BM25 (NaN — фрагмент не попал в её топ).",
    "bm25_rel": "BM25 относительно лучшего BM25 вопроса: сырой BM25 растёт с длиной вопроса, доля — нет.",
    "dense": "Косинус эмбеддингов e5: перефразы без общих слов с документом.",
    "dense_rank": "Место в выдаче dense (NaN — фрагмент не попал в её топ).",
    "dense_gap_top": "Отставание от лучшего косинуса вопроса: шкала косинусов у вопросов тоже сдвинута по-разному.",
    "doc_n_cands": "Сколько кандидатов из того же документа: документ, который всплывает отовсюду, скорее всего по теме.",
    "rank_in_doc": (
        "Какой по счёту кусок своего документа (по основному порядку): второй-третий кусок того же"
        " документа реже несёт новое доказательство, чем лучший кусок другого документа."
    ),
    "doc_rank": "Место документа среди документов кандидатов (по его лучшему куску).",
    "archived": "Документ архивный: вопросы о правилах почти всегда про действующую редакцию.",
    "q_cov": "Доля слов вопроса во фрагменте (с заголовком): прямое совпадение лексики.",
    "head_cov": "Доля слов вопроса в названии документа и раздела: заголовок часто называет тему точнее текста.",
    "n_words": "Длина фрагмента в словах: длинный кусок чаще содержит слова вопроса просто по объёму.",
    "q_len": (
        "Число значимых слов вопроса. Одинаково у всех кандидатов вопроса, но деревья учитывают"
        " взаимодействия: на длинных вопросах BM25 надёжнее, на коротких — эмбеддинги."
    ),
}
FEATURE_NAMES: tuple[str, ...] = tuple(FEATURE_DESCRIPTIONS)


def candidate_features(
    question: str, candidates: Sequence[ScoredChunk], *, with_header: bool = True
) -> np.ndarray:
    """Матрица признаков ``(len(candidates), len(FEATURE_NAMES))`` — строка на кандидата, в его порядке.

    ``with_header`` — как у индекса (``chunking.with_header``): покрытие слов вопроса считается по тому же
    тексту фрагмента, который видели поисковики и реранкер.
    """
    n = len(candidates)
    if n == 0:
        return np.zeros((0, len(FEATURE_NAMES)))
    ids = [item.chunk.chunk_id for item in candidates]
    rerank = _column(candidates, "rerank")
    rrf = _column(candidates, "hybrid")
    bm25 = _column(candidates, "bm25")
    dense = _column(candidates, "dense")

    # Основной порядок для признаков документа: реранкер, без него — гибрид, а в режиме одного
    # поисковика — score, с которым кандидат пришёл. Ранги считаем сами.
    primary = next(
        (values for values in (rerank, rrf) if np.isfinite(values).any()),
        np.array([float(item.score) for item in candidates]),
    )
    order = _order(primary, ids)
    rank_in_doc = np.empty(n)
    doc_rank = np.empty(n)
    seen_in_doc: dict[str, int] = {}
    doc_position: dict[str, int] = {}
    for idx in order:
        doc_id = candidates[idx].chunk.doc_id
        seen_in_doc[doc_id] = seen_in_doc.get(doc_id, 0) + 1
        doc_position.setdefault(doc_id, len(doc_position) + 1)
        rank_in_doc[idx] = seen_in_doc[doc_id]
        doc_rank[idx] = doc_position[doc_id]
    doc_counts: dict[str, int] = {}
    for item in candidates:
        doc_counts[item.chunk.doc_id] = doc_counts.get(item.chunk.doc_id, 0) + 1

    question_stems = set(analyze(question))
    n_stems = len(question_stems)
    columns = {
        "rerank": rerank,
        "rerank_rank": _ranks(rerank, ids),
        "rerank_gap_top": _gap_to_top(rerank),
        "rerank_z": _zscore(rerank),
        "rrf": rrf,
        "rrf_rank": _ranks(rrf, ids),
        "bm25": bm25,
        "bm25_rank": _column(candidates, "bm25_rank"),
        "bm25_rel": _relative_to_top(bm25),
        "dense": dense,
        "dense_rank": _column(candidates, "dense_rank"),
        "dense_gap_top": _gap_to_top(dense),
        "doc_n_cands": np.array([doc_counts[item.chunk.doc_id] for item in candidates], dtype=float),
        "rank_in_doc": rank_in_doc,
        "doc_rank": doc_rank,
        "archived": np.array(
            [float(item.chunk.metadata.get("status") == ARCHIVED_STATUS) for item in candidates]
        ),
        "q_cov": np.array(
            [_share(question_stems, item.chunk.index_text(with_header)) for item in candidates]
        ),
        "head_cov": np.array(
            [_share(question_stems, f"{item.chunk.title} {item.chunk.section}") for item in candidates]
        ),
        "n_words": np.array([len(item.chunk.text.split()) for item in candidates], dtype=float),
        "q_len": np.full(n, float(n_stems)),
    }
    return np.column_stack([columns[name] for name in FEATURE_NAMES])


# ---------------------------------------------------------------------------
# Помощники
# ---------------------------------------------------------------------------


def _column(candidates: Sequence[ScoredChunk], key: str) -> np.ndarray:
    return np.array([float(item.details.get(key, NAN)) for item in candidates])


def _order(values: np.ndarray, ids: Sequence[str]) -> list[int]:
    """Индексы по убыванию значения; NaN — в конце; при равенстве — по ``chunk_id`` (воспроизводимо)."""
    return sorted(range(len(ids)), key=lambda i: (np.isnan(values[i]), -np.nan_to_num(values[i]), ids[i]))


def _ranks(values: np.ndarray, ids: Sequence[str]) -> np.ndarray:
    """Место по убыванию (1 — лучший) среди кандидатов со значением; у NaN место тоже NaN."""
    ranks = np.full(len(ids), NAN)
    for position, idx in enumerate(i for i in _order(values, ids) if not np.isnan(values[i])):
        ranks[idx] = position + 1
    return ranks


def _gap_to_top(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    return values - finite.max() if finite.size else np.full(values.shape, NAN)


def _relative_to_top(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if not finite.size or finite.max() <= 0:
        return np.full(values.shape, NAN)
    return values / finite.max()


def _zscore(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return np.full(values.shape, NAN)
    # +1e-9: у одного кандидата (или у равных score) std = 0, а z-оценка тогда — честный 0.
    return (values - finite.mean()) / (finite.std() + 1e-9)


def _share(question_stems: set[str], text: str) -> float:
    if not question_stems:
        return NAN
    return len(question_stems & set(analyze(text))) / len(question_stems)
