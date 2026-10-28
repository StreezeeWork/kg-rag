"""Признаки для классификатора отказа: что выдача поиска «говорит» о том, есть ли ответ в базе.

Зачем это нужно. Пороговый метод (``RAGPipeline.should_abstain``) смотрит на одно число —
лучший score одного источника. Но вопрос без ответа выдаёт себя и по-другому: лучший фрагмент
почти не отличается от второго (поиск «не нашёл ничего особенного»), BM25 и dense спорят, какой
фрагмент лучший, в найденном тексте нет половины слов вопроса. Каждый такой сигнал по отдельности
слабый, вместе — сильнее. Классическая модель (логистическая регрессия или градиентный бустинг)
учится взвешивать их на размеченных вопросах.

Два жёстких правила, на которых держится честность оценки:

* **Признаки — только из выдачи поиска и текста вопроса.** Никогда не из эталона (evidence,
  ответ, тип вопроса): в сервисе эталона нет, и признак, который его видит, — утечка: модель
  блестяще работала бы на оценке и была бы бесполезна в проде.
* **Те же фрагменты, по которым считается уверенность.** ``contexts`` — первые ``cfg.retrieval.top_k``
  фрагментов выдачи, даже если запрос задал другой ``top_k`` (см. ``RAGPipeline.answer``): в сервисе —
  из ``RAGPipeline.retrieve``, в оценке — ``retrieved[:top_k]``. Выдача с бо́льшим k — продолжение
  выдачи с меньшим, поэтому числа совпадают (это проверяет тест).

Недоступное значение — NaN, а не 0: «у фрагментов нет score реранкера» (реранкер выключен) и
«реранкер поставил 0» — разные вещи. CatBoost обрабатывает NaN сам, логистическая регрессия
заменяет его медианой из обучающей выборки (``models.make_estimator``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from kbrag.eval.metrics import ARCHIVED_STATUS
from kbrag.text import analyze
from kbrag.types import ScoredChunk

NAN = float("nan")

# Источники score в ScoredChunk.details, по которым строятся признаки «формы выдачи».
# Тот же набор, что у порогового метода (``abstention.score_source``).
SCORE_SOURCES = ("bm25", "dense", "rerank")

# Имя признака → почему он может помочь отличить вопрос без ответа.
# Порядок ФИКСИРОВАН: в нём идут столбцы матрицы признаков, а обученная модель читает признаки
# по номеру столбца. Поэтому новый признак добавляют в конец, а файл модели хранит список имён
# (``AbstentionModel.feature_names``) — модель со старым списком сервис не загрузит.
FEATURE_DESCRIPTIONS: dict[str, str] = {
    "bm25_top1": "Лучший BM25-score: если в базе нет ни одного редкого слова вопроса, он низкий.",
    "bm25_top2": "Второй BM25-score: вместе с первым показывает, выделяется ли лидер выдачи.",
    "bm25_gap": "Отрыв лидера по BM25: точное попадание обычно заметно лучше остальных фрагментов.",
    "bm25_mean": "Средний BM25 выдачи: «всё одинаково слабо» — типичная картина вопроса без ответа.",
    "bm25_top1_per_term": (
        "Лучший BM25 на одно слово вопроса: сырой BM25 растёт с длиной вопроса, а это отношение"
        " сравнимо между короткими и длинными вопросами (линейная модель сама делить не умеет)."
    ),
    "dense_top1": "Лучший косинус эмбеддингов — это и есть уверенность порогового метода (score_source: dense).",
    "dense_top2": "Второй косинус: вместе с первым показывает, выделяется ли лидер выдачи.",
    "dense_gap": "Отрыв лидера по косинусу: перефраз настоящего ответа обычно заметно ближе остальных.",
    "dense_mean": "Средний косинус выдачи: тематическая близость без конкретного ответа поднимает всех сразу.",
    "rerank_top1": "Лучший score cross-encoder: он читает вопрос и фрагмент вместе (нет реранкера — NaN).",
    "rerank_top2": "Второй score реранкера: вместе с первым показывает, выделяется ли лидер.",
    "rerank_gap": "Отрыв лидера по реранкеру: уверенный выбор одного фрагмента — признак найденного ответа.",
    "rerank_mean": "Средний score реранкера по выдаче: фон, на котором оценивается лидер.",
    "bm25_dense_same_top1": (
        "1, если BM25 и эмбеддинги считают лучшим один и тот же фрагмент: когда ответ есть,"
        " два разных поисковика чаще сходятся (NaN, если нет одного из источников)."
    ),
    "n_docs": "Сколько разных документов в выдаче: разброс по базе — признак того, что поиск «шарит наугад».",
    "archived_share": "Доля архивных фрагментов: вопрос «про старое» или путаница редакций меняет картину score.",
    "q_len": "Число значимых слов (стемов) вопроса: короткие вопросы дают шумные score, их порог другой.",
    "q_cov_top1": (
        "Доля слов вопроса, найденных в лучшем фрагменте (с заголовком): если половины слов нет,"
        " фрагмент, скорее всего, про соседнюю тему."
    ),
    "q_cov_topk": (
        "Доля слов вопроса, найденных хоть где-то в выдаче: слово, которого нет нигде"
        " («столовая», «Казань»), часто и есть та деталь, которой нет в базе."
    ),
    "q_has_digits": "1, если в вопросе есть цифры: вопросы про конкретные числа и коды проверяются строже.",
}
FEATURE_NAMES: tuple[str, ...] = tuple(FEATURE_DESCRIPTIONS)


def extract_features(question: str, contexts: Sequence[ScoredChunk]) -> dict[str, float]:
    """Признаки одного вопроса по его выдаче: ``{имя признака: число}`` для всех ``FEATURE_NAMES``.

    ``contexts`` — первые ``retrieval.top_k`` фрагментов в том порядке, в каком их видит генератор
    (после реранкера или слияния гибрида). «Лучший фрагмент» для признаков покрытия — первый
    в этом порядке: его генератор читает первым.
    """
    question_stems = set(analyze(question))
    features: dict[str, float] = {}
    for source in SCORE_SOURCES:
        features.update(_source_features(contexts, source))
    features["bm25_top1_per_term"] = _safe_ratio(features["bm25_top1"], len(question_stems))
    features["bm25_dense_same_top1"] = _same_top1(contexts, "bm25", "dense")
    features["n_docs"] = float(len({item.chunk.doc_id for item in contexts}))
    features["archived_share"] = _safe_ratio(
        sum(1 for item in contexts if item.chunk.metadata.get("status") == ARCHIVED_STATUS), len(contexts)
    )
    features["q_len"] = float(len(question_stems))
    features.update(_coverage_features(question_stems, contexts))
    features["q_has_digits"] = 1.0 if any(char.isdigit() for char in question) else 0.0
    # Возвращаем в каноническом порядке: так словарь легко сравнить глазами с FEATURE_NAMES.
    return {name: features[name] for name in FEATURE_NAMES}


def features_to_matrix(
    rows: Sequence[Mapping[str, float]], feature_names: Sequence[str] = FEATURE_NAMES
) -> np.ndarray:
    """Список словарей признаков → матрица ``(n_вопросов, n_признаков)`` в порядке ``feature_names``.

    Отсутствующий ключ — NaN (так же, как недоступное значение). Модели получают только матрицу,
    поэтому порядок столбцов задаётся здесь и только здесь.
    """
    matrix = np.full((len(rows), len(feature_names)), np.nan, dtype=np.float64)
    for i, row in enumerate(rows):
        for j, name in enumerate(feature_names):
            value = row.get(name)
            if value is not None:
                matrix[i, j] = float(value)
    return matrix


# ---------------------------------------------------------------------------
# Группы признаков
# ---------------------------------------------------------------------------


def _source_features(contexts: Sequence[ScoredChunk], source: str) -> dict[str, float]:
    """``{source}_top1/top2/gap/mean`` по ``details[source]`` фрагментов выдачи.

    Значения сортируем сами, а не берём по позиции: в гибриде выдача упорядочена по RRF, и лучший
    по dense фрагмент может стоять третьим. ``top1`` — максимум, как у ``RAGPipeline.confidence``,
    поэтому ``dense_top1`` в точности равен уверенности порогового метода. Фрагменты без этого
    источника (в гибриде компонент мог его не найти) пропускаются.
    """
    values = sorted(
        (float(item.details[source]) for item in contexts if source in item.details), reverse=True
    )
    top1 = values[0] if values else NAN
    top2 = values[1] if len(values) > 1 else NAN
    return {
        f"{source}_top1": top1,
        f"{source}_top2": top2,
        f"{source}_gap": top1 - top2 if len(values) > 1 else NAN,
        f"{source}_mean": sum(values) / len(values) if values else NAN,
    }


def _same_top1(contexts: Sequence[ScoredChunk], first: str, second: str) -> float:
    """1.0, если лучший фрагмент по ``first`` и по ``second`` — один и тот же; NaN, если сравнить не с чем."""
    best_first = _best_chunk_id(contexts, first)
    best_second = _best_chunk_id(contexts, second)
    if best_first is None or best_second is None:
        return NAN
    return 1.0 if best_first == best_second else 0.0


def _best_chunk_id(contexts: Sequence[ScoredChunk], source: str) -> str | None:
    # max() при равенстве берёт первый по порядку выдачи — результат детерминирован.
    scored = [item for item in contexts if source in item.details]
    if not scored:
        return None
    return max(scored, key=lambda item: item.details[source]).chunk.chunk_id


def _coverage_features(question_stems: set[str], contexts: Sequence[ScoredChunk]) -> dict[str, float]:
    """Какая доля стемов вопроса встречается в лучшем фрагменте и во всей выдаче.

    Текст фрагмента берём вместе с заголовком документа и раздела (``Chunk.index_text``): слова
    вопроса «сколько дней отпуска» часто стоят в названии документа, а не в тексте раздела,
    и генератор этот заголовок тоже видит. Стемы — тот же ``text.analyze``, что у BM25.

    Пустая выдача: в «лучшем фрагменте» доля не определена (NaN), а во всей выдаче — 0
    (ни одно слово не найдено). Вопрос из одних стоп-слов: доля не определена (NaN) в обоих случаях.
    """
    if not question_stems:
        return {"q_cov_top1": NAN, "q_cov_topk": NAN}
    fragment_stems = [set(analyze(item.chunk.index_text(with_header=True))) for item in contexts]
    found_anywhere = set().union(*fragment_stems) if fragment_stems else set()
    return {
        "q_cov_top1": (
            len(question_stems & fragment_stems[0]) / len(question_stems) if fragment_stems else NAN
        ),
        "q_cov_topk": len(question_stems & found_anywhere) / len(question_stems),
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Отношение или NaN: деление на ноль и NaN в числителе дают «значение недоступно»."""
    if denominator == 0 or math.isnan(numerator):
        return NAN
    return numerator / denominator
