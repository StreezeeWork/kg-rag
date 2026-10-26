"""Метрики качества RAG: поиск, генерация, отказы.

Все функции чистые (без состояния и случайности), поэтому их легко проверить
на примерах, посчитанных руками, — см. ``tests/test_metrics.py``.

Как устроены метрики поиска. Для одного вопроса строится **матрица покрытия**
``coverage[i][j]`` — покрывает ли i-й найденный фрагмент j-е доказательство (evidence).
Строки идут в порядке выдачи (ранг 1, 2, …), столбцы — по доказательствам. Все метрики
поиска считаются по этой матрице, так что правило «что такое релевантный фрагмент»
живёт ровно в одном месте — ``dataset.is_relevant``.

Метрики поиска имеют смысл только для вопросов, у которых есть доказательства
(answerable). Пустая выдача даёт 0 по всем метрикам.
"""

from __future__ import annotations

import itertools
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence

from kbrag.eval.dataset import Span, is_relevant
from kbrag.text import STOPWORDS, stem, tokenize
from kbrag.types import Chunk, Citation

CoverageMatrix = list[list[bool]]

# ---------------------------------------------------------------------------
# Поиск
# ---------------------------------------------------------------------------


def coverage_matrix(
    chunks: Sequence[Chunk], spans: Sequence[Span], min_coverage: float = 0.5
) -> CoverageMatrix:
    """``matrix[i][j]`` = покрывает ли ``chunks[i]`` доказательство ``spans[j]``."""
    return [[is_relevant(chunk, span, min_coverage) for span in spans] for chunk in chunks]


def _check_k(k: int) -> None:
    if k < 1:
        raise ValueError(f"k должно быть ≥ 1, получено {k}")


def _n_spans(coverage: CoverageMatrix) -> int:
    """Число доказательств (столбцов). Нужно recall и nDCG, а у вопроса без них эти метрики не определены."""
    n_spans = len(coverage[0])
    if n_spans == 0:
        raise ValueError("метрики поиска определены только для вопросов с evidence")
    return n_spans


def hit_at_k(coverage: CoverageMatrix, k: int) -> float:
    """1.0, если среди первых k фрагментов есть хотя бы один релевантный, иначе 0.0."""
    _check_k(k)
    return 1.0 if any(any(row) for row in coverage[:k]) else 0.0


def recall_at_k(coverage: CoverageMatrix, k: int) -> float:
    """Доля доказательств, покрытых хотя бы одним из первых k фрагментов.

    Для multi_hop-вопроса с двумя доказательствами найти одно — это recall 0.5,
    а не «попадание», поэтому recall строже, чем hit.
    """
    _check_k(k)
    if not coverage:
        return 0.0
    n_spans = _n_spans(coverage)
    top = coverage[:k]
    covered = sum(1 for j in range(n_spans) if any(row[j] for row in top))
    return covered / n_spans


def precision_at_k(coverage: CoverageMatrix, k: int) -> float:
    """Доля релевантных фрагментов среди первых k; знаменатель — всегда k.

    Если поиск вернул меньше k фрагментов, недостающие места считаются нерелевантными:
    иначе система, вернувшая один удачный фрагмент, получала бы precision 1.0.
    """
    _check_k(k)
    relevant = sum(1 for row in coverage[:k] if any(row))
    return relevant / k


def mrr(coverage: CoverageMatrix) -> float:
    """1 / ранг первого релевантного фрагмента (0.0, если релевантных нет).

    Чем выше в выдаче первое доказательство, тем больше шансов, что генератор его «заметит».
    """
    for rank, row in enumerate(coverage, start=1):
        if any(row):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(coverage: CoverageMatrix, k: int) -> float:
    """Нормированный DCG с бинарной пользой.

    Польза фрагмента на ранге r равна 1, только если он покрывает **новое**
    доказательство — ещё не покрытое фрагментами выше. Иначе три куска одной и той
    же цитаты принесли бы тройную пользу, и nDCG вышел бы больше 1.
    Идеальный DCG — когда первые ``min(k, число доказательств)`` мест дают по новому
    доказательству.

    Известное ограничение: если обе цитаты лежат в одном фрагменте, идеальная выдача
    получит nDCG < 1 (польза бинарная, а идеал рассчитан на два фрагмента).
    """
    _check_k(k)
    if not coverage:
        return 0.0
    n_spans = _n_spans(coverage)
    seen: set[int] = set()
    dcg = 0.0
    for rank, row in enumerate(coverage[:k], start=1):
        new = {j for j, covered in enumerate(row) if covered and j not in seen}
        if new:
            dcg += 1.0 / math.log2(rank + 1)
            seen |= new
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(k, n_spans) + 1))
    return dcg / ideal


def retrieval_metrics(
    chunks: Sequence[Chunk],
    spans: Sequence[Span],
    k_values: Sequence[int] = (1, 3, 5, 10),
    min_coverage: float = 0.5,
) -> dict[str, float]:
    """Все метрики поиска для одного вопроса под именами из отчёта: ``hit@1``, …, ``mrr``."""
    if not spans:
        raise ValueError("метрики поиска определены только для вопросов с evidence")
    coverage = coverage_matrix(chunks, spans, min_coverage)
    result: dict[str, float] = {}
    for prefix, metric in (
        ("hit", hit_at_k),
        ("recall", recall_at_k),
        ("precision", precision_at_k),
        ("ndcg", ndcg_at_k),
    ):
        for k in k_values:
            result[f"{prefix}@{k}"] = metric(coverage, k)
    result["mrr"] = mrr(coverage)
    return result


# ---------------------------------------------------------------------------
# Генерация
# ---------------------------------------------------------------------------

# Ссылки вида [1], [1, 2]: их цифры — номера фрагментов, а не факты ответа.
_CITATION_RE = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")
_CYRILLIC_WORD_RE = re.compile(r"[а-я]+")
# Хвост «гласные/й/ь (+х|м|ми)» — типичные падежные окончания.
_SOFT_ENDING_RE = re.compile(r"[аеиоуыэюяйь]+(?:х|м|ми)?$")
# Корни, у которых Snowball не может срезать окончание: основа слишком короткая
# («днях» остаётся «днях»). Только для них soft_stem срезает хвост — см. его докстринг.
_SHORT_ROOTS = frozenset({"дн"})

# Нормализация чисел перед токенизацией (иначе «1 500» — это два терма «1» и «500»,
# а «10:00» — «10» и «00», и ответ «500 рублей, 1 раз» засчитался бы за факт «1 500 рублей»).
_DIGIT_GROUP_RE = re.compile(r"(?<=\d)[    ](?=\d{3}(?!\d))")  # «60 000» → «60000»
_TIME_RE = re.compile(r"(?<![\d:])(\d{1,2}):(\d{2})(?![\d:])")  # «9:00» → «09h00»
_PERCENT_RE = re.compile(r"(\d)\s*(?:%|процент[а-я]*)", re.IGNORECASE)  # «10 %» → «10pct»
# Частицы, которые переворачивают смысл следующего слова. Они стоп-слова, и без особой
# обработки «не оплачиваются» и «оплачиваются» дали бы один и тот же терм.
_NEGATIONS = frozenset({"не", "ни", "без"})


def strip_citations(text: str) -> str:
    """Убирает маркеры ``[n]``, чтобы «[3]» не засчитывалось как факт «3 дня»."""
    return _CITATION_RE.sub(" ", text)


def soft_stem(stem: str) -> str:
    """Дополнительно обрезает падежное окончание у коротких русских основ.

    Стеммер Snowball не трогает окончание, если основа слишком короткая:
    «дней» → «дне», но «днях» → «днях». Для метрик это значит, что ответ
    «28 календарных днях» не засчитался бы за факт «28 календарных дней».
    Поэтому срезаем хвост из гласных (+ «х»/«м»/«ми»): обе формы дают «дн».

    Срезаем **только** когда остаётся известный короткий корень (``_SHORT_ROOTS``).
    Общее правило «срезать гласный хвост у любой основы» склеивало разные слова:
    «приём», «премии» и «прямо» → «пр», «режим» и «реже» → «реж» — и метрика засчитывала
    факт «приём» ответу про премии. Числа и латиница не меняются («28» ≠ «280», VPN ≠ VPS).
    """
    if not _CYRILLIC_WORD_RE.fullmatch(stem):
        return stem
    shorter = _SOFT_ENDING_RE.sub("", stem)
    return shorter if shorter in _SHORT_ROOTS else stem


def _normalize_numbers(text: str) -> str:
    """Склеивает разряды, время и проценты в один токен — до того, как их разрежет токенизатор."""
    text = _DIGIT_GROUP_RE.sub("", text)
    text = _TIME_RE.sub(lambda m: f" {int(m.group(1)):02d}h{m.group(2)} ", text)
    return _PERCENT_RE.sub(r"\1pct ", text)


def _term(token: str) -> str:
    return soft_stem(stem(token))


def match_terms(text: str, remove_stopwords: bool = True) -> list[str]:
    """Термы для сравнения ответов: стемы ``text`` + ``soft_stem``, без маркеров ссылок.

    Отличия от ``text.analyze`` (он общий с BM25, поэтому его не трогаем):

    * числа: «60 000» = «60000», «8:00» = «08:00» (время — один терм), «10%» ≠ «10»;
    * отрицание склеивается со стемом следующего слова в один терм: «не более» → ``не_бол``.
      Так ответ «оплачиваются» не получит балл за факт «не оплачиваются»,
      и наоборот; «не более 2 дней» ≠ «не менее 2 дней». Порядок остальных слов не важен.

    Ограничение: это сравнение лексическое. Синонимы («запрещены» вместо «не допускаются»),
    переставленный диапазон («с 22:00 до 08:00» против «с 08:00 до 22:00») и смысловые
    противоречия оно не видит — для LLM-ответов его стоит дополнять LLM-судьёй.
    """
    tokens = tokenize(_normalize_numbers(strip_citations(text)))
    terms: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in _NEGATIONS and i + 1 < len(tokens):
            terms.append(f"{token}_{_term(tokens[i + 1])}")
            i += 2
            continue
        if not (remove_stopwords and token in STOPWORDS):
            terms.append(_term(token))
        i += 1
    return terms


def token_f1(pred: str, gold: str) -> float:
    """F1 по пересечению мультимножеств термов (как в SQuAD, но со стеммингом).

    Ловит «примерно тот же ответ», но штрафует за многословность: лишние слова
    снижают precision. Если оба текста пустые — 1.0, если пуст один — 0.0.
    """
    pred_terms = match_terms(pred)
    gold_terms = match_terms(gold)
    if not pred_terms or not gold_terms:
        return float(pred_terms == gold_terms)
    common = sum((Counter(pred_terms) & Counter(gold_terms)).values())
    if common == 0:
        return 0.0
    precision = common / len(pred_terms)
    recall = common / len(gold_terms)
    return 2 * precision * recall / (precision + recall)


def fact_recall(pred: str, key_facts: Sequence[str]) -> float | None:
    """Доля ключевых фактов, **все** термы которых встречаются в ответе.

    Строже token_f1 там, где это важно: ответ «14 календарных дней» на вопрос с фактом
    «28 календарных дней» получит 0, хотя слова почти совпадают.
    Факт только из стоп-слов («нет») сравнивается без удаления стоп-слов.
    Как сравниваются отрицания, числа, время и проценты — см. ``match_terms``.
    Нет фактов → None (метрика неприменима, например для неотвечаемого вопроса).
    """
    pred_terms = set(match_terms(pred))
    pred_terms_all = set(match_terms(pred, remove_stopwords=False))
    checked = 0
    found = 0
    for fact in key_facts:
        fact_terms = match_terms(fact)
        pool = pred_terms
        if not fact_terms:
            fact_terms = match_terms(fact, remove_stopwords=False)
            pool = pred_terms_all
        if not fact_terms:
            continue  # факт без единого слова (например, только знаки препинания)
        checked += 1
        if all(term in pool for term in fact_terms):
            found += 1
    return found / checked if checked else None


def _unique_chunk_ids(citations: Sequence[Citation]) -> list[str]:
    return list(dict.fromkeys(c.chunk_id for c in citations))


def citation_precision(
    citations: Sequence[Citation],
    chunks_by_id: Mapping[str, Chunk],
    spans: Sequence[Span],
    min_coverage: float = 0.5,
) -> float | None:
    """Доля процитированных фрагментов, которые действительно содержат доказательство.

    Низкое значение = модель ссылается «для вида». Нет ссылок или нет доказательств → None.
    """
    if not spans:
        return None
    cited_ids = _unique_chunk_ids(citations)
    if not cited_ids:
        return None
    relevant = 0
    for chunk_id in cited_ids:
        chunk = chunks_by_id.get(chunk_id)
        if chunk is not None and any(is_relevant(chunk, span, min_coverage) for span in spans):
            relevant += 1
    return relevant / len(cited_ids)


def citation_recall(
    citations: Sequence[Citation],
    chunks_by_id: Mapping[str, Chunk],
    spans: Sequence[Span],
    min_coverage: float = 0.5,
) -> float | None:
    """Доля доказательств, на которые в ответе есть ссылка. Нет доказательств → None."""
    if not spans:
        return None
    cited = [chunks_by_id[cid] for cid in _unique_chunk_ids(citations) if cid in chunks_by_id]
    covered = sum(1 for span in spans if any(is_relevant(chunk, span, min_coverage) for chunk in cited))
    return covered / len(spans)


# ---------------------------------------------------------------------------
# Отказы
# ---------------------------------------------------------------------------


def _safe_div(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def abstention_report(abstained: Sequence[bool], answerable: Sequence[bool]) -> dict[str, float | None]:
    """Качество детекции неотвечаемых вопросов.

    «Положительный» класс — вопрос без ответа в базе (на него правильно отказаться):

    * ``abstain_precision`` — из всех отказов какая доля была по делу;
    * ``abstain_recall`` — какую долю неотвечаемых вопросов система распознала;
    * ``abstain_f1`` — их гармоническое среднее;
    * ``false_abstention_rate`` — доля отвечаемых вопросов, на которые система зря отказалась.

    Деление на ноль даёт None («не определено»), а не 0: «ни разу не отказались»
    и «отказывались всегда неправильно» — разные ситуации.
    """
    if len(abstained) != len(answerable):
        raise ValueError("abstained и answerable должны быть одной длины")
    pairs = list(zip(abstained, answerable, strict=True))
    tp = sum(1 for a, ans in pairs if a and not ans)
    fp = sum(1 for a, ans in pairs if a and ans)
    fn = sum(1 for a, ans in pairs if not a and not ans)
    n_answerable = sum(1 for ans in answerable if ans)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "abstain_precision": precision,
        "abstain_recall": recall,
        "abstain_f1": f1,
        "false_abstention_rate": _safe_div(fp, n_answerable),
    }


def _as_score(value: float | None) -> float:
    # Нет уверенности (None/NaN) — считаем её минус бесконечностью: такой вопрос всегда отклоняется порогом.
    if value is None or math.isnan(value):
        return -math.inf
    return float(value)


def _balanced_accuracy(scores: Sequence[float], answerable: Sequence[bool], threshold: float) -> float:
    """Среднее из «доли отвеченных отвечаемых» и «доли отклонённых неотвечаемых»."""
    pairs = list(zip(scores, answerable, strict=True))
    answered_ok = sum(1 for s, ans in pairs if ans and s >= threshold)
    rejected_ok = sum(1 for s, ans in pairs if not ans and s < threshold)
    n_pos = sum(1 for ans in answerable if ans)
    n_neg = len(answerable) - n_pos
    return (answered_ok / n_pos + rejected_ok / n_neg) / 2


def choose_threshold(confidences: Sequence[float | None], answerable: Sequence[bool]) -> float | None:
    """Подбирает порог уверенности: отвечаем, если ``confidence >= threshold``, иначе отказ.

    Критерий — balanced accuracy: неотвечаемых вопросов обычно мало, и обычная
    точность была бы максимальна при «никогда не отказываться».
    Кандидаты — середины между соседними уникальными значениями уверенности
    (+ минимальное значение, если есть None: такой порог отсекает только None).
    При равенстве берётся меньший порог — система реже отказывается.

    Возвращает None, если порог не нужен: нет одного из классов (например, все
    вопросы отвечаемые) или никакой порог не лучше, чем «не отказываться вовсе»
    (у такой стратегии balanced accuracy ровно 0.5).
    """
    if len(confidences) != len(answerable):
        raise ValueError("confidences и answerable должны быть одной длины")
    n_pos = sum(1 for ans in answerable if ans)
    if n_pos == 0 or n_pos == len(answerable):
        return None

    scores = [_as_score(c) for c in confidences]
    finite = sorted({s for s in scores if math.isfinite(s)})
    candidates = [(low + high) / 2 for low, high in itertools.pairwise(finite)]
    if finite and any(s == -math.inf for s in scores):
        candidates.insert(0, finite[0])

    best_threshold: float | None = None
    best_score = 0.5  # «никогда не отказываться»: все отвечаемые отвечены, все неотвечаемые — нет
    for threshold in candidates:  # по возрастанию, поэтому при равенстве остаётся меньший
        score = _balanced_accuracy(scores, answerable, threshold)
        if score > best_score + 1e-12:
            best_threshold, best_score = threshold, score
    return best_threshold


# ---------------------------------------------------------------------------
# Агрегаты
# ---------------------------------------------------------------------------


def percentile(values: Sequence[float | None], q: float) -> float | None:
    """Перцентиль с линейной интерполяцией (как ``numpy.percentile``); None пропускаются.

    Для латентности смотрим p50 и p95, а не среднее: один медленный запрос
    сильно сдвигает среднее, но почти не влияет на медиану.
    """
    if not 0 <= q <= 100:
        raise ValueError(f"q должен быть в [0, 100], получено {q}")
    data = sorted(float(v) for v in values if v is not None)
    if not data:
        return None
    position = (len(data) - 1) * q / 100
    low = math.floor(position)
    high = math.ceil(position)
    return data[low] + (data[high] - data[low]) * (position - low)


def mean_ignore_none(values: Sequence[float | None]) -> float | None:
    """Среднее без None (None = «метрика к вопросу неприменима»); пусто → None."""
    present = [float(v) for v in values if v is not None]
    return sum(present) / len(present) if present else None
