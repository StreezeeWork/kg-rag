"""Тесты метрик. Ожидаемые значения посчитаны руками — расчёт приведён в комментариях."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from kbrag.eval.metrics import (
    abstention_report,
    bootstrap_ci,
    choose_threshold,
    citation_precision,
    citation_recall,
    cites_stale_document,
    coverage_matrix,
    fact_recall,
    hit_at_k,
    match_terms,
    mean_ignore_none,
    mrr,
    ndcg_at_k,
    paired_bootstrap_diff,
    percentile,
    precision_at_k,
    recall_at_k,
    retrieval_metrics,
    soft_stem,
    strip_citations,
    token_f1,
)
from kbrag.eval.results import retrieval_metric_names
from kbrag.types import Chunk, Citation

LOG2_3 = math.log2(3)  # ≈ 1.585: знаменатель для ранга 2
LOG2_5 = math.log2(5)  # ≈ 2.322: знаменатель для ранга 4


def make_chunk(cid: str, start: int, end: int, doc_id: str = "d") -> Chunk:
    return Chunk(chunk_id=cid, doc_id=doc_id, text="x" * (end - start), start=start, end=end)


# Два доказательства в документе "d": E1 = [0, 100), E2 = [500, 600).
E1 = ("d", 0, 100)
E2 = ("d", 500, 600)
NOISE = make_chunk("noise", 200, 300)  # ничего не покрывает
C_E1 = make_chunk("e1", 0, 100)
C_E1_PART = make_chunk("e1-part", 40, 100)  # 60% от E1 — тоже покрывает E1
C_E2 = make_chunk("e2", 480, 620)


# ---------------------------------------------------------------------------
# Метрики поиска
# ---------------------------------------------------------------------------


def test_coverage_matrix_rows_are_ranks_columns_are_evidence() -> None:
    matrix = coverage_matrix([NOISE, C_E1, C_E2], [E1, E2])
    assert matrix == [[False, False], [True, False], [False, True]]


class TestTwoEvidenceQuestion:
    """Вопрос с двумя доказательствами (как multi_hop).

    Выдача: [шум, E1, кусок E1, E2] → матрица покрытия:
        ранг 1: [F, F]
        ранг 2: [T, F]
        ранг 3: [T, F]   ← E1 уже найдено, это НЕ новое доказательство
        ранг 4: [F, T]
    """

    coverage = coverage_matrix([NOISE, C_E1, C_E1_PART, C_E2], [E1, E2])

    def test_hit(self) -> None:
        assert hit_at_k(self.coverage, 1) == 0.0  # на первом месте шум
        assert hit_at_k(self.coverage, 3) == 1.0

    def test_recall_is_partial_until_both_found(self) -> None:
        assert recall_at_k(self.coverage, 1) == 0.0
        assert recall_at_k(self.coverage, 3) == 0.5  # найдено 1 из 2 доказательств
        assert recall_at_k(self.coverage, 4) == 1.0
        assert recall_at_k(self.coverage, 10) == 1.0  # k больше длины выдачи — не ошибка

    def test_precision_denominator_is_k(self) -> None:
        assert precision_at_k(self.coverage, 3) == pytest.approx(2 / 3)  # ранги 2 и 3 релевантны
        # Выдача из 4 фрагментов, но знаменатель всё равно 5: 3 релевантных / 5 = 0.6.
        assert precision_at_k(self.coverage, 5) == pytest.approx(0.6)

    def test_mrr(self) -> None:
        assert mrr(self.coverage) == 0.5  # первый релевантный на ранге 2

    def test_ndcg(self) -> None:
        # nDCG@3: польза только на ранге 2 (ранг 3 повторяет E1).
        #   DCG = 1/log2(3) ≈ 0.631; IDCG = 1 + 1/log2(3) ≈ 1.631 (идеал: два новых доказательства подряд)
        #   nDCG ≈ 0.631 / 1.631 ≈ 0.387
        assert ndcg_at_k(self.coverage, 3) == pytest.approx((1 / LOG2_3) / (1 + 1 / LOG2_3))
        assert ndcg_at_k(self.coverage, 3) == pytest.approx(0.387, abs=1e-3)
        # nDCG@5: польза на рангах 2 и 4: DCG = 1/log2(3) + 1/log2(5) ≈ 0.631 + 0.431 = 1.062
        #   IDCG по min(5, 2) = 2 позициям ≈ 1.631 → nDCG ≈ 0.651
        assert ndcg_at_k(self.coverage, 5) == pytest.approx((1 / LOG2_3 + 1 / LOG2_5) / (1 + 1 / LOG2_3))
        assert ndcg_at_k(self.coverage, 5) == pytest.approx(0.651, abs=1e-3)


def test_many_chunks_of_one_evidence_do_not_push_ndcg_above_one() -> None:
    # Три куска одной цитаты. Без правила «новое доказательство» DCG был бы
    # 1 + 1/log2(3) + 1/log2(4) ≈ 2.13 при IDCG = 1, т.е. nDCG ≈ 2.13.
    coverage = coverage_matrix([C_E1, C_E1_PART, make_chunk("e1-b", 0, 60)], [E1])
    assert ndcg_at_k(coverage, 3) == 1.0
    assert precision_at_k(coverage, 3) == 1.0  # для precision все три релевантны
    assert recall_at_k(coverage, 3) == 1.0


def test_ndcg_is_always_between_zero_and_one() -> None:
    # Перебор ВСЕХ выдач длины 4 над двумя доказательствами: 4 варианта строки ^ 4 ранга = 256 выдач.
    rows = [[False, False], [True, False], [False, True], [True, True]]
    for ranking in itertools.product(rows, repeat=4):
        coverage = [list(row) for row in ranking]
        for k in (1, 2, 3, 4, 5):
            assert 0.0 <= ndcg_at_k(coverage, k) <= 1.0 + 1e-12


def test_one_chunk_covering_both_evidence_is_a_known_ndcg_limitation() -> None:
    # Польза бинарная: один фрагмент с обеими цитатами даёт DCG = 1, а IDCG рассчитан на два
    # фрагмента (1 + 1/log2(3)). Поэтому nDCG < 1, хотя recall уже полный. Это задокументировано.
    both = make_chunk("both", 0, 600)
    coverage = coverage_matrix([both], [E1, E2])
    assert recall_at_k(coverage, 1) == 1.0
    assert ndcg_at_k(coverage, 5) == pytest.approx(1 / (1 + 1 / LOG2_3))


def test_no_relevant_chunks_gives_zero_everywhere() -> None:
    coverage = coverage_matrix([NOISE, make_chunk("other-doc", 0, 100, doc_id="x")], [E1])
    assert mrr(coverage) == 0.0
    assert hit_at_k(coverage, 5) == 0.0
    assert recall_at_k(coverage, 5) == 0.0
    assert precision_at_k(coverage, 5) == 0.0
    assert ndcg_at_k(coverage, 5) == 0.0


def test_empty_retrieval_gives_zero_everywhere() -> None:
    assert retrieval_metrics([], [E1, E2], k_values=(1, 5)) == {
        "hit@1": 0.0,
        "hit@5": 0.0,
        "recall@1": 0.0,
        "recall@5": 0.0,
        "precision@1": 0.0,
        "precision@5": 0.0,
        "ndcg@1": 0.0,
        "ndcg@5": 0.0,
        "mrr": 0.0,
    }


def test_retrieval_metrics_uses_report_names_and_matches_single_functions() -> None:
    chunks = [NOISE, C_E1, C_E1_PART, C_E2]
    result = retrieval_metrics(chunks, [E1, E2])

    assert list(result) == retrieval_metric_names()  # те же имена и порядок, что в отчёте
    assert result["recall@3"] == 0.5
    assert result["mrr"] == 0.5
    assert result["precision@5"] == pytest.approx(0.6)


def test_retrieval_metrics_reject_questions_without_evidence() -> None:
    with pytest.raises(ValueError, match="evidence"):
        retrieval_metrics([C_E1], [])
    with pytest.raises(ValueError, match="evidence"):
        recall_at_k([[]], 5)  # строка есть, а доказательств (столбцов) нет


@pytest.mark.parametrize("metric", [hit_at_k, recall_at_k, precision_at_k, ndcg_at_k])
def test_k_must_be_positive(metric) -> None:
    with pytest.raises(ValueError):
        metric([[True]], 0)


# ---------------------------------------------------------------------------
# Метрики генерации
# ---------------------------------------------------------------------------


def test_soft_stem_unifies_short_russian_forms_but_not_numbers_or_latin() -> None:
    # Snowball: «дней» → «дне», «днях» → «днях» (основа короткая, окончание не срезано).
    assert soft_stem("дне") == soft_stem("днях") == soft_stem("дня") == "дн"
    assert soft_stem("28") == "28"
    assert soft_stem("vpn") == "vpn"
    assert soft_stem("я") == "я"  # слишком короткий остаток — основа не меняется
    assert match_terms("28 календарных дней") == match_terms("28 календарных днях")


def test_strip_citations_removes_reference_markers() -> None:
    assert strip_citations("До 3 дней [1, 2] в неделю [3].").split() == [
        "До",
        "3",
        "дней",
        "в",
        "неделю",
        ".",
    ]


def test_token_f1_word_forms_match() -> None:
    assert token_f1("28 календарных днях", "28 календарных дней") == 1.0


def test_token_f1_partial_overlap() -> None:
    # pred → [отпуск, длит, 28, календарн, дн] (5 термов), gold → [28, календарн, дн] (3 терма)
    # общих 3: precision = 3/5, recall = 3/3, F1 = 2·0.6·1 / 1.6 = 0.75
    assert token_f1("Отпуск длится 28 календарных дней.", "28 календарных дней") == pytest.approx(0.75)


def test_token_f1_counts_repeated_terms_once_per_occurrence() -> None:
    # pred: [шлюз, шлюз], gold: [шлюз] → общих 1: P = 1/2, R = 1 → F1 = 2/3
    assert token_f1("Шлюз, шлюз", "Шлюз") == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    ("pred", "gold", "expected"),
    [
        ("", "", 1.0),
        ("", "28 дней", 0.0),
        ("28 дней", "", 0.0),
        ("Клиент Шлюз", "14 дней", 0.0),
        ("28 дней [1]", "28 дней", 1.0),  # маркер ссылки не считается лишним словом
    ],
)
def test_token_f1_edge_cases(pred: str, gold: str, expected: float) -> None:
    assert token_f1(pred, gold) == expected


@pytest.mark.parametrize(
    ("pred", "facts", "expected"),
    [
        ("Отпуск — 28 календарных днях.", ["28 календарных дней"], 1.0),  # другая словоформа
        ("Отпуск — 14 календарных дней.", ["28 календарных дней"], 0.0),  # слова те же, число другое
        ("28 календарных дней, через «Нимбус.Люди»", ["28 календарных дней", "14 дней"], 0.5),
        ("Можно 2 дня в неделю [3].", ["3 дней"], 0.0),  # «3» из ссылки [3] — не факт
        ("Нет, нельзя.", ["нет"], 1.0),  # факт только из стоп-слова
        ("Используется клиент «Шлюз».", ["Шлюз"], 1.0),
    ],
)
def test_fact_recall(pred: str, facts: list[str], expected: float) -> None:
    assert fact_recall(pred, facts) == expected


@pytest.mark.parametrize(
    ("pred", "facts"),
    [
        # Регрессия: стоп-слово «не» выбрасывалось, и ответ с обратным смыслом получал 1.0.
        ("Переработки оплачиваются в двойном размере [1].", ["не оплачиваются"]),
        ("Да, SMS-коды допускаются как второй фактор.", ["не допускаются"]),
        ("Сотрудник подвергается дисциплинарному взысканию.", ["не подвергается дисциплинарному взысканию"]),
        ("Командировка длится не менее 2 дней.", ["не более 2 дней"]),
        ("Командировка длится более 2 дней.", ["не более 2 дней"]),
        # Время и разряды чисел раньше распадались на отдельные термы.
        ("Сообщить нужно к 9:00, срок 10 дней.", ["до 10:00"]),
        ("Выплата 500 рублей, 1 раз в месяц.", ["1 500 рублей"]),
        ("Премия 20% оклада, отпуск 10 дней.", ["10%"]),
        # soft_stem склеивал «приём» и «премии» в одну основу «пр».
        ("Премии выплачиваются ежеквартально.", ["приём"]),
    ],
)
def test_fact_recall_rejects_answers_with_opposite_meaning(pred: str, facts: list[str]) -> None:
    assert fact_recall(pred, facts) == 0.0


@pytest.mark.parametrize(
    ("pred", "facts"),
    [
        ("Сверхурочные часы не оплачиваются [1].", ["не оплачиваются"]),
        (
            "Дисциплинарному взысканию сотрудник не подвергается.",
            ["не подвергается дисциплинарному взысканию"],
        ),
        ("Компенсация — 60000 рублей.", ["60 000 рублей"]),
        ("Компенсация — 1 500 рублей.", ["1 500 рублей"]),  # неразрывный пробел между разрядами
        ("Начало смены в 8:00.", ["08:00"]),
        ("Целевой размер — 10 % от суммы окладов.", ["10%", "суммы окладов"]),
        ("Ставка — 10 процентов.", ["10%"]),
        ("Отпуск без сохранения заработной платы.", ["без сохранения заработной платы"]),
    ],
)
def test_fact_recall_keeps_correct_rephrasings(pred: str, facts: list[str]) -> None:
    assert fact_recall(pred, facts) == 1.0


def test_match_terms_binds_negation_and_keeps_numbers_whole() -> None:
    assert match_terms("не более 2 дней") == ["не_бол", "2", "дн"]  # «не» + стем следующего слова
    assert match_terms("с 9:00 до 18:00") == ["09h00", "18h00"]
    assert match_terms("1 500 рублей") == match_terms("1500 рублей")
    assert soft_stem("прием") == "прием" and soft_stem("прем") == "прем"  # нет склейки в «пр»


def test_fact_recall_without_facts_is_not_applicable() -> None:
    assert fact_recall("любой ответ", []) is None
    assert fact_recall("любой ответ", ["!!!"]) is None  # «факт» без слов игнорируется


def _citation(index: int, chunk: Chunk) -> Citation:
    return Citation(index=index, chunk_id=chunk.chunk_id, doc_id=chunk.doc_id, title="")


CHUNKS_BY_ID = {c.chunk_id: c for c in (NOISE, C_E1, C_E2)}


def test_citation_precision_and_recall() -> None:
    citations = [_citation(1, C_E1), _citation(2, NOISE)]
    # Из 2 ссылок релевантна 1 → precision 0.5; из 2 доказательств процитировано 1 → recall 0.5.
    assert citation_precision(citations, CHUNKS_BY_ID, [E1, E2]) == 0.5
    assert citation_recall(citations, CHUNKS_BY_ID, [E1, E2]) == 0.5

    full = [_citation(1, C_E1), _citation(2, C_E2)]
    assert citation_precision(full, CHUNKS_BY_ID, [E1, E2]) == 1.0
    assert citation_recall(full, CHUNKS_BY_ID, [E1, E2]) == 1.0


def test_cites_stale_document() -> None:
    old = Chunk(chunk_id="old::0", doc_id="old", text="x", start=0, end=1, metadata={"status": "archived"})
    new = Chunk(chunk_id="new::0", doc_id="new", text="x", start=0, end=1, metadata={"status": "active"})
    chunks = {c.chunk_id: c for c in (old, new)}
    cite_old, cite_new = _citation(1, old), _citation(2, new)

    assert cites_stale_document([cite_new, cite_old], chunks, {"new"})  # архив вне эталона — ошибка
    assert not cites_stale_document([cite_new], chunks, {"new"})
    assert not cites_stale_document([cite_old], chunks, {"old"})  # вопрос про старую редакцию — ок
    assert not cites_stale_document([], chunks, {"new"})


def test_citation_metrics_edge_cases() -> None:
    assert citation_precision([], CHUNKS_BY_ID, [E1]) is None  # нет ссылок — precision не определена
    assert citation_recall([], CHUNKS_BY_ID, [E1]) == 0.0  # а recall — ноль
    assert citation_precision([_citation(1, C_E1)], CHUNKS_BY_ID, []) is None  # вопрос без доказательств
    assert citation_recall([_citation(1, C_E1)], CHUNKS_BY_ID, []) is None
    # Повторная ссылка на тот же фрагмент не накручивает precision.
    assert (
        citation_precision([_citation(1, C_E1), _citation(1, C_E1), _citation(2, NOISE)], CHUNKS_BY_ID, [E1])
        == 0.5
    )
    # Ссылка на неизвестный фрагмент считается нерелевантной.
    ghost = Citation(index=3, chunk_id="ghost", doc_id="d", title="")
    assert citation_precision([_citation(1, C_E1), ghost], CHUNKS_BY_ID, [E1]) == 0.5


# ---------------------------------------------------------------------------
# Отказы
# ---------------------------------------------------------------------------


def test_abstention_report_hand_computed() -> None:
    abstained = [True, True, False, False, True]
    answerable = [False, True, False, True, True]
    # TP (отказ на неотвечаемом) = 1 (#0); FP (отказ на отвечаемом) = 2 (#1, #4); FN (ответ на неотвечаемый) = 1 (#2)
    # precision = 1/3, recall = 1/2, F1 = 2·(1/3)·(1/2) / (5/6) = 0.4; ложные отказы = 2 из 3 отвечаемых
    report = abstention_report(abstained, answerable)
    assert report["abstain_precision"] == pytest.approx(1 / 3)
    assert report["abstain_recall"] == pytest.approx(0.5)
    assert report["abstain_f1"] == pytest.approx(0.4)
    assert report["false_abstention_rate"] == pytest.approx(2 / 3)


def test_abstention_report_division_by_zero_gives_none() -> None:
    # Ни одного отказа: precision не определена, recall = 0 (неотвечаемый вопрос пропущен).
    never = abstention_report([False, False], [True, False])
    assert never["abstain_precision"] is None
    assert never["abstain_recall"] == 0.0
    assert never["abstain_f1"] is None
    assert never["false_abstention_rate"] == 0.0

    # Только отвечаемые вопросы: recall не определён.
    only_answerable = abstention_report([True, False], [True, True])
    assert only_answerable["abstain_recall"] is None
    assert only_answerable["abstain_precision"] == 0.0
    assert only_answerable["false_abstention_rate"] == 0.5

    # Только неотвечаемые: доля ложных отказов не определена.
    assert abstention_report([True], [False])["false_abstention_rate"] is None


def test_abstention_report_f1_is_zero_when_precision_and_recall_are_zero() -> None:
    report = abstention_report([True, False], [True, False])  # отказались не там, ответили не там
    assert report["abstain_precision"] == 0.0
    assert report["abstain_recall"] == 0.0
    assert report["abstain_f1"] == 0.0


def test_abstention_report_length_mismatch() -> None:
    with pytest.raises(ValueError):
        abstention_report([True], [True, False])


def _accept(confidences, threshold):
    return [c is not None and c >= threshold for c in confidences]


def test_choose_threshold_perfectly_separable() -> None:
    confidences = [0.9, 0.7, 0.2, 0.3]
    answerable = [True, True, False, False]
    # Уникальные: 0.2, 0.3, 0.7, 0.9 → кандидаты 0.25, 0.5, 0.8; при 0.5 всё разделено идеально.
    threshold = choose_threshold(confidences, answerable)
    assert threshold == pytest.approx(0.5)
    assert _accept(confidences, threshold) == answerable


def test_choose_threshold_not_separable() -> None:
    confidences = [0.9, 0.8, 0.4, 0.5, 0.1]
    answerable = [True, True, True, False, False]
    # Кандидаты (balanced accuracy = (доля отвеченных отвечаемых + доля отклонённых неотвечаемых) / 2):
    #   0.25 → (3/3 + 1/2) / 2 = 0.750
    #   0.45 → (2/3 + 1/2) / 2 = 0.583
    #   0.65 → (2/3 + 2/2) / 2 = 0.833  ← лучший
    #   0.85 → (1/3 + 2/2) / 2 = 0.667
    assert choose_threshold(confidences, answerable) == pytest.approx(0.65)


def test_choose_threshold_returns_none_when_no_threshold_helps() -> None:
    # Уверенность никак не связана с отвечаемостью: лучший порог 0.5 даёт BA = 0.5 —
    # не лучше, чем не отказываться вовсе.
    assert choose_threshold([0.2, 0.8, 0.2, 0.8], [True, True, False, False]) is None


def test_choose_threshold_treats_none_as_minus_infinity() -> None:
    # Для неотвечаемых поиск ничего не вернул (уверенность None) — их должен отсечь любой порог,
    # а все отвечаемые должны пройти. Кандидаты: 0.6 (минимум, т.к. есть None) и 0.7.
    confidences = [0.6, 0.8, None, None]
    answerable = [True, True, False, False]
    threshold = choose_threshold(confidences, answerable)
    assert threshold == pytest.approx(0.6)
    assert _accept(confidences, threshold) == answerable


def test_choose_threshold_with_none_among_answerable() -> None:
    # Кандидаты 0.3 и 0.6: 0.3 → (1/2 + 0) / 2 = 0.25; 0.6 → (1/2 + 1) / 2 = 0.75.
    assert choose_threshold([None, 0.9, 0.3], [True, True, False]) == pytest.approx(0.6)


def test_choose_threshold_prefers_lower_threshold_on_tie() -> None:
    # 0.3 → (2/2 + 1/2) / 2 = 0.75; 0.6 → 0.5; 0.8 → (1/2 + 2/2) / 2 = 0.75 → равенство, берём меньший.
    assert choose_threshold([0.5, 0.9, 0.1, 0.7], [True, True, False, False]) == pytest.approx(0.3)


@pytest.mark.parametrize(
    ("confidences", "answerable"),
    [
        ([0.1, 0.5, 0.9], [True, True, True]),  # всё отвечаемое — отказываться не нужно
        ([0.1, 0.5], [False, False]),  # не на чем откалибровать «отвечать»
        ([None, None], [True, False]),  # нет ни одного числа
        ([], []),
    ],
)
def test_choose_threshold_degenerate_inputs_give_none(confidences, answerable) -> None:
    assert choose_threshold(confidences, answerable) is None


def test_choose_threshold_is_deterministic_and_order_independent() -> None:
    confidences = [0.9, 0.8, 0.4, 0.5, 0.1, None, 0.45]
    answerable = [True, True, True, False, False, False, True]
    expected = choose_threshold(confidences, answerable)
    assert choose_threshold(confidences, answerable) == expected
    order = [3, 0, 6, 5, 1, 4, 2]
    assert choose_threshold([confidences[i] for i in order], [answerable[i] for i in order]) == expected


def test_choose_threshold_length_mismatch() -> None:
    with pytest.raises(ValueError):
        choose_threshold([0.1], [True, False])


# ---------------------------------------------------------------------------
# Агрегаты
# ---------------------------------------------------------------------------


def test_percentile_matches_numpy_linear_interpolation() -> None:
    values = [12.0, 3.0, 7.5, 40.0, 1.0, 9.0]
    for q in (0, 25, 50, 95, 100):
        assert percentile(values, q) == pytest.approx(float(np.percentile(values, q)))
    # Руками: [1, 2, 3, 4], p95 → позиция 0.95·3 = 2.85 → 3 + 0.85·(4 − 3) = 3.85
    assert percentile([4, 1, 3, 2], 95) == pytest.approx(3.85)


def test_percentile_edge_cases() -> None:
    assert percentile([], 50) is None
    assert percentile([None, 5.0], 50) == 5.0
    with pytest.raises(ValueError):
        percentile([1.0], 101)


def test_mean_ignore_none() -> None:
    assert mean_ignore_none([1.0, None, 3.0]) == 2.0
    assert mean_ignore_none([None, None]) is None
    assert mean_ignore_none([]) is None
    assert mean_ignore_none([0.0]) == 0.0  # ноль — это значение, а не «нет данных»


# ---------------------------------------------------------------------------
# Доверительные интервалы
# ---------------------------------------------------------------------------


def test_bootstrap_ci_is_deterministic_and_contains_the_mean() -> None:
    values = [1.0] * 100 + [0.0] * 24 + [None]  # Recall@5 ≈ 0.806 на 124 вопросах
    interval = bootstrap_ci(values)
    assert interval == bootstrap_ci(values)  # seed фиксирован — отчёт воспроизводим
    low, high = interval
    assert low < 100 / 124 < high
    assert 0.05 < high - low < 0.2  # на ~120 вопросах интервал шириной около ±0.07


def test_bootstrap_ci_edge_cases() -> None:
    assert bootstrap_ci([]) is None
    assert bootstrap_ci([None, None]) is None
    assert bootstrap_ci([0.5, 0.5, 0.5]) == (0.5, 0.5)  # нет разброса — интервал вырожден


def test_paired_bootstrap_diff_identical_and_shifted() -> None:
    a = [1.0, 0.0, 0.5, 1.0, 0.0]
    assert paired_bootstrap_diff(a, a) == (0.0, 0.0, 0.0)
    shifted = [x + 0.25 for x in a]
    assert paired_bootstrap_diff(shifted, a) == pytest.approx((0.25, 0.25, 0.25))
    assert paired_bootstrap_diff([None, 1.0], [0.0, None]) is None  # нет общих вопросов
    with pytest.raises(ValueError):
        paired_bootstrap_diff([1.0], [1.0, 0.0])


def test_paired_bootstrap_diff_one_question_out_of_many_is_noise() -> None:
    # Регрессия к находке «нет значимости»: 0.863 против 0.855 — это 1 вопрос из 124.
    b = [1.0] * 106 + [0.0] * 18
    a = [1.0] * 107 + [0.0] * 17  # на один решённый вопрос больше
    diff, low, high = paired_bootstrap_diff(a, b)
    assert diff == pytest.approx(1 / 124)
    assert low <= 0 <= high  # интервал включает 0 — «победа» в пределах шума
    much_better = [1.0] * 124
    _, low, _ = paired_bootstrap_diff(much_better, b)
    assert low > 0  # 18 решённых вопросов из 124 — уже не шум
