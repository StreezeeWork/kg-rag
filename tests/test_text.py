"""Тесты общего анализатора текста (kbrag.text).

Анализатор используют BM25, extractive-генератор, метрики и нарезка, поэтому ошибка здесь
незаметно портит всё остальное. Тесты фиксируют поведение, на которое опираются другие модули.
"""

from __future__ import annotations

import sys
import threading

import pytest
import snowballstemmer

from kbrag.text import (
    STOPWORDS,
    analyze,
    find_span,
    normalize,
    split_sentences,
    stem,
    tokenize,
    word_spans,
)

# ---------------------------------------------------------------- normalize


def test_normalize_lowercases_and_replaces_yo() -> None:
    assert normalize("ЁЛКА и Ёж") == "елка и еж"


def test_normalize_collapses_whitespace_and_strips() -> None:
    assert normalize("  Пароль\n\tменяется   каждые   90 дней \n") == "пароль меняется каждые 90 дней"


def test_normalize_applies_nfkc() -> None:
    # NFKC сводит «визуально одинаковые» символы к одному виду: полноширинные цифры, лигатуры.
    assert normalize("１２ ﬁle") == "12 file"


def test_normalize_drops_number_sign() -> None:
    # Регрессия: NFKC превращал «№» в латинское «no», и в тексте появлялось слово «no».
    assert normalize("Приказ № 14-П") == "приказ 14-п"
    assert tokenize("договор №ПП-2025/0417") == ["договор", "пп-2025", "0417"]


def test_normalize_empty() -> None:
    assert normalize("") == ""
    assert normalize(" \n\t ") == ""


# ---------------------------------------------------------------- tokenize


def test_tokenize_keeps_codes_and_hostnames_whole() -> None:
    tokens = tokenize("Подключитесь к VPN-01 через help.nimbus.local.")
    assert "vpn-01" in tokens
    # Точка в конце предложения не входит в токен, а внутренние точки — входят.
    assert "help.nimbus.local" in tokens
    assert tokens == ["подключитесь", "к", "vpn-01", "через", "help.nimbus.local"]


def test_tokenize_drops_punctuation_and_normalizes() -> None:
    assert tokenize("«Нимбус.Люди» — HR-портал! Ёмкость: 2.5 ГБ?") == [
        "нимбус.люди",
        "hr-портал",
        "емкость",
        "2.5",
        "гб",
    ]


def test_tokenize_strips_dangling_hyphens_and_dots() -> None:
    assert tokenize("-abc- .def. --") == ["abc", "def"]


def test_tokenize_email_splits_on_at_sign() -> None:
    assert tokenize("security@nimbus.example") == ["security", "nimbus.example"]


def test_tokenize_empty() -> None:
    assert tokenize("") == []
    assert tokenize("... — !!!") == []


# ---------------------------------------------------------------- stem


@pytest.mark.parametrize(
    "forms", [("отпуск", "отпуска", "отпуском"), ("заявление", "заявления", "заявлением")]
)
def test_stem_merges_russian_word_forms(forms: tuple[str, ...]) -> None:
    assert len({stem(form) for form in forms}) == 1


def test_stem_keeps_system_name_orbita_together() -> None:
    # Snowball режет «Орбите» до «орб» (как глагол на «-ите»), остальные формы — до «орбит».
    forms = ("орбита", "орбите", "орбиту", "орбиты", "орбитой", "орбит")
    assert {stem(form) for form in forms} == {"орбит"}
    assert analyze("Заявка в «Орбите»") == analyze("заявка в Орбита")


def test_stem_uses_english_stemmer_for_latin() -> None:
    assert stem("connections") == stem("connection") == "connect"


@pytest.mark.parametrize("token", ["2025", "1111", "vpn-01", "2.5", "help.nimbus.local"])
def test_stem_keeps_numbers_and_codes(token: str) -> None:
    # Коды и числа стеммить нельзя: "vpn-01" и "vpn-02" должны остаться разными терминами.
    assert stem(token) == token


def test_stem_is_thread_safe() -> None:
    # Регрессия: /ask в API выполняется в пуле потоков, а объект snowballstemmer хранит
    # состояние разбора (current, cursor) в себе. Без блокировки параллельные вызовы портили
    # друг другу слово: возвращался стем чужого слова или падал IndexError.
    roots = ["отпуск", "заявлен", "руководител", "подразделен", "сотрудник", "компенсац", "договор", "проект"]
    prefixes = ["", "пере", "под", "вы", "при", "раз", "от", "за", "до", "на"]
    endings = ["а", "ом", "ами", "ах", "ов", "ей", "ям", "ями", "ию", "ия", "ого", "ому", "ыми", "ую", "ешь"]
    words = [p + r + e for p in prefixes for r in roots for e in endings]  # 1200 разных словоформ
    reference_stemmer = snowballstemmer.stemmer("russian")  # отдельный объект, только в этом потоке
    expected = {word: reference_stemmer.stemWord(word) for word in words}

    n_threads = 8
    barrier = threading.Barrier(n_threads)
    errors: list[Exception] = []
    wrong: list[tuple[str, str]] = []

    def worker(part: list[str]) -> None:
        barrier.wait()  # стартуем одновременно, чтобы вызовы действительно пересекались
        try:
            for word in part:
                result = stem(word)
                if result != expected[word]:
                    wrong.append((word, result))
        except Exception as exc:  # исключение в потоке иначе потерялось бы, а тест прошёл
            errors.append(exc)

    stem.cache_clear()  # иначе ответы придут из lru_cache и стеммер вообще не будет вызван
    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)  # частое переключение потоков: гонка воспроизводится надёжно
    try:
        threads = [threading.Thread(target=worker, args=(words[i::n_threads],)) for i in range(n_threads)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        sys.setswitchinterval(old_interval)

    assert not errors, errors[:3]
    assert not wrong, wrong[:5]


# ---------------------------------------------------------------- analyze


def test_analyze_removes_stopwords_and_stems() -> None:
    terms = analyze("Сколько дней в отпуске и как его оформить?")
    assert terms == [stem("сколько"), stem("дней"), stem("отпуске"), stem("оформить")]
    assert not set(tokenize("в и как его")) & set(terms)


def test_analyze_can_keep_stopwords() -> None:
    assert analyze("в отпуске", remove_stopwords=False) == [stem("в"), stem("отпуске")]
    assert analyze("в отпуске") == [stem("отпуске")]


def test_analyze_only_stopwords_gives_empty_list() -> None:
    assert analyze("и в на это the of") == []


def test_analyze_handles_yo_in_stopwords() -> None:
    # "её" после нормализации превращается в "ее" — оно тоже должно считаться стоп-словом.
    assert analyze("Её отпуск") == [stem("отпуск")]


def test_stopwords_are_lowercase() -> None:
    assert all(word == word.lower() for word in STOPWORDS)


@pytest.mark.parametrize(
    "word",
    ["один", "одна", "два", "две", "три", "четыре", "пять", "раз", "нет", "можно", "нельзя", "надо", "может"]
    + ["более", "менее", "больше", "меньше", "лучше"],
)
def test_fact_words_are_not_stopwords(word: str) -> None:
    # Регрессия: из списка NLTK в стоп-словах были «один», «два», «три», «нельзя», «более»…,
    # но не «две», «четыре», «менее». В базе знаний такие слова и есть факт.
    assert word not in STOPWORDS


def test_numerals_are_treated_alike() -> None:
    # Раньше «три» выбрасывалось как стоп-слово, а «пять» — нет.
    assert analyze("три дня") == [stem("три"), stem("дня")]
    assert analyze("пять дней") == [stem("пять"), stem("дней")]


def test_comparisons_are_kept() -> None:
    # «не более 6 часов» и «не менее 6 часов» — противоположные факты, их термы должны различаться.
    assert analyze("не более 6 часов") != analyze("не менее 6 часов")
    assert analyze("можно") != analyze("нельзя") != []


def test_compound_prepositions_are_stopwords() -> None:
    # Токенизатор держит «из-за» целиком; без записи в стоп-словах оно становилось термом «из-з».
    assert analyze("из-за болезни") == [stem("болезни")]
    assert analyze("из-под стола") == [stem("стола")]


# ---------------------------------------------------------------- analyze: составные токены


def test_analyze_keeps_compounds_whole_by_default() -> None:
    assert analyze("VPN-шлюза") == [stem("vpn-шлюза")]
    # Без разбиения «VPN-шлюза» и «шлюз VPN» не имеют ни одного общего терма.
    assert not set(analyze("VPN-шлюза")) & set(analyze("шлюз VPN"))


def test_split_compounds_mixed_matches_reordered_phrase() -> None:
    query = analyze("VPN-шлюза", split_compounds="mixed")
    assert query == [stem("vpn-шлюза"), stem("vpn"), stem("шлюза")]  # целый токен остаётся
    assert {stem("шлюз"), stem("vpn")} <= set(query) & set(analyze("шлюз VPN", split_compounds="mixed"))


def test_split_compounds_mixed_leaves_codes_and_hostnames_whole() -> None:
    # В «mixed» раскладываются только смеси латиницы и кириллицы: коды и адреса — как есть.
    text = "vpn-01 help.nimbus.local нимбус.люди 2.5"
    assert analyze(text, split_compounds="mixed") == analyze(text)


def test_split_compounds_all_adds_parts_of_any_compound() -> None:
    assert analyze("нимбус.люди", split_compounds="all") == [
        stem("нимбус.люди"),
        stem("нимбус"),
        stem("люди"),
    ]
    # Части проходят через тот же фильтр стоп-слов, что и обычные токены.
    assert analyze("из-за", split_compounds="all") == []


def test_split_compounds_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="split_compounds"):
        analyze("VPN-шлюз", split_compounds="hyphen")  # type: ignore[arg-type]


# ---------------------------------------------------------------- word_spans


def test_word_spans_point_to_non_space_runs() -> None:
    text = "  Пароль —\n12 символов. "
    spans = word_spans(text)
    assert [text[a:b] for a, b in spans] == ["Пароль", "—", "12", "символов."]


# ---------------------------------------------------------------- split_sentences


def _assert_valid_sentence_spans(text: str, spans: list[tuple[int, int]]) -> None:
    """Общие свойства: без пробелов по краям, по порядку, без перекрытий и без потерь."""
    previous_end = 0
    for start, end in spans:
        piece = text[start:end]
        assert piece, "пустое предложение"
        assert piece == piece.strip(), f"пробелы по краям: {piece!r}"
        assert start >= previous_end, "предложения перекрываются или идут не по порядку"
        previous_end = end
    # Между предложениями теряются только пробельные символы.
    joined = "".join(text[start:end] for start, end in spans)
    assert "".join(joined.split()) == "".join(text.split())


def test_split_sentences_basic() -> None:
    text = "Первое предложение. Второе! Третье? Четвёртое без точки"
    spans = split_sentences(text)
    assert [text[a:b] for a, b in spans] == [
        "Первое предложение.",
        "Второе!",
        "Третье?",
        "Четвёртое без точки",
    ]
    _assert_valid_sentence_spans(text, spans)


def test_split_sentences_returns_offsets_in_original_text() -> None:
    text = "  Текст первый.   Текст второй.  "
    spans = split_sentences(text)
    assert spans == [(2, 15), (18, 31)]


def test_split_sentences_does_not_split_numbers_or_lowercase() -> None:
    # Нет пробела после точки (2.5) или дальше строчная буква (т.е. ...) — это не граница.
    text = "Версия 2.5 вышла т.е. недавно. Далее."
    assert [text[a:b] for a, b in split_sentences(text)] == ["Версия 2.5 вышла т.е. недавно.", "Далее."]


def test_split_sentences_keeps_address_abbreviations() -> None:
    # «ул.» и «д.» перед заглавной буквой и цифрой — сокращения, а не конец предложения.
    text = "Офис находится на ул. Облачная, д. 7. Пропуск выдаёт охрана."
    assert [text[a:b] for a, b in split_sentences(text)] == [
        "Офис находится на ул. Облачная, д. 7.",
        "Пропуск выдаёт охрана.",
    ]


def test_split_sentences_keeps_reference_abbreviations() -> None:
    # Регрессия: «См. п. 3. Подробности ниже.» резалось на «См. п.» | «3.» | «Подробности ниже.».
    text = "См. п. 3. Подробности ниже."
    assert [text[a:b] for a, b in split_sentences(text)] == ["См. п. 3.", "Подробности ниже."]
    text = "Основание — ст. 81 ТК РФ. Схема на рис. 2 и в табл. 1. Далее."
    assert [text[a:b] for a, b in split_sentences(text)] == [
        "Основание — ст. 81 ТК РФ.",
        "Схема на рис. 2 и в табл. 1.",
        "Далее.",
    ]


def test_split_sentences_known_tradeoff_after_td_and_year() -> None:
    # Осознанный компромисс (см. комментарий к _ABBREVIATIONS): после «т. д.» и «г.» граница
    # не ставится, даже если предложение там действительно кончается.
    text = "Ноутбук, монитор и т. д. Далее идёт список. Введено в 2023 г. Новый абзац."
    assert [text[a:b] for a, b in split_sentences(text)] == [
        "Ноутбук, монитор и т. д. Далее идёт список.",
        "Введено в 2023 г. Новый абзац.",
    ]


def test_split_sentences_abbreviation_guard_needs_whole_word() -> None:
    # «под.» оканчивается на «д.», но это не сокращение «д.» — граница остаётся.
    text = "Всё идёт под. Новое предложение."
    assert [text[a:b] for a, b in split_sentences(text)] == ["Всё идёт под.", "Новое предложение."]


def test_split_sentences_treats_newline_as_boundary() -> None:
    text = "Список:\n- пункт один\n- пункт два\n\n| a | b |\n| 1 | 2 |"
    spans = split_sentences(text)
    assert [text[a:b] for a, b in spans] == [
        "Список:",
        "- пункт один",
        "- пункт два",
        "| a | b |",
        "| 1 | 2 |",
    ]
    _assert_valid_sentence_spans(text, spans)


def test_split_sentences_before_quote_and_digit() -> None:
    text = "Он сказал. «Привет». 28 дней. (Примечание)"
    assert [text[a:b] for a, b in split_sentences(text)] == [
        "Он сказал.",
        "«Привет».",
        "28 дней.",
        "(Примечание)",
    ]


@pytest.mark.parametrize("text", ["", "   ", "\n\n\t"])
def test_split_sentences_empty(text: str) -> None:
    assert split_sentences(text) == []


def test_split_sentences_on_markdown_document(mini_docs) -> None:
    for doc in mini_docs:
        _assert_valid_sentence_spans(doc.text, split_sentences(doc.text))


# ---------------------------------------------------------------- find_span


def test_find_span_ignores_whitespace_differences() -> None:
    haystack = "Пароль должен\nсодержать   не менее 12 символов и меняется каждые 90 дней."
    span = find_span(haystack, "должен содержать не\nменее 12 символов")
    assert span is not None
    start, end = span
    assert haystack[start:end] == "должен\nсодержать   не менее 12 символов"


def test_find_span_strips_needle_edges() -> None:
    haystack = "Отпуск составляет 28 календарных дней."
    start, end = find_span(haystack, "  28 календарных\tдней \n")
    assert haystack[start:end] == "28 календарных дней"


def test_find_span_escapes_regex_special_characters() -> None:
    haystack = "# Политика удалённой работы (редакция 2023)\nСумма: 1.5 * [x]"
    start, end = find_span(haystack, "работы (редакция 2023)")
    assert haystack[start:end] == "работы (редакция 2023)"
    assert find_span(haystack, "1.5 * [x]") is not None
    assert find_span(haystack, "1x5") is None  # точка в цитате — именно точка, а не «любой символ»


def test_find_span_returns_first_occurrence() -> None:
    assert find_span("abc abc", "abc") == (0, 3)


@pytest.mark.parametrize(
    "needle",
    [
        "29 календарных дней",  # другое число
        "28 рабочих дней",  # другое слово
        "28 КАЛЕНДАРНЫХ дней",  # регистр важен: цитата должна быть дословной
        "28календарных дней",  # пробел внутри цитаты нельзя «потерять»
    ],
)
def test_find_span_returns_none_when_no_match(needle: str) -> None:
    assert find_span("Отпуск составляет 28 календарных дней.", needle) is None


@pytest.mark.parametrize("needle", ["", "   ", "\n"])
def test_find_span_empty_needle(needle: str) -> None:
    assert find_span("любой текст", needle) is None


def test_find_span_finds_gold_quotes_in_mini_docs(mini_docs, mini_gold_records) -> None:
    docs_by_id = {doc.doc_id: doc for doc in mini_docs}
    for record in mini_gold_records:
        for evidence in record["evidence"]:
            text = docs_by_id[evidence["doc_id"]].text
            span = find_span(text, evidence["quote"])
            assert span is not None, record["id"]
            assert " ".join(text[span[0] : span[1]].split()) == " ".join(evidence["quote"].split())
