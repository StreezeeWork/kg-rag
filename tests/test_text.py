"""Тесты общего анализатора текста (kbrag.text).

Анализатор используют BM25, extractive-генератор, метрики и нарезка, поэтому ошибка здесь
незаметно портит всё остальное. Тесты фиксируют поведение, на которое опираются другие модули.
"""

from __future__ import annotations

import pytest

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


def test_stem_uses_english_stemmer_for_latin() -> None:
    assert stem("connections") == stem("connection") == "connect"


@pytest.mark.parametrize("token", ["2025", "1111", "vpn-01", "2.5", "help.nimbus.local"])
def test_stem_keeps_numbers_and_codes(token: str) -> None:
    # Коды и числа стеммить нельзя: "vpn-01" и "vpn-02" должны остаться разными терминами.
    assert stem(token) == token


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
