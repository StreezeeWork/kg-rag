"""Тесты офлайн extractive-генератора: выбор предложений, ссылки, отказы, чистка markdown."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from kbrag.generation.base import GenerationResult, Generator
from kbrag.generation.extractive import ExtractiveGenerator, clean_markdown
from kbrag.generation.prompts import ABSTAIN_TEXT
from kbrag.types import Chunk, Document, ScoredChunk

_PARAGRAPH_RE = re.compile(r"[^\n]+(?:\n[^\n]+)*")
_CITATION_RE = re.compile(r"\[\d+\]")
VACATION_QUESTION = "Сколько календарных дней длится ежегодный отпуск?"
VACATION_SENTENCE = "Продолжительность ежегодного оплачиваемого отпуска составляет 28 календарных дней."


def make_context(docs: list[Document], doc_id: str, needle: str) -> ScoredChunk:
    """Фрагмент = абзац мини-документа, содержащий ``needle`` (чанкер здесь не нужен)."""
    doc = next(d for d in docs if d.doc_id == doc_id)
    for i, match in enumerate(_PARAGRAPH_RE.finditer(doc.text)):
        if needle in match.group():
            chunk = Chunk(
                chunk_id=f"{doc_id}::{i:03d}", doc_id=doc_id, text=match.group(),
                start=match.start(), end=match.end(), title=doc.title, metadata=dict(doc.metadata),
            )
            return ScoredChunk(chunk=chunk, score=1.0, source="bm25")
    raise AssertionError(f"абзац с {needle!r} не найден в {doc_id}")


def make_chunk(text: str, chunk_id: str = "doc::000") -> ScoredChunk:
    chunk = Chunk(chunk_id=chunk_id, doc_id=chunk_id.split("::")[0], text=text, start=0, end=len(text))
    return ScoredChunk(chunk=chunk, score=1.0, source="bm25")


@pytest.fixture()
def vacation_contexts(mini_docs: list[Document]) -> list[ScoredChunk]:
    return [
        make_context(mini_docs, "hr-vacation", "28 календарных"),
        make_context(mini_docs, "hr-vacation", "Заявление на отпуск"),
    ]


def test_is_a_generator() -> None:
    generator = ExtractiveGenerator()
    assert isinstance(generator, Generator)
    assert generator.name == "extractive"


def test_answers_vacation_question_with_citation(vacation_contexts: list[ScoredChunk]) -> None:
    result = ExtractiveGenerator().generate(VACATION_QUESTION, vacation_contexts)

    assert isinstance(result, GenerationResult)
    assert not result.abstained
    assert "28 календарных дней" in result.text
    # Лучшее предложение идёт первым и сразу снабжено ссылкой на свой фрагмент.
    assert result.text.startswith(f"{VACATION_SENTENCE} [1]")
    assert result.usage == {}


def test_is_deterministic(vacation_contexts: list[ScoredChunk]) -> None:
    generator = ExtractiveGenerator()
    answers = {generator.generate(VACATION_QUESTION, vacation_contexts).text for _ in range(3)}
    assert len(answers) == 1


def test_citation_number_is_rank_of_source_fragment(mini_docs: list[Document]) -> None:
    contexts = [
        make_context(mini_docs, "it-vpn", "клиент «Шлюз»"),
        make_context(mini_docs, "hr-vacation", "28 календарных"),
    ]
    result = ExtractiveGenerator(max_sentences=1).generate(VACATION_QUESTION, contexts)
    assert result.text == f"{VACATION_SENTENCE} [2]"


def test_every_sentence_has_a_citation(vacation_contexts: list[ScoredChunk]) -> None:
    text = ExtractiveGenerator(max_sentences=3).generate(VACATION_QUESTION, vacation_contexts).text
    sentences_with_refs = re.findall(r"[^\[\]]+?\s\[\d+\]", text)
    assert len(sentences_with_refs) == len(_CITATION_RE.findall(text)) == 3


def test_respects_max_sentences(vacation_contexts: list[ScoredChunk]) -> None:
    text = ExtractiveGenerator(max_sentences=1).generate(VACATION_QUESTION, vacation_contexts).text
    assert text == f"{VACATION_SENTENCE} [1]"


def test_min_overlap_filters_weak_sentences(vacation_contexts: list[ScoredChunk]) -> None:
    # У лучшего предложения 4 общих стема с вопросом (ежегодн, отпуск, календарн, дне).
    strict = ExtractiveGenerator(min_overlap=4).generate(VACATION_QUESTION, vacation_contexts)
    assert strict.text == f"{VACATION_SENTENCE} [1]"

    too_strict = ExtractiveGenerator(min_overlap=5).generate(VACATION_QUESTION, vacation_contexts)
    assert too_strict.abstained and too_strict.text == ABSTAIN_TEXT


def test_abstains_when_nothing_overlaps(mini_docs: list[Document]) -> None:
    contexts = [
        make_context(mini_docs, "it-vpn", "клиент «Шлюз»"),
        make_context(mini_docs, "it-vpn", "Пароль должен"),
    ]
    result = ExtractiveGenerator().generate("Сколько стоит обед в столовой казанского офиса?", contexts)

    assert result.abstained
    assert result.text == ABSTAIN_TEXT


def test_abstains_without_contexts() -> None:
    result = ExtractiveGenerator().generate(VACATION_QUESTION, [])
    assert result.abstained and result.text == ABSTAIN_TEXT


def test_abstains_when_question_has_only_stopwords(vacation_contexts: list[ScoredChunk]) -> None:
    result = ExtractiveGenerator().generate("И что это?", vacation_contexts)
    assert result.abstained


def test_heading_lines_are_never_used(mini_docs: list[Document]) -> None:
    # Заголовок «# Ежегодный оплачиваемый отпуск» пересекается с вопросом, но ответом не является.
    heading_only = make_context(mini_docs, "hr-vacation", "# Ежегодный")
    assert ExtractiveGenerator().generate(VACATION_QUESTION, [heading_only]).abstained

    # Нумерованный заголовок делится на «## 2.» и «Порядок … отпуска» — вторая часть тоже не ответ.
    with_heading = make_chunk(f"## 2. Порядок предоставления ежегодного отпуска\n\n{VACATION_SENTENCE}")
    result = ExtractiveGenerator(max_sentences=3).generate(VACATION_QUESTION, [with_heading])
    assert result.text == f"{VACATION_SENTENCE} [1]"
    assert "#" not in result.text and "Порядок" not in result.text


def test_short_sentences_are_skipped() -> None:
    result = ExtractiveGenerator().generate(VACATION_QUESTION, [make_chunk("Отпуск длится.\nОтпуск.")])
    assert result.abstained


def test_markdown_is_cleaned_in_answer() -> None:
    chunk = make_chunk(
        "- **Отпуск** составляет 28 календарных дней\n"
        "| Ежегодный отпуск | 28 календарных дней |\n"
        "|---|---|\n"
        "* Отпуск делится на части по 14 календарных дней"
    )
    text = ExtractiveGenerator(max_sentences=3).generate(VACATION_QUESTION, [chunk]).text

    for garbage in ("**", "|", "- ", "* ", "---"):
        assert garbage not in text
    assert "Ежегодный отпуск — 28 календарных дней [1]" in text
    assert "Отпуск составляет 28 календарных дней [1]" in text
    assert "Отпуск делится на части по 14 календарных дней [1]" in text


def test_duplicate_sentences_from_overlapping_chunks_are_merged() -> None:
    contexts = [make_chunk(VACATION_SENTENCE, "a::000"), make_chunk(VACATION_SENTENCE, "a::001")]
    result = ExtractiveGenerator().generate(VACATION_QUESTION, contexts)
    assert result.text == f"{VACATION_SENTENCE} [1]"


def test_ties_are_broken_by_fragment_rank() -> None:
    contexts = [make_chunk("Отпуск согласует руководитель.", "b::000"), make_chunk("Отпуск оформляют заранее.", "a::000")]
    result = ExtractiveGenerator().generate("Что известно про отпуск?", contexts)
    assert result.text == "Отпуск согласует руководитель. [1] Отпуск оформляют заранее. [2]"


# --- Пункты списка не отрываются от вводной фразы (иначе запрет читается как совет) ---


def test_list_item_gets_section_heading_as_context() -> None:
    chunk = make_chunk(
        "## 4. Что запрещено\n\n"
        "- устанавливать средства удалённого управления (TeamViewer, AnyDesk и аналоги);\n"
        "- запускать программы для майнинга криптовалют."
    )
    text = ExtractiveGenerator().generate("Можно ли установить AnyDesk или TeamViewer?", [chunk]).text

    assert text.startswith("Что запрещено: устанавливать средства удалённого управления")
    assert "4." not in text  # номер раздела контекстом не считается


def test_list_item_takes_heading_from_section_when_chunk_starts_mid_list() -> None:
    # Фрагмент начался посреди списка: строки «## …» в нём нет, заголовок берём из Chunk.section.
    text = "- устанавливать TeamViewer и AnyDesk на ноутбук;"
    chunk = Chunk(
        chunk_id="it::001",
        doc_id="it",
        text=text,
        start=0,
        end=len(text),
        title="Установка ПО",
        section="Правила > 4. Что запрещено",
    )
    context = ScoredChunk(chunk=chunk, score=1.0, source="bm25")
    result = ExtractiveGenerator().generate("Можно ли установить AnyDesk?", [context])
    assert result.text == "Что запрещено: устанавливать TeamViewer и AnyDesk на ноутбук; [1]"


def test_list_item_prefers_lead_in_phrase_and_lead_in_is_not_an_answer() -> None:
    chunk = make_chunk(
        "## 5. Действия при утечке\n\n"
        "Сканер блокирует MR. Если секрет попал в репозиторий или чат:\n\n"
        "1. Немедленно отзовите и замените секрет в Vault.\n"
        "2. Сообщите об утечке в службу ИБ."
    )
    text = ExtractiveGenerator().generate("Что делать, если секрет попал в чат?", [chunk]).text

    lead_in = "Если секрет попал в репозиторий или чат"
    assert f"{lead_in}: Немедленно отзовите и замените секрет в Vault. [1]" in text
    assert f"{lead_in}: Сообщите об утечке в службу ИБ. [1]" in text
    assert f"{lead_in}: [1]" not in text  # вводная сама по себе не ответ


def test_lead_in_applies_only_to_the_list_right_after_it() -> None:
    chunk = make_chunk(
        "Порядок рассмотрения заявки:\n"
        "1. Руководитель подтверждает заявку на программу.\n"
        "После отказа заявку на программу можно подать через 3 месяца.\n"
        "- Заявку на программу проверяет служба ИБ."
    )
    text = ExtractiveGenerator(max_sentences=3).generate("Как подать заявку на программу?", [chunk]).text

    assert "Порядок рассмотрения заявки: Руководитель подтверждает" in text
    assert "После отказа заявку" in text and "заявки: После" not in text
    # Абзац между ними закрыл список: у второго списка вводной нет, а заголовка во фрагменте нет.
    assert "Заявку на программу проверяет служба ИБ. [1]" in text


def test_real_prohibition_is_never_shown_as_advice() -> None:
    """Регрессия eng-028 / it-006 на настоящих документах: пункт из «Что запрещено» — только с контекстом."""
    from kbrag.config import load_config
    from kbrag.ingest import chunk_corpus, load_corpus, make_chunker

    root = Path(__file__).resolve().parents[1]
    docs = [d for d in load_corpus(root / "data" / "corpus") if d.doc_id in {"eng-secrets", "it-software"}]
    chunks = chunk_corpus(docs, make_chunker(load_config(root / "configs" / "offline.yaml").chunking))
    contexts = [ScoredChunk(chunk=chunk, score=1.0, source="bm25") for chunk in chunks]
    generator = ExtractiveGenerator(max_sentences=len(chunks) * 10)  # смотрим на ВСЕ предложения-кандидаты

    secret_question = "Что делать, если секрет случайно попал в репозиторий или в чат?"
    anydesk_question = "Можно ли установить на корпоративный ноутбук AnyDesk или TeamViewer?"
    for question, item in [
        (secret_question, "хранить секреты в репозиториях"),
        (anydesk_question, "устанавливать средства удалённого"),
    ]:
        text = generator.generate(question, contexts).text
        assert item in text
        assert f"Что запрещено: {item}" in text
        assert text.count(item) == text.count(f"Что запрещено: {item}")

    steps = generator.generate(secret_question, contexts).text
    assert "Если секрет попал в репозиторий или чат: Немедленно отзовите" in steps
    assert "Если секрет попал в репозиторий или чат: [" not in steps


@pytest.mark.parametrize("kwargs", [{"max_sentences": 0}, {"min_overlap": 0}])
def test_invalid_parameters_are_rejected(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        ExtractiveGenerator(**kwargs)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("- Пункт списка", "Пункт списка"),
        ("* **Жирный** пункт", "Жирный пункт"),
        ("2) Нумерованный пункт", "Нумерованный пункт"),
        ("| Срок | 5 дней |", "Срок — 5 дней"),
        ("Портал [Нимбус.Люди](https://people.local) открыт", "Портал Нимбус.Люди открыт"),
        ("Команда `vpn connect`  запускает   клиент", "Команда vpn connect запускает клиент"),
        ("Обычное предложение без разметки.", "Обычное предложение без разметки."),
    ],
)
def test_clean_markdown(raw: str, expected: str) -> None:
    assert clean_markdown(raw) == expected
