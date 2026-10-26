"""Тесты нарезки на фрагменты (kbrag.ingest.chunking).

Главный инвариант: ``doc.text[c.start:c.end] == c.text``. Его проверяем на мини-корпусе и на
синтетических документах с «неудобными» случаями: абзацы и предложения длиннее лимита,
документ только из заголовков, пустой документ, блоки кода с ``#`` внутри.
"""

from __future__ import annotations

import bisect
import itertools
import random

import pytest
from chunk_helpers import assert_chunk_invariants

from kbrag.config import ChunkingConfig
from kbrag.ingest import Chunker, FixedSizeChunker, RecursiveChunker, chunk_corpus, make_chunker
from kbrag.text import find_span, word_spans
from kbrag.types import Chunk, Document

# ---------------------------------------------------------------- синтетические документы

# Словарь с «неудобными» токенами: коды, хосты, кавычки, скобки, числа.
_VOCAB = (
    "отпуск", "заявление", "руководитель", "срок", "дней", "пароль", "клиент", "VPN-01",
    "help.nimbus.local", "компенсация", "«Место»", "28", "сотрудник", "портал", "согласует",
    "(редакция", "2025)",
)  # fmt: skip


def _sentence(rng: random.Random, n_words: int) -> str:
    words = [rng.choice(_VOCAB) for _ in range(n_words)]
    return " ".join(words).capitalize() + "."


def _long_document(seed: int) -> Document:
    """Документ на ~1000 слов со случайной (но воспроизводимой, seed) структурой."""
    rng = random.Random(seed)
    blocks = ["# Синтетический регламент"]
    for number in range(1, 6):
        blocks.append(f"## {number}. Раздел {number}")
        for _ in range(rng.randint(0, 4)):
            blocks.append(" ".join(_sentence(rng, rng.randint(3, 30)) for _ in range(rng.randint(1, 8))))
        if rng.random() < 0.6:
            blocks.append(f"### {number}.1. Подраздел")
            blocks.append(
                "\n".join(f"- {_sentence(rng, rng.randint(2, 12))}" for _ in range(rng.randint(2, 6)))
            )
    blocks.append("## 6. Длинное предложение без точек")
    blocks.append(" ".join(rng.choice(_VOCAB) for _ in range(70)))
    text = "\n\n".join(blocks) + "\n"
    return Document(doc_id=f"synthetic-{seed}", title="Синтетический регламент", text=text)


EDGE_TEXTS: dict[str, str] = {
    "empty": "",
    "blank": "\n \n\t\n",
    "headings_only": "# Заголовок\n\n## Раздел A\n\n### Подраздел A.1\n\n## Раздел B\n",
    "no_headings": "Первый абзац без заголовка. Ещё одно предложение.\n\nВторой абзац.\n",
    "long_paragraph": "## Раздел\n\n" + " ".join(f"Предложение номер {i} про отпуск." for i in range(30)),
    "long_sentence": "## Раздел\n\n" + " ".join(f"слово{i}" for i in range(75)) + ".",
    "code_block": (
        "# Runbook\n\n## Перезапуск\n\n```bash\n# остановить сервис\nsystemctl stop app\n\n"
        "# запустить снова\nsystemctl start app\n```\n\nПроверьте логи.\n"
    ),
    "preamble": "Вступление до заголовков.\n\n## Раздел\n\nТекст раздела.\n",
    "huge_word": "x" * 500,
    "crlf": "# T\r\n\r\n## Раздел\r\n\r\nПервое предложение.\r\nВторое предложение.\r\n",
}

SYNTHETIC_DOCS = [_long_document(seed) for seed in (1, 2, 3)] + [
    Document(doc_id=f"edge-{name}", title="Заголовок", text=text) for name, text in EDGE_TEXTS.items()
]

CHUNKERS = [
    FixedSizeChunker(chunk_size=5, chunk_overlap=0),
    FixedSizeChunker(chunk_size=7, chunk_overlap=3),
    FixedSizeChunker(chunk_size=40, chunk_overlap=10),
    RecursiveChunker(chunk_size=5),
    RecursiveChunker(chunk_size=12),
    RecursiveChunker(chunk_size=40),
    RecursiveChunker(chunk_size=180),
]


def _doc(text: str, title: str = "Документ", **metadata: str) -> Document:
    return Document(doc_id="doc", title=title, text=text, metadata=dict(metadata))


def _texts(chunks: list[Chunk]) -> list[str]:
    return [chunk.text for chunk in chunks]


def _n_words(text: str) -> int:
    return len(word_spans(text))


# ---------------------------------------------------------------- общие инварианты


@pytest.mark.parametrize("chunker", CHUNKERS, ids=repr)
def test_invariants_on_mini_docs(chunker: Chunker, mini_docs: list[Document]) -> None:
    for doc in mini_docs:
        chunks = chunker.split(doc)
        assert chunks, doc.doc_id
        assert_chunk_invariants(doc, chunks, chunker.chunk_size)


@pytest.mark.parametrize("doc", SYNTHETIC_DOCS, ids=lambda doc: doc.doc_id)
@pytest.mark.parametrize("chunker", CHUNKERS, ids=repr)
def test_invariants_on_synthetic_docs(chunker: Chunker, doc: Document) -> None:
    assert_chunk_invariants(doc, chunker.split(doc), chunker.chunk_size)


@pytest.mark.parametrize("text", ["", "   ", "\n\n\t \n"])
@pytest.mark.parametrize("chunker", CHUNKERS, ids=repr)
def test_empty_document_gives_no_chunks(chunker: Chunker, text: str) -> None:
    assert chunker.split(_doc(text)) == []


@pytest.mark.parametrize("chunker", CHUNKERS, ids=repr)
def test_split_is_deterministic(chunker: Chunker) -> None:
    doc = _long_document(seed=42)
    assert chunker.split(doc) == chunker.split(doc)


@pytest.mark.parametrize("chunker", CHUNKERS, ids=repr)
def test_metadata_copies_only_selected_fields(chunker: Chunker) -> None:
    doc = _doc(
        "## Раздел\n\nТекст про отпуск и заявление.",
        status="active",
        department="HR",
        updated="2025-03-01",
        owner="HR-отдел",
        source_path="hr/doc.md",
    )
    chunks = chunker.split(doc)
    assert chunks
    for chunk in chunks:
        assert chunk.metadata == {"status": "active", "department": "HR", "updated": "2025-03-01"}
        assert chunk.metadata is not doc.metadata
    if len(chunks) > 1:  # у каждого фрагмента — свой словарь, а не общий на всех
        assert chunks[0].metadata is not chunks[1].metadata


def test_metadata_empty_when_document_has_none() -> None:
    chunks = RecursiveChunker(50).split(_doc("Просто текст."))
    assert chunks[0].metadata == {}


# ---------------------------------------------------------------- FixedSizeChunker


def _word_index_ranges(doc: Document, chunks: list[Chunk]) -> list[tuple[int, int]]:
    """Для каждого фрагмента — (индекс первого слова, индекс последнего слова) в документе."""
    words = word_spans(doc.text)
    index_by_start = {start: i for i, (start, _) in enumerate(words)}
    index_by_end = {end: i for i, (_, end) in enumerate(words)}
    return [(index_by_start[chunk.start], index_by_end[chunk.end]) for chunk in chunks]


@pytest.mark.parametrize(("size", "overlap"), [(5, 0), (5, 1), (7, 3), (10, 9), (40, 10), (180, 30)])
def test_fixed_windows_have_exact_overlap_and_cover_all_words(
    size: int, overlap: int, mini_docs: list[Document]
) -> None:
    chunker = FixedSizeChunker(chunk_size=size, chunk_overlap=overlap)
    for doc in [*mini_docs, *SYNTHETIC_DOCS[:3]]:
        chunks = chunker.split(doc)
        n_words = _n_words(doc.text)
        ranges = _word_index_ranges(doc, chunks)

        assert ranges[0][0] == 0, "первое окно начинается с первого слова"
        assert ranges[-1][1] == n_words - 1, "последнее окно заканчивается последним словом"
        for (first, last), (next_first, _) in itertools.pairwise(ranges):
            assert last - first + 1 == size, "все окна, кроме последнего, — ровно chunk_size слов"
            assert next_first == first + (size - overlap), "шаг окна = chunk_size - chunk_overlap"
            assert last - next_first + 1 == overlap, "соседние окна делят ровно chunk_overlap слов"
        # Сколько окон должно получиться: первое + сколько шагов нужно, чтобы дойти до конца.
        step = size - overlap
        expected_count = 1 + max(0, -(-(n_words - size) // step))
        assert len(chunks) == expected_count


def test_fixed_short_document_is_one_chunk() -> None:
    doc = _doc("\n  Всего три слова  \n")
    chunks = FixedSizeChunker(chunk_size=10, chunk_overlap=2).split(doc)
    assert _texts(chunks) == ["Всего три слова"]
    assert (chunks[0].start, chunks[0].end) == (3, 18)


def test_fixed_windows_cut_text_regardless_of_structure() -> None:
    doc = _doc("a b c\n\nd e. F g h")
    assert _texts(FixedSizeChunker(chunk_size=4, chunk_overlap=1).split(doc)) == [
        "a b c\n\nd",
        "d e. F g",
        "g h",
    ]


def test_fixed_section_is_path_at_chunk_start(mini_docs: list[Document]) -> None:
    doc = next(d for d in mini_docs if d.doc_id == "hr-vacation")
    chunks = FixedSizeChunker(chunk_size=8, chunk_overlap=2).split(doc)
    section_2_start = doc.text.index("## 2. Оформление")
    # Первое окно начинается с "# Ежегодный оплачиваемый отпуск" — это название документа,
    # оно уже лежит в title, поэтому в путь раздела не попадает.
    assert chunks[0].section == ""
    for chunk in chunks[1:]:
        expected = "2. Оформление" if chunk.start >= section_2_start else "1. Продолжительность"
        assert chunk.section == expected, chunk.text


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (-5, 0), (5, -1), (5, 5), (5, 7)])
def test_fixed_rejects_bad_parameters(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=size, chunk_overlap=overlap)


# ---------------------------------------------------------------- RecursiveChunker


def test_recursive_one_chunk_per_small_section(mini_docs: list[Document]) -> None:
    doc = next(d for d in mini_docs if d.doc_id == "hr-vacation")
    chunks = RecursiveChunker(chunk_size=180).split(doc)
    assert [chunk.section for chunk in chunks] == ["1. Продолжительность", "2. Оформление"]
    # Строка заголовка входит в текст своего раздела; раздел целиком — один фрагмент.
    assert chunks[0].text.startswith("## 1. Продолжительность\n\nПродолжительность ежегодного")
    assert chunks[0].text.endswith("не менее 14 календарных дней.")
    assert chunks[1].text.startswith("## 2. Оформление")
    assert chunks[1].end == len(doc.text.rstrip())
    # "# Ежегодный оплачиваемый отпуск" — раздел без текста, отдельного фрагмента не даёт.
    assert all(not chunk.text.startswith("# ") for chunk in chunks)


def test_recursive_mini_corpus_sections(mini_docs: list[Document]) -> None:
    chunks = chunk_corpus(mini_docs, RecursiveChunker(chunk_size=180))
    assert {(chunk.doc_id, chunk.section) for chunk in chunks} == {
        ("hr-remote-2023", "1. Формат работы"),
        ("hr-remote-2023", "2. Компенсации"),
        ("hr-remote-2025", "1. Формат работы"),
        ("hr-remote-2025", "2. Компенсации"),
        ("hr-vacation", "1. Продолжительность"),
        ("hr-vacation", "2. Оформление"),
        ("it-vpn", "1. Клиент"),
        ("it-vpn", "2. Пароли"),
    }


def test_recursive_nested_section_paths() -> None:
    text = (
        "# Документ\n\n## A\n\nТекст a.\n\n### B\n\nТекст b.\n\n#### C\n\nТекст c.\n\n"
        "## D\n\nТекст d.\n\n### E\n\nТекст e.\n"
    )
    chunks = RecursiveChunker(chunk_size=50).split(_doc(text, title="Документ"))
    assert [(chunk.section, chunk.text) for chunk in chunks] == [
        ("A", "## A\n\nТекст a."),
        ("A > B", "### B\n\nТекст b."),
        ("A > B > C", "#### C\n\nТекст c."),
        ("D", "## D\n\nТекст d."),  # "## D" закрывает и "#### C", и "### B", и "## A"
        ("D > E", "### E\n\nТекст e."),
    ]


def test_recursive_keeps_h1_that_differs_from_title() -> None:
    text = "# Документ\n\n## Раздел\n\nТекст.\n\n# Приложение\n\n## Таблица\n\nДанные.\n"
    chunks = RecursiveChunker(chunk_size=50).split(_doc(text, title="Документ"))
    assert [chunk.section for chunk in chunks] == ["Раздел", "Приложение > Таблица"]


def test_recursive_title_match_ignores_case_and_yo() -> None:
    doc = _doc("# ОТПУСК НА ЕЛКЕ\n\n## Сроки\n\nТекст.", title="Отпуск на ёлке")
    assert [chunk.section for chunk in RecursiveChunker(chunk_size=50).split(doc)] == ["Сроки"]


def test_recursive_preamble_has_empty_section() -> None:
    doc = _doc(EDGE_TEXTS["preamble"])
    chunks = RecursiveChunker(chunk_size=50).split(doc)
    assert [(chunk.section, chunk.text) for chunk in chunks] == [
        ("", "Вступление до заголовков."),
        ("Раздел", "## Раздел\n\nТекст раздела."),
    ]


def test_recursive_headings_only_document_gives_no_chunks() -> None:
    assert RecursiveChunker(chunk_size=50).split(_doc(EDGE_TEXTS["headings_only"])) == []


def test_recursive_ignores_hash_lines_inside_code_blocks() -> None:
    doc = _doc(EDGE_TEXTS["code_block"], title="Runbook")
    chunks = RecursiveChunker(chunk_size=180).split(doc)
    assert len(chunks) == 1
    assert chunks[0].section == "Перезапуск"
    assert "# остановить сервис" in chunks[0].text
    # Даже при мелкой нарезке "# остановить сервис" не становится разделом.
    assert {chunk.section for chunk in RecursiveChunker(chunk_size=3).split(doc)} == {"Перезапуск"}


def test_recursive_packs_paragraphs_greedily() -> None:
    doc = _doc("a b c d\n\ne f g h\n\ni j k l")
    assert _texts(RecursiveChunker(chunk_size=12).split(doc)) == ["a b c d\n\ne f g h\n\ni j k l"]
    assert _texts(RecursiveChunker(chunk_size=10).split(doc)) == ["a b c d\n\ne f g h", "i j k l"]
    assert _texts(RecursiveChunker(chunk_size=4).split(doc)) == ["a b c d", "e f g h", "i j k l"]


def test_recursive_splits_long_paragraph_by_sentences() -> None:
    doc = _doc("Один два три. Четыре пять шесть. Семь восемь девять.")
    assert _texts(RecursiveChunker(chunk_size=6).split(doc)) == [
        "Один два три. Четыре пять шесть.",
        "Семь восемь девять.",
    ]
    assert _texts(RecursiveChunker(chunk_size=4).split(doc)) == [
        "Один два три.",
        "Четыре пять шесть.",
        "Семь восемь девять.",
    ]


def test_recursive_splits_long_sentence_into_word_windows() -> None:
    doc = _doc(" ".join(f"w{i}" for i in range(10)))
    assert _texts(RecursiveChunker(chunk_size=4).split(doc)) == ["w0 w1 w2 w3", "w4 w5 w6 w7", "w8 w9"]


def test_recursive_drops_heading_left_alone_after_packing() -> None:
    # Заголовок (2 слова) + абзац (6 слов) не влезают в 6 слов вместе. Фрагмент из одного
    # заголовка бесполезен — он отбрасывается, а название раздела остаётся в section.
    doc = _doc("## Раздел\n\nРаз два три четыре пять шесть.")
    chunks = RecursiveChunker(chunk_size=6).split(doc)
    assert [(chunk.chunk_id, chunk.section, chunk.text) for chunk in chunks] == [
        ("doc::000", "Раздел", "Раз два три четыре пять шесть.")
    ]


def test_recursive_handles_crlf_text() -> None:
    doc = _doc(EDGE_TEXTS["crlf"], title="T")
    chunks = RecursiveChunker(chunk_size=3).split(doc)
    assert _texts(chunks) == ["Первое предложение.", "Второе предложение."]
    assert {chunk.section for chunk in chunks} == {"Раздел"}


def _line_of(text: str, pos: int) -> str:
    """Строка текста, в которой находится символ pos (без пробелов и \\r по краям)."""
    line_start = text.rfind("\n", 0, pos) + 1
    line_end = text.find("\n", pos)
    return text[line_start : line_end if line_end != -1 else len(text)].strip()


@pytest.mark.parametrize("chunk_size", [3, 5, 12, 40, 180])
@pytest.mark.parametrize("doc", SYNTHETIC_DOCS, ids=lambda doc: doc.doc_id)
def test_recursive_chunks_do_not_overlap_and_lose_no_text(chunk_size: int, doc: Document) -> None:
    chunks = RecursiveChunker(chunk_size=chunk_size).split(doc)
    for chunk, next_chunk in itertools.pairwise(chunks):
        assert chunk.end <= next_chunk.start, "фрагменты не должны перекрываться"

    # Сколько фрагментов покрывает каждое слово документа (слова фрагмент не режет — см. инварианты).
    words = word_spans(doc.text)
    word_starts = [start for start, _ in words]
    coverage = [0] * len(words)
    for chunk in chunks:
        for i in range(
            bisect.bisect_left(word_starts, chunk.start), bisect.bisect_left(word_starts, chunk.end)
        ):
            coverage[i] += 1

    for (start, end), covered_by in zip(words, coverage, strict=True):
        if _line_of(doc.text, start).startswith("#"):
            assert covered_by <= 1  # строку заголовка можно отбросить, но не продублировать
        else:
            assert covered_by == 1, f"слово {doc.text[start:end]!r} потеряно или продублировано"


@pytest.mark.parametrize("doc", SYNTHETIC_DOCS[:3], ids=lambda doc: doc.doc_id)
def test_recursive_sections_are_contiguous_paths(doc: Document) -> None:
    sections = [chunk.section for chunk in RecursiveChunker(chunk_size=12).split(doc)]
    # Разделы идут блоками: раз раздел закончился, он больше не встречается.
    blocks = [section for i, section in enumerate(sections) if i == 0 or section != sections[i - 1]]
    assert len(blocks) == len(set(blocks))
    # Путь — цепочка заголовков без '#' и без названия документа.
    allowed = {f"{n}. Раздел {n}" for n in range(1, 6)}
    allowed |= {f"{n}. Раздел {n} > {n}.1. Подраздел" for n in range(1, 6)}
    allowed.add("6. Длинное предложение без точек")
    assert set(sections) <= allowed
    assert "6. Длинное предложение без точек" in sections


def test_recursive_rejects_bad_chunk_size() -> None:
    with pytest.raises(ValueError):
        RecursiveChunker(chunk_size=0)


# ---------------------------------------------------------------- make_chunker / chunk_corpus


def test_make_chunker_builds_strategy_from_config() -> None:
    fixed = make_chunker(ChunkingConfig(strategy="fixed", chunk_size=50, chunk_overlap=10))
    assert fixed == FixedSizeChunker(chunk_size=50, chunk_overlap=10)
    recursive = make_chunker(ChunkingConfig(strategy="recursive", chunk_size=120, chunk_overlap=10))
    assert recursive == RecursiveChunker(chunk_size=120)
    assert isinstance(fixed, Chunker) and isinstance(recursive, Chunker)


def test_make_chunker_default_config() -> None:
    assert make_chunker(ChunkingConfig()) == RecursiveChunker(chunk_size=180)


def test_make_chunker_propagates_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        make_chunker(ChunkingConfig(strategy="fixed", chunk_size=20, chunk_overlap=20))


def test_chunk_corpus_concatenates_documents_in_order(mini_docs: list[Document]) -> None:
    chunker = RecursiveChunker(chunk_size=12)
    chunks = chunk_corpus(mini_docs, chunker)
    expected = [chunk for doc in mini_docs for chunk in chunker.split(doc)]
    assert chunks == expected
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    # Нумерация начинается заново в каждом документе.
    assert [chunk.chunk_id for chunk in chunks if chunk.chunk_id.endswith("::000")] == [
        f"{doc.doc_id}::000" for doc in mini_docs
    ]


def test_chunk_corpus_empty() -> None:
    assert chunk_corpus([], RecursiveChunker(chunk_size=50)) == []


def test_gold_evidence_lands_in_some_chunk(mini_docs: list[Document], mini_gold_records: list[dict]) -> None:
    # Ради этого всё и затевалось: цитата из эталона целиком лежит внутри фрагмента
    # (при разумном размере фрагмента), значит поиск в принципе может её найти.
    chunks = chunk_corpus(mini_docs, make_chunker(ChunkingConfig()))
    docs_by_id = {doc.doc_id: doc for doc in mini_docs}
    for record in mini_gold_records:
        for evidence in record["evidence"]:
            span = find_span(docs_by_id[evidence["doc_id"]].text, evidence["quote"])
            assert span is not None, record["id"]
            start, end = span
            assert any(
                chunk.doc_id == evidence["doc_id"] and chunk.start <= start and end <= chunk.end
                for chunk in chunks
            ), record["id"]
