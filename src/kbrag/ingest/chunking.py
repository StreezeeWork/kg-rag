"""Нарезка документов на фрагменты (chunks) — единицы поиска.

Зачем резать: поисковик и LLM работают с короткими кусками текста. Документ целиком
«размывает» релевантность (вопрос про пароли совпадёт со всем документом про VPN) и плохо
помещается в контекст генератора. Слишком мелкие куски теряют смысл («Срок — 5 дней» —
срок чего?). Поэтому здесь две стратегии:

* ``FixedSizeChunker`` — скользящее окно по словам с перекрытием. Базовая линия (baseline):
  ничего не знает о структуре текста и может разрезать предложение пополам.
* ``RecursiveChunker`` — уважает структуру markdown: раздел → абзацы → предложения → слова.
  Фрагмент по возможности совпадает с целым разделом или с группой соседних абзацев.

Обе стратегии гарантируют инвариант ``doc.text[chunk.start:chunk.end] == chunk.text``:
фрагмент — это ровно кусок исходного текста. На этом держится оценка качества: эталонные
цитаты ищутся в документе по смещениям, и можно честно проверить, попала ли цитата во фрагмент.
"""

from __future__ import annotations

import bisect
import itertools
import logging
import re
from dataclasses import dataclass
from typing import NamedTuple, Protocol, runtime_checkable

from kbrag.config import ChunkingConfig
from kbrag.text import normalize, split_sentences, word_spans
from kbrag.types import Chunk, Document

logger = logging.getLogger(__name__)

# Поля документа, которые копируются в каждый фрагмент: по ним генератор отличает
# действующую политику от архивной, не заглядывая в сам документ.
CHUNK_METADATA_KEYS = ("status", "department", "updated")

Span = tuple[int, int]  # полуинтервал [start, end) смещений в doc.text

_LINE_RE = re.compile(r"^.*$", re.MULTILINE)
# Заголовок markdown: до 3 пробелов отступа, 1–6 символов '#', пробел, текст ("## 2. Оформление").
_HEADING_RE = re.compile(r" {0,3}(#{1,6})[ \t]+(\S.*)")
# Граница абзаца — пустая строка (в ней могут быть пробелы).
_PARAGRAPH_BREAK_RE = re.compile(r"\n\s*\n")


@runtime_checkable
class Chunker(Protocol):
    """Стратегия нарезки: любой объект с методом ``split``.

    Protocol вместо базового класса: чтобы подключить свою стратегию, достаточно написать
    класс с методом ``split`` — наследоваться ни от чего не нужно.
    """

    def split(self, doc: Document) -> list[Chunk]: ...


@dataclass(frozen=True)
class FixedSizeChunker:
    """Окна по ``chunk_size`` слов, соседние окна перекрываются на ``chunk_overlap`` слов.

    Перекрытие нужно, чтобы факт, попавший на границу окон, целиком оказался хотя бы в
    одном фрагменте. Плата — дублирование текста в индексе.
    """

    chunk_size: int
    chunk_overlap: int

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size должен быть ≥ 1, получено {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap не может быть отрицательным, получено {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            # Шаг окна = size - overlap; при overlap >= size окно стояло бы на месте или шло назад.
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) должен быть меньше chunk_size ({self.chunk_size})"
            )

    def split(self, doc: Document) -> list[Chunk]:
        step = self.chunk_size - self.chunk_overlap
        spans = _word_windows(doc.text, 0, len(doc.text), self.chunk_size, step)
        return _build_chunks(doc, spans, _find_headings(doc.text))


@dataclass(frozen=True)
class RecursiveChunker:
    """Нарезка «сверху вниз» по структуре markdown, фрагменты не длиннее ``chunk_size`` слов.

    1. Документ делится на разделы по заголовкам (строка заголовка входит в свой раздел).
    2. Раздел помещается в лимит → один фрагмент.
    3. Иначе раздел делится на абзацы, длинный абзац — на предложения, длинное
       предложение — на окна слов; получившиеся куски жадно собираются во фрагменты
       до лимита. Фрагменты одного раздела не перекрываются.

    Фрагмент, который состоит ровно из одной строки заголовка, отбрасывается. Так бывает,
    когда раздел пуст (``# Название документа`` сразу перед ``## 1. ...``) или когда первый
    абзац раздела не поместился в один фрагмент с заголовком. Смысла в таком фрагменте нет:
    название документа лежит в ``Chunk.title``, заголовок раздела — в ``Chunk.section``
    соседних фрагментов, а «пустышка» из одного заголовка только засоряла бы выдачу поиска.
    """

    chunk_size: int

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size должен быть ≥ 1, получено {self.chunk_size}")

    def split(self, doc: Document) -> list[Chunk]:
        headings = _find_headings(doc.text)
        spans: list[Span] = []
        for start, end in _section_spans(doc.text, headings):
            spans.extend(self._split_section(doc.text, start, end))
        heading_lines = {_strip_span(doc.text, h.line_start, h.line_end) for h in headings}
        spans = [span for span in spans if span not in heading_lines]
        return _build_chunks(doc, spans, headings)

    def _split_section(self, text: str, start: int, end: int) -> list[Span]:
        if _count_words(text, start, end) <= self.chunk_size:
            return [(start, end)]
        pieces: list[_Piece] = []
        for par_start, par_end in _paragraph_spans(text, start, end):
            pieces.extend(self._split_paragraph(text, par_start, par_end))
        return _pack_greedily(pieces, self.chunk_size)

    def _split_paragraph(self, text: str, start: int, end: int) -> list[_Piece]:
        """Абзац → куски, каждый не длиннее лимита (сам абзац, его предложения или окна слов)."""
        n_words = _count_words(text, start, end)
        if n_words <= self.chunk_size:
            return [_Piece(start, end, n_words)]
        pieces: list[_Piece] = []
        for rel_start, rel_end in split_sentences(text[start:end]):
            sent_start, sent_end = start + rel_start, start + rel_end
            n_words = _count_words(text, sent_start, sent_end)
            if n_words <= self.chunk_size:
                pieces.append(_Piece(sent_start, sent_end, n_words))
                continue
            # Предложение длиннее лимита (обычно это таблица или список без точек) —
            # режем по словам без перекрытия: другого способа уложиться в лимит нет.
            for win_start, win_end in _word_windows(
                text, sent_start, sent_end, self.chunk_size, self.chunk_size
            ):
                pieces.append(_Piece(win_start, win_end, _count_words(text, win_start, win_end)))
        return pieces


def make_chunker(cfg: ChunkingConfig) -> Chunker:
    """Создаёт стратегию нарезки по конфигу (секция ``chunking`` в YAML)."""
    if cfg.strategy == "fixed":
        return FixedSizeChunker(chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)
    if cfg.strategy == "recursive":
        return RecursiveChunker(chunk_size=cfg.chunk_size)
    raise ValueError(f"Неизвестная стратегия нарезки: {cfg.strategy!r}")


def chunk_corpus(docs: list[Document], chunker: Chunker) -> list[Chunk]:
    """Режет все документы; фрагменты идут в порядке документов, внутри — по ``start``."""
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunker.split(doc))
    logger.info("Нарезка %r: документов %d → фрагментов %d", chunker, len(docs), len(chunks))
    return chunks


# ---------------------------------------------------------------------------
# Внутренние помощники
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Heading:
    line_start: int  # смещение начала строки заголовка в тексте
    line_end: int  # смещение конца этой строки (позиция перед "\n")
    level: int  # число символов '#': 1 — название документа, 2 — раздел, 3 — подраздел, ...
    title: str  # текст заголовка без '#'


class _Piece(NamedTuple):
    """Неделимый кусок для жадной упаковки: абзац, предложение или окно слов."""

    start: int
    end: int
    n_words: int


def _find_headings(text: str) -> list[_Heading]:
    """Находит заголовки markdown, пропуская блоки кода.

    Внутри блока ```...``` строка ``# перезапустить сервис`` — комментарий shell, а не раздел.
    Без этой проверки runbook-и с примерами команд резались бы на бессмысленные «разделы».
    """
    headings: list[_Heading] = []
    in_code_block = False
    for line in _LINE_RE.finditer(text):
        content = line.group()
        if content.lstrip().startswith(("```", "~~~")):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        match = _HEADING_RE.match(content)
        if match:
            headings.append(_Heading(line.start(), line.end(), len(match.group(1)), match.group(2).strip()))
    return headings


class _SectionIndex:
    """Отвечает на вопрос «в каком разделе находится символ с номером pos».

    Путь раздела — цепочка вложенных заголовков через " > ": ``"2. Оформление > 2.1. Сроки"``.
    Заголовок, совпадающий с названием документа, в путь не входит: название уже лежит в
    ``Chunk.title``, а повтор лишь удлинил бы шапку фрагмента при индексации.
    """

    def __init__(self, headings: list[_Heading], doc_title: str) -> None:
        self._starts = [heading.line_start for heading in headings]
        self._paths: list[str] = []
        title_key = normalize(doc_title)
        stack: list[_Heading] = []  # открытые сейчас заголовки, от верхнего уровня к нижнему
        for heading in headings:
            # Новый заголовок закрывает все открытые заголовки того же уровня и глубже:
            # после "### 1.2" и "## 2" путь — просто "2", а не "1 > 1.2 > 2".
            while stack and stack[-1].level >= heading.level:
                stack.pop()
            stack.append(heading)
            self._paths.append(" > ".join(h.title for h in stack if normalize(h.title) != title_key))

    def section_at(self, pos: int) -> str:
        # Последний заголовок, строка которого начинается не позже pos (бинарный поиск).
        index = bisect.bisect_right(self._starts, pos) - 1
        return self._paths[index] if index >= 0 else ""


def _build_chunks(doc: Document, spans: list[Span], headings: list[_Heading]) -> list[Chunk]:
    """Превращает смещения в объекты ``Chunk`` со сквозной нумерацией внутри документа."""
    sections = _SectionIndex(headings, doc.title)
    metadata = {key: doc.metadata[key] for key in CHUNK_METADATA_KEYS if key in doc.metadata}
    return [
        Chunk(
            chunk_id=f"{doc.doc_id}::{i:03d}",
            doc_id=doc.doc_id,
            text=doc.text[start:end],
            start=start,
            end=end,
            title=doc.title,
            section=sections.section_at(start),
            # Своя копия у каждого фрагмента: общий изменяемый словарь — источник трудноуловимых багов.
            metadata=dict(metadata),
        )
        for i, (start, end) in enumerate(spans)
    ]


def _section_spans(text: str, headings: list[_Heading]) -> list[Span]:
    """Делит текст на разделы: раздел — от строки заголовка до следующего заголовка.

    Текст до первого заголовка («преамбула») — тоже раздел, просто без заголовка.
    """
    boundaries = [0] + [heading.line_start for heading in headings] + [len(text)]
    spans: list[Span] = []
    for section_start, section_end in itertools.pairwise(boundaries):  # соседние пары границ
        section = _strip_span(text, section_start, section_end)
        if section:
            spans.append(section)
    return spans


def _paragraph_spans(text: str, start: int, end: int) -> list[Span]:
    """Абзацы внутри ``text[start:end]`` (разделитель — пустая строка), без пробелов по краям."""
    spans: list[Span] = []
    pos = start
    for match in _PARAGRAPH_BREAK_RE.finditer(text, start, end):
        paragraph = _strip_span(text, pos, match.start())
        if paragraph:
            spans.append(paragraph)
        pos = match.end()
    paragraph = _strip_span(text, pos, end)
    if paragraph:
        spans.append(paragraph)
    return spans


def _pack_greedily(pieces: list[_Piece], limit: int) -> list[Span]:
    """Склеивает соседние куски, пока сумма слов не превысит ``limit``.

    Фрагмент — от начала первого куска до конца последнего, поэтому пробелы и пустые строки
    между кусками остаются внутри фрагмента и инвариант со смещениями не нарушается.
    Каждый кусок сам по себе не длиннее ``limit``, так что и фрагмент в лимит укладывается.
    """
    spans: list[Span] = []
    current_start = current_end = 0
    current_words = 0  # 0 — «текущий фрагмент пуст» (в любом куске есть хотя бы одно слово)
    for piece in pieces:
        if current_words and current_words + piece.n_words > limit:
            spans.append((current_start, current_end))
            current_words = 0
        if not current_words:
            current_start = piece.start
        current_end = piece.end
        current_words += piece.n_words
    if current_words:
        spans.append((current_start, current_end))
    return spans


def _word_windows(text: str, start: int, end: int, size: int, step: int) -> list[Span]:
    """Окна по ``size`` слов со сдвигом ``step`` внутри ``text[start:end]``.

    Последнее окно может быть короче ``size``: как только окно дошло до конца текста,
    останавливаемся, чтобы не плодить фрагменты, целиком лежащие внутри предыдущего.
    """
    words = [(start + a, start + b) for a, b in word_spans(text[start:end])]
    windows: list[Span] = []
    for first in range(0, len(words), step):
        last = min(first + size, len(words))  # индекс за последним словом окна
        windows.append((words[first][0], words[last - 1][1]))
        if last == len(words):
            break
    return windows


def _count_words(text: str, start: int, end: int) -> int:
    # «Слово» — то же, что в text.word_spans: последовательность непробельных символов.
    return len(word_spans(text[start:end]))


def _strip_span(text: str, start: int, end: int) -> Span | None:
    """Сдвигает границы внутрь, отбрасывая пробельные символы; пустой результат → None."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if end > start else None
