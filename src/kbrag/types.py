"""Базовые структуры данных, которыми обмениваются все модули.

Модули не передают друг другу «сырые» словари: так проще читать код и ловить
ошибки на этапе разработки.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    """Один исходный документ базы знаний (markdown-файл без front matter)."""

    doc_id: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """Фрагмент документа — единица поиска.

    Инвариант (проверяется тестами): ``document.text[start:end] == chunk.text``.
    Благодаря точным смещениям можно честно проверять, попал ли в выдачу
    фрагмент с доказательством ответа, независимо от способа нарезки.
    """

    chunk_id: str
    doc_id: str
    text: str
    start: int
    end: int
    title: str = ""
    section: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def index_text(self, with_header: bool = True) -> str:
        """Текст, который видят поисковые индексы.

        Если ``with_header`` включён, к фрагменту приписываются заголовок документа и
        раздела (так называемый contextual chunk header): фрагмент «Срок — 5 рабочих
        дней» без заголовка непонятно к чему относится.
        """
        if not with_header:
            return self.text
        header = " / ".join(part for part in (self.title, self.section) if part)
        return f"{header}\n{self.text}" if header else self.text


@dataclass
class ScoredChunk:
    """Фрагмент с оценкой релевантности от конкретного этапа поиска."""

    chunk: Chunk
    score: float
    source: str  # "bm25" | "dense" | "hybrid" | "rerank"
    details: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Citation:
    """Ссылка из ответа на фрагмент контекста: ``[index]`` в тексте ответа."""

    index: int  # 1-based номер фрагмента в контексте, переданном генератору
    chunk_id: str
    doc_id: str
    title: str


@dataclass
class Answer:
    """Итог работы пайплайна на один вопрос."""

    question: str
    text: str
    citations: list[Citation]
    contexts: list[ScoredChunk]
    abstained: bool
    timings_ms: dict[str, float] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    invalid_citations: int = 0
    confidence: float | None = None  # «уверенность» поиска, по ней принимается решение об отказе
