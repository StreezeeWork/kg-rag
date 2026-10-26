"""Общие помощники тестов поиска и нарезки: построить фрагменты и проверить их инварианты.

Это обычные функции, а не фикстуры: они принимают аргументы, и явный импорт
``from chunk_helpers import paragraph_chunks`` проще проследить, чем фикстуру, возвращающую
функцию. Папка ``tests/`` без ``__init__.py``, поэтому pytest сам кладёт её в ``sys.path``
(режим импорта по умолчанию, ``prepend``) — импорт работает из любого тестового файла.
"""

from __future__ import annotations

from kbrag.text import word_spans
from kbrag.types import Chunk, Document


def paragraph_chunks(docs: list[Document]) -> list[Chunk]:
    """Фрагменты «по абзацам» без чанкера из ingest: строки-заголовки задают section.

    Тестам поиска нужен корпус фрагментов, который не меняется вместе с чанкером:
    тогда падение теста BM25 говорит о BM25, а не о новой нарезке.
    """
    chunks: list[Chunk] = []
    for doc in docs:
        section, pos, number = "", 0, 0
        for para in (p.strip() for p in doc.text.split("\n\n")):
            if not para:
                continue
            start = doc.text.index(para, pos)
            pos = start + len(para)
            if para.startswith("#"):
                section = para.lstrip("#").strip()
                continue
            chunks.append(
                Chunk(f"{doc.doc_id}::{number:03d}", doc.doc_id, para, start, pos, doc.title, section)
            )
            number += 1
    return chunks


def plain_chunk(chunk_id: str, text: str, title: str = "") -> Chunk:
    """Фрагмент из одной строки текста (doc_id берётся из ``chunk_id`` до ``::``)."""
    return Chunk(
        chunk_id=chunk_id, doc_id=chunk_id.split("::")[0], text=text, start=0, end=len(text), title=title
    )


def assert_chunk_invariants(doc: Document, chunks: list[Chunk], chunk_size: int) -> None:
    """Свойства, которые обязаны выполняться для любой стратегии нарезки и любого документа."""
    previous_start = -1
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_id == f"{doc.doc_id}::{i:03d}", "нумерация должна быть сквозной"
        assert chunk.doc_id == doc.doc_id
        assert chunk.title == doc.title
        assert doc.text[chunk.start : chunk.end] == chunk.text, "текст фрагмента не совпадает со смещениями"
        assert chunk.text, "пустой фрагмент"
        assert chunk.text == chunk.text.strip(), "пробелы по краям фрагмента"
        assert chunk.start > previous_start, "фрагменты должны идти по возрастанию start"
        previous_start = chunk.start
        assert len(word_spans(chunk.text)) <= chunk_size, f"фрагмент длиннее {chunk_size} слов"
        # Фрагмент не режет слово пополам: слева и справа от него — пробел или край текста.
        assert chunk.start == 0 or doc.text[chunk.start - 1].isspace()
        assert chunk.end == len(doc.text) or doc.text[chunk.end].isspace()
