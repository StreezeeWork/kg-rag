"""Разбор ссылок ``[n]`` в ответе и привязка их к фрагментам контекста.

Ссылки — главный способ проверить ответ: по ним пользователь открывает документ,
а метрики citation precision/recall считают, на те ли фрагменты сослалась модель.
Ссылку на несуществующий номер («[7]» при пяти фрагментах) модель выдумала —
такие считаем отдельно, это сигнал о галлюцинациях.
"""

from __future__ import annotations

import re

from kbrag.types import Citation, ScoredChunk

# «[1]», «[1, 2]», «[1,2;3]»; подряд идущие «[1][3]» — это просто два совпадения.
_CITATION_RE = re.compile(r"\[\s*(\d+(?:\s*[,;]\s*\d+)*)\s*\]")
_NUMBER_RE = re.compile(r"\d+")


def parse_citations(text: str, contexts: list[ScoredChunk]) -> tuple[list[Citation], int]:
    """Возвращает уникальные ссылки в порядке появления и число невалидных номеров.

    ``contexts`` — ровно тот список, что был передан генератору: номер ``[n]``
    означает ``contexts[n - 1]``. Невалидные номера (0 или больше числа фрагментов)
    тоже считаются по одному разу на номер, как и валидные.
    """
    citations: list[Citation] = []
    seen: set[int] = set()
    invalid: set[int] = set()
    for match in _CITATION_RE.finditer(text):
        for raw_number in _NUMBER_RE.findall(match.group(1)):
            number = int(raw_number)
            if not 1 <= number <= len(contexts):
                invalid.add(number)
                continue
            if number in seen:
                continue
            seen.add(number)
            chunk = contexts[number - 1].chunk
            citations.append(
                Citation(index=number, chunk_id=chunk.chunk_id, doc_id=chunk.doc_id, title=chunk.title)
            )
    return citations, len(invalid)
