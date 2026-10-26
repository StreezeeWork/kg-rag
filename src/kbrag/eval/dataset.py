"""Эталонный набор вопросов (gold): загрузка, проверка, разбиение на dev/test.

Оценка RAG честна ровно настолько, насколько честен эталон. Поэтому здесь:

* каждая «улика» (evidence) — дословная цитата из документа, а не номер фрагмента:
  так разметка не зависит от способа нарезки и переживает смену ``chunk_size``;
* ``validate_gold`` ловит битую разметку до того, как она тихо испортит метрики;
* разбиение на dev/test детерминировано: один и тот же вопрос всегда попадает
  в одну и ту же часть, и порог отказа никогда не подбирается на test.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from kbrag.text import find_span
from kbrag.types import Chunk, Document

QUESTION_TYPES = (
    "factoid",
    "numeric",
    "procedural",
    "paraphrase",
    "keyword",
    "confusable",
    "multi_hop",
    "unanswerable",
)
SPLITS = ("dev", "test")

# Положение цитаты в корпусе: (doc_id, start, end) — смещения в Document.text.
Span = tuple[str, int, int]


@dataclass(frozen=True)
class Evidence:
    """Доказательство ответа: дословная цитата из конкретного документа."""

    doc_id: str
    quote: str


@dataclass(frozen=True)
class GoldItem:
    """Один эталонный вопрос.

    Кортежи вместо списков — чтобы объект был по-настоящему неизменяемым
    и его можно было класть в множества и использовать как ключ.
    """

    qid: str
    question: str
    answer: str
    key_facts: tuple[str, ...]
    evidence: tuple[Evidence, ...]
    qtype: str
    split: str

    @property
    def answerable(self) -> bool:
        """Есть ли ответ в базе знаний (на «неотвечаемые» вопросы правильно отказаться)."""
        return self.qtype != "unanswerable"


# ---------------------------------------------------------------------------
# Чтение и запись JSONL
# ---------------------------------------------------------------------------


def item_from_record(record: Mapping[str, Any]) -> GoldItem:
    """Превращает одну JSON-запись в ``GoldItem``; при битой структуре — ``ValueError``.

    ``split`` может отсутствовать: в частях набора (``data/eval/parts``) его ещё нет,
    он назначается позже через ``assign_splits``.
    """
    missing = [key for key in ("id", "question", "type") if key not in record]
    if missing:
        raise ValueError(f"нет обязательных полей: {', '.join(missing)}")

    key_facts = record.get("key_facts") or []
    # Строка вместо списка превратилась бы в кортеж отдельных букв — ловим явно.
    if not isinstance(key_facts, list):
        raise ValueError("key_facts должен быть списком строк")

    raw_evidence = record.get("evidence") or []
    if not isinstance(raw_evidence, list):
        raise ValueError("evidence должен быть списком объектов {doc_id, quote}")
    evidence: list[Evidence] = []
    for ev in raw_evidence:
        if not isinstance(ev, Mapping) or "doc_id" not in ev or "quote" not in ev:
            raise ValueError("каждый элемент evidence должен содержать doc_id и quote")
        evidence.append(Evidence(doc_id=str(ev["doc_id"]), quote=str(ev["quote"])))

    return GoldItem(
        qid=str(record["id"]),
        question=str(record["question"]),
        answer=str(record.get("answer") or ""),
        key_facts=tuple(str(fact) for fact in key_facts),
        evidence=tuple(evidence),
        qtype=str(record["type"]),
        split=str(record.get("split") or ""),
    )


def item_to_record(item: GoldItem) -> dict[str, Any]:
    """Обратное преобразование: ``GoldItem`` → словарь в формате ``gold.jsonl``."""
    return {
        "id": item.qid,
        "question": item.question,
        "answer": item.answer,
        "key_facts": list(item.key_facts),
        "evidence": [{"doc_id": ev.doc_id, "quote": ev.quote} for ev in item.evidence],
        "type": item.qtype,
        "split": item.split,
    }


def load_gold(path: str | Path) -> list[GoldItem]:
    """Читает JSONL (одна строка — один вопрос). Пустые строки пропускаются.

    В сообщении об ошибке есть номер строки: в наборе из сотен вопросов
    иначе битую запись не найти.
    """
    items: list[GoldItem] = []
    # utf-8-sig спокойно читает файлы с BOM, которые любят сохранять редакторы на Windows.
    with Path(path).open(encoding="utf-8-sig") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("строка должна быть JSON-объектом")
                items.append(item_from_record(record))
            except ValueError as exc:  # json.JSONDecodeError — тоже ValueError
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
    return items


def save_gold(items: Sequence[GoldItem], path: str | Path) -> None:
    """Пишет набор в JSONL (UTF-8, кириллица без экранирования, переводы строк ``\\n``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n": файл одинаков на Windows и Linux, git не видит лишних изменений.
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for item in items:
            fh.write(json.dumps(item_to_record(item), ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Проверка разметки
# ---------------------------------------------------------------------------


def validate_gold(items: Sequence[GoldItem], docs: Sequence[Document]) -> list[str]:
    """Возвращает список ошибок разметки (пустой список — всё хорошо).

    Не бросаем исключение на первой ошибке: удобнее увидеть и исправить все сразу.
    """
    docs_by_id = {doc.doc_id: doc for doc in docs}
    errors: list[str] = []
    seen_ids: set[str] = set()

    for item in items:
        where = item.qid or "<без id>"
        if item.qid in seen_ids:
            errors.append(f"{where}: повторяющийся id")
        seen_ids.add(item.qid)

        if not item.question.strip():
            errors.append(f"{where}: пустой вопрос")
        if item.qtype not in QUESTION_TYPES:
            errors.append(f"{where}: неизвестный type {item.qtype!r}")
        if item.split not in SPLITS:
            errors.append(f"{where}: неизвестный split {item.split!r} (ожидается dev или test)")

        if item.answerable:
            if not item.evidence:
                errors.append(f"{where}: у отвечаемого вопроса нет evidence")
            if not item.key_facts:
                errors.append(f"{where}: у отвечаемого вопроса нет key_facts")
        else:
            if item.evidence:
                errors.append(f"{where}: у неотвечаемого вопроса не должно быть evidence")
            if item.key_facts:
                errors.append(f"{where}: у неотвечаемого вопроса не должно быть key_facts")

        for ev in item.evidence:
            doc = docs_by_id.get(ev.doc_id)
            if doc is None:
                errors.append(f"{where}: неизвестный doc_id {ev.doc_id!r}")
            elif find_span(doc.text, ev.quote) is None:
                errors.append(f"{where}: цитата не найдена в документе {ev.doc_id!r}: {ev.quote[:60]!r}")
    return errors


# ---------------------------------------------------------------------------
# Релевантность фрагментов
# ---------------------------------------------------------------------------


def evidence_spans(item: GoldItem, docs_by_id: Mapping[str, Document]) -> list[Span]:
    """Находит каждую цитату вопроса в тексте документа и возвращает её смещения.

    Если цитата не находится, это ошибка разметки: молча пропустить её нельзя,
    иначе recall посчитается от меньшего числа доказательств и окажется завышен.
    """
    spans: list[Span] = []
    for ev in item.evidence:
        doc = docs_by_id.get(ev.doc_id)
        if doc is None:
            raise ValueError(f"{item.qid}: неизвестный doc_id {ev.doc_id!r} (запустите validate_gold)")
        found = find_span(doc.text, ev.quote)
        if found is None:
            raise ValueError(f"{item.qid}: цитата не найдена в {ev.doc_id!r} (запустите validate_gold)")
        spans.append((ev.doc_id, found[0], found[1]))
    return spans


def is_relevant(chunk: Chunk, span: Span, min_coverage: float = 0.5) -> bool:
    """Считается ли фрагмент найденным доказательством для цитаты ``span``.

    Правило: тот же документ и пересечение ≥ ``min_coverage`` от длины **меньшего**
    из двух отрезков. Почему от меньшего:

    * маленький фрагмент целиком внутри длинной цитаты — релевантен (весь он — доказательство);
    * короткая цитата целиком внутри большого фрагмента — релевантна (доказательство в нём есть);
    * цитата, разрезанная границей фрагментов пополам, засчитывается обеим половинам;
    * фрагмент, задевший цитату парой символов на краю, — нет.
    """
    doc_id, start, end = span
    if chunk.doc_id != doc_id:
        return False
    overlap = min(end, chunk.end) - max(start, chunk.start)
    if overlap <= 0:
        return False
    shorter = min(end - start, chunk.end - chunk.start)
    return overlap >= min_coverage * shorter


# ---------------------------------------------------------------------------
# Разбиение на dev/test
# ---------------------------------------------------------------------------


def _split_key(seed: int, qid: str) -> str:
    # Хэш вместо random.shuffle: место вопроса не зависит ни от порядка строк в файле,
    # ни от того, какие ещё вопросы добавили в набор.
    return hashlib.sha1(f"{seed}:{qid}".encode()).hexdigest()


def assign_splits(items: Sequence[GoldItem], dev_fraction: float = 0.3, seed: int = 13) -> list[GoldItem]:
    """Назначает каждому вопросу ``split`` = ``dev`` или ``test``.

    Стратификация по типу вопроса: в dev попадает ~``dev_fraction`` вопросов **каждого**
    типа. Иначе, например, все неотвечаемые вопросы могли бы случайно уйти в test,
    и порог отказа было бы не на чем калибровать. Порядок элементов сохраняется.
    """
    if not 0.0 <= dev_fraction <= 1.0:
        raise ValueError(f"dev_fraction должен быть в [0, 1], получено {dev_fraction}")

    by_type: dict[str, list[GoldItem]] = defaultdict(list)
    for item in items:
        by_type[item.qtype].append(item)

    dev_ids: set[str] = set()
    for qtype in sorted(by_type):
        group = sorted(by_type[qtype], key=lambda it: (_split_key(seed, it.qid), it.qid))
        # Обычное округление (0.5 → вверх), а не «банковское» round() из Python.
        n_dev = math.floor(len(group) * dev_fraction + 0.5)
        dev_ids.update(it.qid for it in group[:n_dev])

    return [replace(item, split="dev" if item.qid in dev_ids else "test") for item in items]
