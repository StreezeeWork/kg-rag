"""Загрузка корпуса: markdown-файлы с front matter → объекты ``Document``.

Формат файла (подробнее — data/DATASET.md)::

    ---
    id: hr-vacation
    title: Ежегодный оплачиваемый отпуск
    department: HR
    status: active
    updated: 2025-03-01
    ---
    # Ежегодный оплачиваемый отпуск
    ...

Front matter — «паспорт» документа (YAML между строками ``---``), всё ниже — текст, по
которому ищем. Метаданные не смешиваем с текстом: по ним, например, отличают действующую
политику от архивной, а в поисковый индекс они попадать не должны.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

import yaml

from kbrag.types import Document

logger = logging.getLogger(__name__)

_FRONT_MATTER_FENCE = "---"
# Заголовок первого уровня: "# Название" (но не "## Раздел").
_H1_RE = re.compile(r"^# +(\S.*)$", re.MULTILINE)


def parse_front_matter(raw: str) -> tuple[dict, str]:
    """Отделяет YAML-шапку от текста: возвращает ``(meta, body)``.

    Шапка — строки между первой строкой ``---`` и следующей строкой ``---``.
    Нет шапки → ``({}, raw)``. Шапка открыта, но не закрыта, или YAML битый → ``ValueError``:
    такой файл почти наверняка испорчен, и молча считать его текстом опаснее, чем упасть.
    Даты сразу превращаются в строки ``YYYY-MM-DD`` (см. ``_to_plain``).
    """
    # BOM («невидимый» символ в начале файла от Блокнота Windows) не должен ломать проверку на "---".
    text = raw.removeprefix("﻿")
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != _FRONT_MATTER_FENCE:
        return {}, raw

    closing = next((i for i in range(1, len(lines)) if lines[i].strip() == _FRONT_MATTER_FENCE), None)
    if closing is None:
        raise ValueError("front matter начинается со строки '---', но не закрыт второй строкой '---'")

    yaml_text = "".join(lines[1:closing])
    body = "".join(lines[closing + 1 :])
    try:
        meta = yaml.safe_load(yaml_text)  # safe_load: YAML из файла не может создать произвольный объект
    except yaml.YAMLError as exc:
        raise ValueError(f"некорректный YAML во front matter: {exc}") from exc
    if meta is None:  # пустая шапка "---\n---"
        meta = {}
    if not isinstance(meta, dict):
        raise ValueError("front matter должен быть набором пар «ключ: значение»")
    return {str(key): _to_plain(value) for key, value in meta.items()}, body


def load_document(path: Path, root: Path | None = None) -> Document:
    """Читает один markdown-файл и превращает его в ``Document``.

    - ``doc_id`` берётся из поля ``id`` (обязательно: по нему ссылается эталонный набор);
    - ``title`` — поле ``title``, иначе первый заголовок ``# ``, иначе ``doc_id``;
    - ``text`` — тело файла как есть, без ведущих пустых строк. Тело не «чистим»:
      смещения фрагментов и эталонные цитаты считаются именно по этому тексту;
    - ``metadata`` — остальные поля шапки + ``source_path`` (путь относительно ``root``).
    """
    path = Path(path)
    # read_text превращает \r\n в \n: смещения в тексте одинаковы на Windows и Linux.
    # utf-8-sig — обычный UTF-8, который вдобавок убирает BOM, если он есть.
    raw = path.read_text(encoding="utf-8-sig")
    try:
        meta, body = parse_front_matter(raw)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc

    meta = dict(meta)
    doc_id = str(meta.pop("id", "") or "").strip()
    if not doc_id:
        raise ValueError(f"{path}: во front matter нет обязательного поля 'id'")

    text = body.lstrip("\r\n")
    title = str(meta.pop("title", "") or "").strip() or _first_h1(text) or doc_id
    meta["source_path"] = _relative_posix(path, root)
    return Document(doc_id=doc_id, title=title, text=text, metadata=meta)


def load_corpus(corpus_dir: str | Path) -> list[Document]:
    """Рекурсивно загружает все ``*.md`` из каталога, отсортированные по ``doc_id``.

    Пропускаем служебные файлы: ``README.md`` и всё, что начинается с ``_`` (черновики,
    шаблоны). Сортировка по ``doc_id`` нужна для детерминизма: порядок обхода файловой
    системы разный на разных ОС, а от порядка документов зависят номера в индексе.
    Два файла с одинаковым ``id`` → ``ValueError``: иначе один молча «затёр» бы другой.
    """
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"Каталог корпуса не найден: {corpus_dir}")

    docs: list[Document] = []
    path_by_id: dict[str, Path] = {}
    for path in sorted(corpus_dir.rglob("*.md")):
        if not path.is_file() or _is_service_file(path):
            continue
        doc = load_document(path, root=corpus_dir)
        if doc.doc_id in path_by_id:
            raise ValueError(f"Дубликат id '{doc.doc_id}': {path_by_id[doc.doc_id]} и {path}")
        path_by_id[doc.doc_id] = path
        docs.append(doc)

    if not docs:
        logger.warning("В каталоге %s не найдено ни одного документа", corpus_dir)
    logger.info("Загружено документов: %d (из %s)", len(docs), corpus_dir)
    return sorted(docs, key=lambda doc: doc.doc_id)


def corpus_fingerprint(docs: list[Document]) -> str:
    """Короткий sha256 содержимого корпуса: id, заголовок, текст и шапка каждого документа.

    Не зависит от порядка чтения файлов и от путей. Им помечают всё, что посчитано на корпусе:
    отчёт оценки (``provenance.corpus_sha256``) и файл калибровки порога отказа (порог,
    подобранный на одном корпусе, к другому не применяется).
    """
    rows = [[doc.doc_id, doc.title, doc.text, doc.metadata] for doc in sorted(docs, key=lambda d: d.doc_id)]
    # sort_keys — порядок ключей шапки не меняет хэш; default=str — для дат из шапок.
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _to_plain(value: Any) -> Any:
    """Приводит значения YAML к простым типам.

    YAML читает ``updated: 2025-03-01`` как объект ``datetime.date``. Дальше метаданные
    попадают в JSON-отчёты и в промпт, а сравнивать даты удобнее как строки ISO
    (они сортируются так же, как даты), поэтому сразу превращаем их в ``"2025-03-01"``.
    """
    if isinstance(value, dt.datetime):  # datetime — подкласс date, поэтому проверяем его первым
        return value.date().isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_plain(item) for key, item in value.items()}
    return value


def _first_h1(text: str) -> str | None:
    """Текст первого заголовка первого уровня (``# Название``) или None."""
    match = _H1_RE.search(text)
    return match.group(1).strip() if match else None


def _relative_posix(path: Path, root: Path | None) -> str:
    """Путь к файлу в виде ``hr/hr-vacation.md`` — с прямыми слэшами на любой ОС.

    Путь справочный (показывается в отчётах), поэтому, если файл лежит вне ``root``,
    не падаем, а возвращаем путь как есть.
    """
    if root is not None:
        try:
            return path.resolve().relative_to(Path(root).resolve()).as_posix()
        except ValueError:
            pass
    return path.as_posix()


def _is_service_file(path: Path) -> bool:
    return path.name.startswith("_") or path.name.lower() == "readme.md"
