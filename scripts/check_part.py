"""Дополнительные авторские проверки частей эталона (data/eval/parts/*.jsonl) против корпуса.

Главный валидатор данных — ``kbrag validate-data`` (его же запускает CI): структура gold.jsonl,
типы вопросов, дословность цитат. Этот скрипт — помощник автора набора: он ловит то, что
``validate-data`` пока не проверяет:

* у ``multi_hop`` цитаты минимум из 2 разных документов;
* у ``unanswerable`` пустой и эталонный ``answer``;
* каждый ``key_fact`` целиком есть в цитатах и в эталонном ответе (сравнение по стемам);
* во front matter документов есть title/department/status/updated, а status — active или archived;
* длина документов и цитат в разумных пределах (предупреждение, не ошибка).

Документы читаются тем же ``parse_front_matter``, что и в основном загрузчике (BOM и CRLF не мешают),
список типов вопросов — ``QUESTION_TYPES`` из ``kbrag.eval.dataset``: правила не расходятся с пакетом.

    python scripts/check_part.py data/eval/parts/hr.jsonl [--corpus data/corpus]

Код возврата 1 — есть ошибки; предупреждения не блокируют.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Скрипт можно запускать и без `pip install -e .`: код пакета берём прямо из src/.
sys.path.insert(0, str(ROOT / "src"))

from kbrag.eval.dataset import QUESTION_TYPES  # noqa: E402
from kbrag.ingest.loader import parse_front_matter  # noqa: E402
from kbrag.text import analyze, find_span  # noqa: E402

REQUIRED = {"id", "question", "answer", "key_facts", "evidence", "type"}
REQUIRED_META = ("id", "title", "department", "status", "updated")
# Допустимые длины (в словах). Одни и те же числа и проверяются, и печатаются в предупреждении.
DOC_WORDS = (250, 1200)
QUOTE_WORDS = (5, 60)  # data/DATASET.md советует 8–40 слов; здесь — мягкие границы


def load_docs(corpus: Path, errors: list[str]) -> dict[str, dict]:
    """Читает документы корпуса; проблемы с файлами дописывает в ``errors`` (а не только печатает)."""
    docs: dict[str, dict] = {}
    for path in sorted(corpus.rglob("*.md")):
        if path.name.startswith("_") or path.name == "README.md":
            continue
        # utf-8-sig убирает BOM, read_text превращает \r\n в \n — как в kbrag.ingest.loader.
        raw = path.read_text(encoding="utf-8-sig")
        try:
            meta, body = parse_front_matter(raw)
        except ValueError as exc:
            errors.append(f"{path}: {exc}")
            continue
        if not meta:
            errors.append(f"{path}: нет front matter")
            continue
        doc_id = meta.get("id")
        if doc_id in docs:
            errors.append(f"{path}: дубликат id {doc_id}")
        docs[doc_id] = {"meta": meta, "text": body.lstrip("\r\n"), "path": path}
    return docs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("parts", nargs="+", type=Path)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data" / "corpus")
    args = parser.parse_args(argv)

    errors: list[str] = []
    warnings: list[str] = []
    docs = load_docs(args.corpus, errors)
    seen: set[str] = set()
    types = collections.Counter()

    for doc_id, doc in docs.items():
        n_words = len(doc["text"].split())
        for field in REQUIRED_META:
            if field not in doc["meta"]:
                errors.append(f"{doc_id}: в front matter нет поля {field}")
        if doc["meta"].get("status") not in ("active", "archived"):
            errors.append(f"{doc_id}: status должен быть active|archived")
        if not DOC_WORDS[0] <= n_words <= DOC_WORDS[1]:
            warnings.append(f"{doc_id}: {n_words} слов (ожидается {DOC_WORDS[0]}–{DOC_WORDS[1]})")

    for part in args.parts:
        for line_no, line in enumerate(part.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            where = f"{part.name}:{line_no}"
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{where}: невалидный JSON ({exc})")
                continue
            missing = REQUIRED - item.keys()
            if missing:
                errors.append(f"{where}: нет полей {sorted(missing)}")
                continue
            qid = item["id"]
            where = f"{where} [{qid}]"
            if qid in seen:
                errors.append(f"{where}: дубликат id")
            seen.add(qid)
            qtype = item["type"]
            types[qtype] += 1
            if qtype not in QUESTION_TYPES:
                errors.append(f"{where}: неизвестный type {qtype}")
            if qtype == "unanswerable":
                if item["evidence"] or item["key_facts"] or item["answer"]:
                    errors.append(f"{where}: у unanswerable должны быть пустые evidence/key_facts/answer")
                continue
            if not item["evidence"]:
                errors.append(f"{where}: нет evidence")
            if not item["key_facts"]:
                errors.append(f"{where}: нет key_facts")
            if qtype == "multi_hop" and len({e.get("doc_id") for e in item["evidence"]}) < 2:
                errors.append(f"{where}: multi_hop должен ссылаться минимум на 2 документа")
            quote_stems: set[str] = set()
            for ev in item["evidence"]:
                doc = docs.get(ev.get("doc_id"))
                if doc is None:
                    errors.append(f"{where}: неизвестный doc_id {ev.get('doc_id')}")
                    continue
                quote = ev.get("quote", "")
                n_quote_words = len(quote.split())
                if find_span(doc["text"], quote) is None:
                    errors.append(f"{where}: цитата не найдена дословно в {ev['doc_id']}: «{quote[:80]}»")
                elif not QUOTE_WORDS[0] <= n_quote_words <= QUOTE_WORDS[1]:
                    warnings.append(
                        f"{where}: цитата из {n_quote_words} слов (ожидается {QUOTE_WORDS[0]}–{QUOTE_WORDS[1]})"
                    )
                quote_stems.update(analyze(quote, remove_stopwords=False))
            for fact in item["key_facts"]:
                fact_stems = set(analyze(fact, remove_stopwords=False))
                if not fact_stems:
                    errors.append(f"{where}: пустой key_fact «{fact}»")
                elif not fact_stems <= quote_stems:
                    warnings.append(f"{where}: key_fact «{fact}» не целиком содержится в цитатах")
            answer_stems = set(analyze(item["answer"], remove_stopwords=False))
            for fact in item["key_facts"]:
                if not set(analyze(fact, remove_stopwords=False)) <= answer_stems:
                    warnings.append(f"{where}: key_fact «{fact}» отсутствует в эталонном answer")

    by_type = dict(sorted(types.items()))
    print(f"Документов: {len(docs)}; вопросов: {sum(types.values())}; по типам: {by_type}")
    for w in warnings:
        print("WARN ", w)
    for e in errors:
        print("ERROR", e)
    print("OK" if not errors else f"ОШИБОК: {len(errors)}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
