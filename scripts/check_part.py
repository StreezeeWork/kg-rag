"""Проверка части эталонного набора (data/eval/parts/*.jsonl) против корпуса.

Независим от пакетов ingest/eval (использует только kbrag.text), поэтому им можно
пользоваться, пока основной код ещё пишется. Запуск:

    python scripts/check_part.py data/eval/parts/hr.jsonl [--corpus data/corpus]

Код возврата 1 — есть ошибки; предупреждения не блокируют.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kbrag.text import analyze, find_span  # noqa: E402

TYPES = {"factoid", "numeric", "procedural", "paraphrase", "keyword", "confusable", "multi_hop", "unanswerable"}
REQUIRED = {"id", "question", "answer", "key_facts", "evidence", "type"}


def load_docs(corpus: Path) -> dict[str, dict]:
    docs: dict[str, dict] = {}
    for path in sorted(corpus.rglob("*.md")):
        if path.name.startswith("_") or path.name == "README.md":
            continue
        raw = path.read_text(encoding="utf-8")
        if not raw.startswith("---"):
            print(f"ERROR {path}: нет front matter")
            continue
        _, fm, body = raw.split("---", 2)
        meta = yaml.safe_load(fm) or {}
        doc_id = meta.get("id")
        if doc_id in docs:
            print(f"ERROR {path}: дубликат id {doc_id}")
        docs[doc_id] = {"meta": meta, "text": body.lstrip("\n"), "path": path}
    return docs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("parts", nargs="+", type=Path)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data" / "corpus")
    args = parser.parse_args()

    docs = load_docs(args.corpus)
    errors: list[str] = []
    warnings: list[str] = []
    seen: set[str] = set()
    types = collections.Counter()

    for doc_id, doc in docs.items():
        n_words = len(doc["text"].split())
        for field in ("id", "title", "department", "status", "updated"):
            if field not in doc["meta"]:
                errors.append(f"{doc_id}: в front matter нет поля {field}")
        if doc["meta"].get("status") not in ("active", "archived"):
            errors.append(f"{doc_id}: status должен быть active|archived")
        if not 250 <= n_words <= 1200:
            warnings.append(f"{doc_id}: {n_words} слов (ожидается 350–900)")

    for part in args.parts:
        for line_no, line in enumerate(part.read_text(encoding="utf-8").splitlines(), 1):
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
            if qtype not in TYPES:
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
                elif not 5 <= n_quote_words <= 60:
                    warnings.append(f"{where}: цитата из {n_quote_words} слов (ожидается 8–40)")
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

    print(f"Документов: {len(docs)}; вопросов: {sum(types.values())}; по типам: {dict(sorted(types.items()))}")
    for w in warnings:
        print("WARN ", w)
    for e in errors:
        print("ERROR", e)
    print("OK" if not errors else f"ОШИБОК: {len(errors)}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
