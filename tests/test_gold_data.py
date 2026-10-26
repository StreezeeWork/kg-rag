"""Регрессионные тесты реальных данных: корпус ``data/corpus`` и эталон ``data/eval``.

Зачем отдельный файл: ``test_dataset.py`` проверяет *код* на мини-корпусе из фикстур,
а ошибки разметки живут в *самих данных*. Здесь закреплены правила из ``data/DATASET.md``,
которые однажды нарушались и были исправлены: если правку данных случайно откатят
или новый вопрос повторит старую ошибку, тест упадёт и покажет, где именно.

Тесты читают части набора (``data/eval/parts``), а не только ``gold.jsonl``: так ошибка
видна сразу после правки части, ещё до ``kbrag build-gold``.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest

from kbrag.eval.dataset import GoldItem, assign_splits, load_gold, validate_gold
from kbrag.eval.metrics import fact_recall
from kbrag.ingest import load_corpus
from kbrag.text import analyze, split_sentences
from kbrag.types import Document

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "data" / "corpus"
PARTS_DIR = ROOT / "data" / "eval" / "parts"
GOLD_PATH = ROOT / "data" / "eval" / "gold.jsonl"

# Сколько общих стемов вопроса paraphrase с цитатой-доказательством ещё допустимо.
# Служебные совпадения («один», «каждый») неизбежны, а три и больше — уже подсказка поиску.
MAX_PARAPHRASE_QUOTE_OVERLAP = 2


@pytest.fixture(scope="module")
def docs() -> dict[str, Document]:
    return {doc.doc_id: doc for doc in load_corpus(CORPUS_DIR)}


@pytest.fixture(scope="module")
def items() -> list[GoldItem]:
    # Тот же порядок, что у `kbrag build-gold`: файлы частей по алфавиту.
    return [item for path in sorted(PARTS_DIR.glob("*.jsonl")) for item in load_gold(path)]


@pytest.fixture(scope="module")
def by_qid(items: list[GoldItem]) -> dict[str, GoldItem]:
    return {item.qid: item for item in items}


def _sentences(doc: Document) -> list[str]:
    return [doc.text[start:end] for start, end in split_sentences(doc.text)]


def _facts_found_in(doc: Document, facts: Iterable[str]) -> set[str]:
    """Ключевые факты, которые целиком (всеми стемами) встречаются в одном предложении документа.

    Предложение, а не весь документ: в длинном тексте стемы факта почти всегда найдутся
    по отдельности («3» в одном пункте, «дня» в другом), и проверка потеряла бы смысл.
    """
    sentences = _sentences(doc)
    return {fact for fact in facts if any(fact_recall(s, [fact]) == 1.0 for s in sentences)}


# ---------------------------------------------------------------------------
# Целостность набора
# ---------------------------------------------------------------------------


def test_gold_file_is_valid_and_built_from_parts(docs: dict[str, Document], items: list[GoldItem]) -> None:
    """gold.jsonl = части + детерминированный split; все цитаты находятся в корпусе."""
    gold = load_gold(GOLD_PATH)
    assert validate_gold(gold, list(docs.values())) == []
    assert gold == assign_splits(items), "gold.jsonl устарел: запустите `kbrag build-gold`"


# ---------------------------------------------------------------------------
# Типы вопросов соответствуют определениям из DATASET.md
# ---------------------------------------------------------------------------

# Автоматически по всем multi_hop эту проверку не запустить: лексическое сравнение даёт ложные
# срабатывания («15:00» есть и в правилах обеда, и в правилах доступа в офис). Поэтому
# проверяем вопросы, которые были ошибочно размечены как multi_hop и исправлены, —
# и всё, что добавляется в этот список при написании новых multi_hop.
MULTI_HOP_CHECKED = ["it-039", "x-001", "x-007", "x-009"]


@pytest.mark.parametrize("qid", MULTI_HOP_CHECKED)
def test_multi_hop_needs_every_evidence_document(
    qid: str, by_qid: dict[str, GoldItem], docs: dict[str, Document]
) -> None:
    """multi_hop = каждый документ из evidence даёт факт, которого нет в остальных.

    Иначе вопрос отвечается по одному документу, а Recall@k и citation_recall штрафуют
    систему, которая нашла этот документ и ответила полностью (так было с it-039, x-001, x-007).
    """
    item = by_qid[qid]
    assert item.qtype == "multi_hop"
    doc_ids = sorted({ev.doc_id for ev in item.evidence})
    assert len(doc_ids) >= 2

    found = {doc_id: _facts_found_in(docs[doc_id], item.key_facts) for doc_id in doc_ids}
    assert set().union(*found.values()) == set(item.key_facts), f"{qid}: не все факты есть в evidence"
    for doc_id in doc_ids:
        others = set().union(*(found[other] for other in doc_ids if other != doc_id))
        assert found[doc_id] - others, f"{qid}: {doc_id} не добавляет ни одного своего факта"


def test_paraphrase_questions_do_not_copy_the_quote(items: list[GoldItem]) -> None:
    """paraphrase проверяет поиск по смыслу: вопрос не повторяет слова цитаты (было у fin-035).

    С остальным текстом документа общие слова допустимы — без них вопрос не задать.
    """
    offenders: dict[str, list[str]] = {}
    for item in items:
        if item.qtype != "paraphrase":
            continue
        quote_stems = set().union(*(set(analyze(ev.quote)) for ev in item.evidence))
        shared = set(analyze(item.question)) & quote_stems
        if len(shared) > MAX_PARAPHRASE_QUOTE_OVERLAP:
            offenders[item.qid] = sorted(shared)
    assert offenders == {}


# ---------------------------------------------------------------------------
# Непротиворечивость мира: «каждый факт живёт ровно в одном документе»
# ---------------------------------------------------------------------------

# (doc_id, фрагмент текста, должен ли он быть в документе). Каждая строка — исправленное
# противоречие; комментарий объясняет, что сломается, если его вернуть.
WORLD_RULES = [
    # Кто запускает выкладку в production, решает только «Релизный процесс». Дубль в CI/CD
    # делал второй равноценный ответ на x-009, которого не было в evidence.
    pytest.param("eng-ci-cd", "доступен только релиз-менеджеру", False, id="deploy-owner-only-in-release"),
    # Шестой этап (deploy-staging) идёт после слияния в main, а не в пайплайне merge request.
    pytest.param("eng-ci-cd", "Пайплайн merge request состоит из шести", False, id="ci-six-stages"),
]


@pytest.mark.parametrize(("doc_id", "fragment", "present"), WORLD_RULES)
def test_world_rule(docs: dict[str, Document], doc_id: str, fragment: str, present: bool) -> None:
    text = " ".join(docs[doc_id].text.split())  # переносы строк внутри абзаца не важны
    assert (fragment in text) is present
