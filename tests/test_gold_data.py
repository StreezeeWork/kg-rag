"""Регрессионные тесты реальных данных: корпус ``data/corpus`` и эталон ``data/eval``.

Зачем отдельный файл: ``test_dataset.py`` проверяет *код* на мини-корпусе из фикстур,
а ошибки разметки живут в *самих данных*. Здесь закреплены правила из ``data/DATASET.md``,
которые однажды нарушались и были исправлены: если правку данных случайно откатят
или новый вопрос повторит старую ошибку, тест упадёт и покажет, где именно.

Тесты читают части набора (``data/eval/parts``), а не только ``gold.jsonl``: так ошибка
видна сразу после правки части, ещё до ``kbrag build-gold``.
"""

from __future__ import annotations

import re
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

# Действующий документ → его архивная редакция («двойник» с другими значениями).
# Новый архивный документ нужно добавить сюда: тест ниже это проверяет.
ARCHIVED_TWINS = {
    "hr-remote-2025": "hr-remote-2023",
    "it-passwords": "it-passwords-2023",
    "eng-release-process": "eng-release-process-2023",
}


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


def _sections(doc: Document) -> list[str]:
    """Разделы документа по заголовкам ``##``: примерно столько текста видит генератор во фрагменте."""
    return [part for part in re.split(r"(?m)^(?=## )", doc.text) if part.strip()]


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
# Ключевые факты отличают верный ответ от неверного
# ---------------------------------------------------------------------------


def test_no_key_fact_is_answered_by_the_question_itself(items: list[GoldItem]) -> None:
    """Факт, все стемы которого есть в вопросе, засчитывается ответу-повтору вопроса.

    Так было у it-004 («ВНЕШНИЙ» при вопросе «…с внешних адресов»), hr-018 и it-037:
    FactRecall получал баллы за пересказ вопроса, а не за ответ.
    """
    offenders = [
        (item.qid, fact)
        for item in items
        for fact in item.key_facts
        if fact_recall(item.question, [fact]) == 1.0
    ]
    assert offenders == []


def test_every_archived_document_has_a_registered_twin(docs: dict[str, Document]) -> None:
    archived = sorted(doc_id for doc_id, doc in docs.items() if doc.metadata.get("status") == "archived")
    assert archived == sorted(ARCHIVED_TWINS.values())


def test_confusable_key_facts_reject_an_answer_from_the_archive(
    items: list[GoldItem], docs: dict[str, Document]
) -> None:
    """confusable проверяет, не перепутала ли система действующую норму с архивной.

    Значит, ответ, собранный из любого раздела архивного двойника, не должен набирать
    FactRecall = 1. Раньше так «проходили» it-037 (архивное «допускаются SMS-коды» содержит
    те же стемы, что «SMS-коды … не допускаются»: «не» — стоп-слово) и hr-009
    («до 3 дней» совпадало с архивным «за 3 рабочих дня»).
    """
    offenders: dict[str, str] = {}
    for item in items:
        if item.qtype != "confusable":
            continue
        for doc_id in {ev.doc_id for ev in item.evidence} & ARCHIVED_TWINS.keys():
            for section in _sections(docs[ARCHIVED_TWINS[doc_id]]):
                if fact_recall(section, item.key_facts) == 1.0:
                    offenders[item.qid] = section.splitlines()[0]
    assert offenders == {}


# (qid, факт, который должен быть среди key_facts, факт, которого там быть не должно).
# Каждая строка — исправленный «слабый» факт; без него метрика снова начнёт ошибаться.
KEY_FACT_FIXES = [
    # Главное для сотрудника — отпуск продлевается; без этого факта неполный ответ получал 1.0.
    pytest.param("hr-008", "отпуск продлевается", None, id="hr-008-main-fact"),
    # «Обучение» засчитывалось любому ответу со словом «обучение», даже про другую процедуру.
    pytest.param("hr-026", "счёт провайдера", "Обучение", id="hr-026-generic-word"),
    # Одно общее слово «запрещено» (19 из 48 документов) пропускало ответ из чужого документа.
    pytest.param("eng-018", "Наказание", None, id="eng-018-single-common-word"),
    # Факт не может быть одной из альтернатив «А или Б»: верный ответ «звоните 1111» терял балл.
    # Кроме того, при утечке регламент ИБ разрешает только звонок.
    pytest.param("eng-028", "1111", "security@nimbus.example", id="eng-028-or-alternative"),
    # «автоматически» есть в 24 документах; факт должен быть про VPN, а не про что угодно.
    pytest.param("x-003", "открывается автоматически", "автоматически", id="x-003-generic-word"),
]


@pytest.mark.parametrize(("qid", "required", "forbidden"), KEY_FACT_FIXES)
def test_key_fact_fix_is_in_place(
    by_qid: dict[str, GoldItem], qid: str, required: str | None, forbidden: str | None
) -> None:
    facts = by_qid[qid].key_facts
    if required is not None:
        assert required in facts
    if forbidden is not None:
        assert forbidden not in facts


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
    # При утечке секрета регламент ИБ требует звонить на 1111; почта «в рабочее время, если не
    # срочно» — неверный совет для утечки, а «или» делало каналы равноценными.
    pytest.param(
        "eng-secrets",
        "security@nimbus.example или по внутреннему номеру 1111",
        False,
        id="secret-leak-channel",
    ),
    # Решение об откате принимает релиз-менеджер («Релизный процесс»); IC только запрашивает
    # откат. Иначе у confusable-вопроса eng-009 два «действующих» ответа.
    pytest.param("eng-incident-severity", "принимает решения об откате", False, id="rollback-decision-owner"),
    # VPN обязателен в командировке, но геофильтр открывался только для «Работы из-за рубежа»,
    # а политика удалённой работы командировки не покрывает: в зарубежной командировке VPN не работал.
    pytest.param(
        "it-vpn", "На время зарубежной командировки доступ из-за рубежа", True, id="vpn-trip-abroad"
    ),
    # Всем сотрудникам нужна «Казна» (командировки, авансовые отчёты), но в базовом наборе её не было.
    pytest.param("it-access-requests", "роль «Сотрудник» в «Казне»", True, id="kazna-in-base-access"),
    # Права локального администратора (8 часов, «Установка ПО») ≠ права администратора системы (CISO).
    pytest.param(
        "it-access-requests", "к правам администратора системы не относятся", True, id="local-admin"
    ),
    # Подрядчикам почтовый ящик не выдаётся («Работа с подрядчиками»).
    pytest.param("it-passwords", "подрядчикам почтовый ящик не выдаётся", True, id="contractor-mail"),
    # «NDA и коммерческая тайна» требует подтвердить уничтожение носителей в обходном листе.
    pytest.param("hr-offboarding", "носители с конфиденциальной информацией", True, id="media-in-clearance"),
    # Пороги закупок не пересекаются: ровно 100 000 рублей — первый уровень.
    pytest.param(
        "fin-procurement", "до 100 000 рублей с НДС включительно", True, id="procurement-lower-tier"
    ),
    pytest.param("fin-procurement", "Закупки от 100 000", False, id="procurement-no-overlap"),
    # Срок заявки на ночную работу в будний день раньше не был определён.
    pytest.param("fin-office-access", "Для работы ночью в будний день", True, id="weekday-night-access"),
    # Этап «оценка руководителя и калибровка (2 недели)» противоречил калибровке «в течение 2 недель после».
    pytest.param("hr-performance-review", "калибровка (2 недели)", False, id="calibration-timing"),
    # «Офисный 0–1» и «Гибридный до 3» пересекались: 1 удалённый день попадал в оба формата.
    pytest.param("hr-remote-2025", "| Гибридный | до 3 |", False, id="hybrid-format-range"),
]


@pytest.mark.parametrize(("doc_id", "fragment", "present"), WORLD_RULES)
def test_world_rule(docs: dict[str, Document], doc_id: str, fragment: str, present: bool) -> None:
    text = " ".join(docs[doc_id].text.split())  # переносы строк внутри абзаца не важны
    assert (fragment in text) is present


def test_onboarding_signs_nda_before_first_login(docs: dict[str, Document]) -> None:
    """«NDA и коммерческая тайна»: без подписанного NDA доступ к системам не предоставляется.

    Значит, в первый день сначала подписание, потом вход. Раньше онбординг выдавал ноутбук
    и учётную запись до подписания договора — прямое противоречие внутри мира.
    """
    assert (
        "без подписанного обязательства доступ к информационным системам не предоставляется"
        in docs["fin-nda"].text
    )
    onboarding = docs["hr-onboarding"].text
    assert onboarding.index("обязательства о неразглашении") < onboarding.index("впервые войти")


# Женские названия систем без родового слова после «в»/«через» не бывают в именительном падеже:
# правильно «в «Казне»», «через «Казну»», «в «Волне»», «в «Орбите»» (норма — в DATASET.md).
# С родовым словом название не склоняется: «в системе «Казна»», «в мессенджере «Волна»».
_UNDECLINED_NAME_RE = re.compile(r"(?<![\w-])(?:в|через)\s+«(?:Казна|Волна|Орбита)»")


def test_system_names_are_declined_consistently(docs: dict[str, Document]) -> None:
    """Корпус писали «разные авторы»: HR и финансы не склоняли названия, ИТ и инженеры — склоняли.

    Для поиска разница почти незаметна (стеммер сводит формы), но корпус, собранный
    «из разных рук», выглядит неряшливо, а DATASET.md требует единых названий.
    """
    offenders = {
        doc_id: _UNDECLINED_NAME_RE.findall(doc.text)
        for doc_id, doc in docs.items()
        if _UNDECLINED_NAME_RE.search(doc.text)
    }
    assert offenders == {}
