"""Тесты эталонного набора: чтение/запись, валидация разметки, релевантность, разбиение dev/test."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from kbrag.eval.dataset import (
    QUESTION_TYPES,
    Evidence,
    GoldItem,
    assign_splits,
    evidence_spans,
    is_relevant,
    load_gold,
    save_gold,
    validate_gold,
)
from kbrag.types import Chunk, Document


def make_item(
    qid: str,
    qtype: str = "factoid",
    split: str = "test",
    evidence: tuple[Evidence, ...] = (),
    key_facts: tuple[str, ...] = ("факт",),
    question: str = "Вопрос?",
) -> GoldItem:
    return GoldItem(
        qid=qid,
        question=question,
        answer="ответ",
        key_facts=key_facts,
        evidence=evidence,
        qtype=qtype,
        split=split,
    )


def make_chunk(start: int, end: int, doc_id: str = "d") -> Chunk:
    # Для is_relevant важны только doc_id и смещения; текст — заглушка нужной длины.
    return Chunk(chunk_id=f"{doc_id}::{start}", doc_id=doc_id, text="x" * (end - start), start=start, end=end)


# ---------------------------------------------------------------------------
# load_gold / save_gold
# ---------------------------------------------------------------------------


def test_load_gold_reads_all_fields(mini_gold_path: Path) -> None:
    items = load_gold(mini_gold_path)

    assert [it.qid for it in items] == ["t-001", "t-002", "t-003", "t-004", "t-005"]
    first = items[0]
    assert first.qtype == "numeric"
    assert first.split == "test"
    assert first.key_facts == ("28 календарных дней",)
    assert first.evidence == (
        Evidence(
            "hr-vacation",
            "Продолжительность ежегодного оплачиваемого отпуска составляет 28 календарных дней.",
        ),
    )
    assert [it.answerable for it in items] == [True, True, True, True, False]


def test_gold_item_is_hashable_and_immutable(mini_gold_path: Path) -> None:
    item = load_gold(mini_gold_path)[0]
    assert item in {item}
    with pytest.raises(AttributeError):
        item.split = "dev"  # type: ignore[misc]


def test_load_gold_skips_blank_lines_and_allows_missing_split(tmp_path: Path) -> None:
    # Так выглядят части набора до `kbrag build-gold`: split ещё не назначен.
    record = {
        "id": "q1",
        "question": "Вопрос?",
        "answer": "",
        "key_facts": [],
        "evidence": [],
        "type": "unanswerable",
    }
    path = tmp_path / "part.jsonl"
    path.write_text("\n" + json.dumps(record, ensure_ascii=False) + "\n\n", encoding="utf-8")

    items = load_gold(path)

    assert len(items) == 1
    assert items[0].split == ""


def test_load_gold_tolerates_utf8_bom(tmp_path: Path, mini_gold_records: list[dict]) -> None:
    path = tmp_path / "bom.jsonl"
    path.write_text(json.dumps(mini_gold_records[0], ensure_ascii=False) + "\n", encoding="utf-8-sig")
    assert load_gold(path)[0].qid == "t-001"


@pytest.mark.parametrize(
    ("bad_line", "message"),
    [
        ("{not json", "2:"),
        ('{"question": "без id", "type": "factoid"}', "id"),
        ('{"id": "q", "question": "?", "type": "factoid", "key_facts": "строка"}', "key_facts"),
        ('{"id": "q", "question": "?", "type": "factoid", "evidence": [{"doc_id": "d"}]}', "quote"),
        ("[1, 2]", "JSON-объектом"),
    ],
)
def test_load_gold_reports_broken_line(
    tmp_path: Path, mini_gold_records: list[dict], bad_line: str, message: str
) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text(
        json.dumps(mini_gold_records[0], ensure_ascii=False) + "\n" + bad_line + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match=message) as excinfo:
        load_gold(path)
    assert ":2:" in str(excinfo.value)  # номер строки помогает найти ошибку в большом файле


def test_save_and_load_roundtrip(tmp_path: Path, mini_gold_path: Path) -> None:
    items = load_gold(mini_gold_path)
    out = tmp_path / "nested" / "gold.jsonl"

    save_gold(items, out)

    assert load_gold(out) == items
    raw = out.read_bytes()
    assert "Продолжительность".encode() in raw  # кириллица не экранирована
    assert b"\r\n" not in raw  # одинаковые переводы строк на любой ОС


# ---------------------------------------------------------------------------
# validate_gold
# ---------------------------------------------------------------------------


def test_validate_gold_accepts_mini_dataset(mini_gold_path: Path, mini_docs: list[Document]) -> None:
    assert validate_gold(load_gold(mini_gold_path), mini_docs) == []


def test_validate_gold_tolerates_whitespace_differences_in_quote(mini_docs: list[Document]) -> None:
    quote = "Продолжительность ежегодного\nоплачиваемого   отпуска составляет 28 календарных дней."
    item = make_item("q1", evidence=(Evidence("hr-vacation", quote),))
    assert validate_gold([item], mini_docs) == []


def test_validate_gold_finds_every_error_category(mini_docs: list[Document]) -> None:
    good_evidence = (Evidence("it-vpn", "Для подключения к VPN используется клиент «Шлюз»"),)
    items = [
        make_item("ok", evidence=good_evidence),
        make_item("ok", evidence=good_evidence),  # дубликат id
        make_item("bad-doc", evidence=(Evidence("no-such-doc", "что угодно"),)),
        make_item("bad-quote", evidence=(Evidence("it-vpn", "Этой фразы в документе нет"),)),
        make_item("no-evidence", evidence=()),
        make_item("no-facts", evidence=good_evidence, key_facts=()),
        make_item("unans-ev", qtype="unanswerable", evidence=good_evidence, key_facts=()),
        make_item("unans-facts", qtype="unanswerable", key_facts=("лишний факт",)),
        make_item("bad-type", qtype="trivia", evidence=good_evidence),
        make_item("bad-split", split="train", evidence=good_evidence),
        make_item("empty-q", evidence=good_evidence, question="   "),
    ]
    expected = [
        ("ok", "повторяющийся id"),
        ("bad-doc", "неизвестный doc_id"),
        ("bad-quote", "цитата не найдена"),
        ("no-evidence", "нет evidence"),
        ("no-facts", "нет key_facts"),
        ("unans-ev", "не должно быть evidence"),
        ("unans-facts", "не должно быть key_facts"),
        ("bad-type", "неизвестный type"),
        ("bad-split", "неизвестный split"),
        ("empty-q", "пустой вопрос"),
    ]

    errors = validate_gold(items, mini_docs)

    for qid, fragment in expected:
        assert any(e.startswith(f"{qid}:") and fragment in e for e in errors), (qid, fragment, errors)
    # Ровно по одной ошибке на сломанную запись: корректные записи не дают ложных срабатываний.
    assert len(errors) == len(expected)


# ---------------------------------------------------------------------------
# evidence_spans / is_relevant
# ---------------------------------------------------------------------------


def test_evidence_spans_maps_quotes_to_offsets(mini_gold_path: Path, mini_docs: list[Document]) -> None:
    docs_by_id = {d.doc_id: d for d in mini_docs}
    items = load_gold(mini_gold_path)

    for item in items:
        spans = evidence_spans(item, docs_by_id)
        assert len(spans) == len(item.evidence)
        for (doc_id, start, end), ev in zip(spans, item.evidence, strict=True):
            assert doc_id == ev.doc_id
            assert docs_by_id[doc_id].text[start:end] == ev.quote


def test_evidence_spans_raises_on_missing_quote(mini_docs: list[Document]) -> None:
    docs_by_id = {d.doc_id: d for d in mini_docs}
    item = make_item("q", evidence=(Evidence("it-vpn", "несуществующая цитата"),))
    with pytest.raises(ValueError, match="цитата не найдена"):
        evidence_spans(item, docs_by_id)
    with pytest.raises(ValueError, match="doc_id"):
        evidence_spans(make_item("q", evidence=(Evidence("nope", "x"),)), docs_by_id)


SPAN = ("d", 100, 200)  # цитата длиной 100 символов


@pytest.mark.parametrize(
    ("chunk", "expected", "why"),
    [
        (make_chunk(0, 300), True, "фрагмент целиком содержит цитату"),
        (make_chunk(150, 170), True, "маленький фрагмент целиком внутри цитаты"),
        (make_chunk(0, 160), True, "покрыто 60 из 100 символов (≥ 50%)"),
        (make_chunk(0, 150), True, "ровно половина цитаты — граница включительно"),
        (make_chunk(0, 120), False, "покрыто 20 из 100 символов (< 50%)"),
        (make_chunk(190, 400), False, "фрагмент задел край цитаты: 10 из 100"),
        (make_chunk(195, 205), True, "короткий фрагмент: 5 из 10 его символов — цитата"),
        (make_chunk(200, 300), False, "соприкасается, но не пересекается"),
        (make_chunk(100, 200, doc_id="other"), False, "другой документ"),
    ],
)
def test_is_relevant_overlap_rule(chunk: Chunk, expected: bool, why: str) -> None:
    assert is_relevant(chunk, SPAN) is expected, why


def test_is_relevant_respects_min_coverage() -> None:
    chunk = make_chunk(0, 130)  # покрыто 30 из 100
    assert not is_relevant(chunk, SPAN, min_coverage=0.5)
    assert is_relevant(chunk, SPAN, min_coverage=0.3)


# ---------------------------------------------------------------------------
# assign_splits
# ---------------------------------------------------------------------------


@pytest.fixture()
def many_items() -> list[GoldItem]:
    """По 10 вопросов каждого типа, все пока в test."""
    return [make_item(f"{qtype}-{i:02d}", qtype=qtype) for qtype in QUESTION_TYPES for i in range(10)]


def test_assign_splits_is_stratified_by_type(many_items: list[GoldItem]) -> None:
    result = assign_splits(many_items, dev_fraction=0.3)

    dev_per_type = Counter(it.qtype for it in result if it.split == "dev")
    assert dev_per_type == {qtype: 3 for qtype in QUESTION_TYPES}
    assert {it.split for it in result} == {"dev", "test"}
    assert sum(it.split == "dev" for it in result) / len(result) == pytest.approx(0.3)


def test_assign_splits_is_deterministic_and_order_independent(many_items: list[GoldItem]) -> None:
    first = {it.qid: it.split for it in assign_splits(many_items)}
    again = {it.qid: it.split for it in assign_splits(many_items)}
    reversed_order = {it.qid: it.split for it in assign_splits(list(reversed(many_items)))}

    assert first == again == reversed_order


def test_assign_splits_keeps_order_and_other_fields(many_items: list[GoldItem]) -> None:
    result = assign_splits(many_items)

    assert [it.qid for it in result] == [it.qid for it in many_items]
    for before, after in zip(many_items, result, strict=True):
        assert (before.question, before.qtype, before.evidence) == (
            after.question,
            after.qtype,
            after.evidence,
        )


def test_assign_splits_seed_changes_assignment(many_items: list[GoldItem]) -> None:
    a = [it.split for it in assign_splits(many_items, seed=13)]
    b = [it.split for it in assign_splits(many_items, seed=14)]
    assert a != b
    assert a.count("dev") == b.count("dev")  # меняется состав, но не пропорция


def test_assign_splits_edge_fractions(many_items: list[GoldItem]) -> None:
    assert {it.split for it in assign_splits(many_items, dev_fraction=0.0)} == {"test"}
    assert {it.split for it in assign_splits(many_items, dev_fraction=1.0)} == {"dev"}
    with pytest.raises(ValueError):
        assign_splits(many_items, dev_fraction=1.5)
