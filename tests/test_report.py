"""Тесты структуры результата (results.py) и отчёта (report.py) на синтетических экспериментах."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from kbrag.eval.report import (
    format_number,
    load_results,
    render_leaderboard,
    safe_file_name,
    write_report,
)
from kbrag.eval.results import ExperimentResult, metric_names, retrieval_metric_names


def split_metrics(recall5: float, **overrides) -> dict:
    """Плоский словарь метрик сплита: всё по умолчанию 0.5, кроме заданного."""
    metrics = {name: 0.5 for name in metric_names()}
    metrics.update({"recall@5": recall5, "n_questions": 4, "latency_p50_ms": 12.3456, "latency_p95_ms": 40.0})
    metrics.update(overrides)
    return metrics


def make_result(name: str, recall5: float, by_type: dict | None = None, **overrides) -> ExperimentResult:
    return ExperimentResult(
        name=name,
        config={"retrieval": {"mode": name}},
        metrics={
            "test": split_metrics(recall5, **overrides),
            "dev": split_metrics(0.111),
            "all": split_metrics(0.222),
            "by_type": by_type or {},
        },
        per_question=[
            {"qid": "t-001", "type": "numeric", "answer": "28 календарных дней [1].", "recall@5": 1.0},
            {"qid": "t-005", "type": "unanswerable", "answer": "В базе знаний нет ответа.", "recall@5": None},
        ],
        threshold=0.4242,
        index_stats={"n_docs": 4, "n_chunks": 12},
    )


@pytest.fixture()
def results() -> list[ExperimentResult]:
    return [
        make_result(
            "bm25",
            0.8,
            by_type={
                "numeric": {"recall@5": 1.0},
                "keyword": {"recall@5": 0.5},
                "unanswerable": {"recall@5": None},
            },
            abstain_f1=None,
        ),
        ExperimentResult.for_skipped(
            "dense-e5", "нет пакета sentence-transformers\n(pip install -e .[neural])"
        ),
        make_result(
            "hybrid",
            0.9,
            by_type={"numeric": {"recall@5": 1.0}, "keyword": {"recall@5": 2 / 3}},
            mrr=1 / 3,
        ),
    ]


def table_rows(markdown: str, first_header_cell: str) -> list[list[str]]:
    """Строки markdown-таблицы, начинающейся с заданного заголовка, как списки ячеек."""
    lines = markdown.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"| {first_header_cell} |"))
    rows = []
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip("|").split(" | ")])
    return rows


# ---------------------------------------------------------------------------
# ExperimentResult
# ---------------------------------------------------------------------------


def test_metric_names_follow_contract() -> None:
    names = metric_names()
    assert names[:4] == ["hit@1", "hit@3", "hit@5", "hit@10"]
    for expected in ("recall@10", "precision@5", "ndcg@3", "mrr", "fact_recall_answered", "abstain_f1"):
        assert expected in names
    assert names[-1] == "n_questions"
    assert retrieval_metric_names((2,)) == ["hit@2", "recall@2", "precision@2", "ndcg@2", "mrr"]


def test_experiment_result_roundtrip(results: list[ExperimentResult]) -> None:
    for result in results:
        restored = ExperimentResult.from_dict(json.loads(json.dumps(result.to_dict())))
        assert restored == result


def test_to_dict_is_a_deep_copy(results: list[ExperimentResult]) -> None:
    data = results[0].to_dict()
    data["metrics"]["test"]["recall@5"] = 0.0
    assert results[0].metric("recall@5") == 0.8


def test_from_dict_fills_optional_fields_and_requires_name() -> None:
    result = ExperimentResult.from_dict({"name": "old-format"})
    assert result.skipped is None and result.threshold is None
    assert result.metrics == {} and result.per_question == []
    with pytest.raises(ValueError):
        ExperimentResult.from_dict({"metrics": {}})


def test_metric_lookup_and_skipped_flag(results: list[ExperimentResult]) -> None:
    bm25, dense, _ = results
    assert bm25.metric("recall@5") == 0.8
    assert bm25.metric("recall@5", split="dev") == 0.111
    assert bm25.metric("no-such-metric") is None
    assert not bm25.is_skipped
    assert dense.is_skipped and dense.metric("recall@5") is None


# ---------------------------------------------------------------------------
# Форматирование
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.9, "0.900"),
        (1 / 3, "0.333"),
        (2 / 3, "0.667"),
        (12.3456, "12.346"),
        (4, "4"),
        (None, "—"),
        (math.nan, "—"),
        (np.float32(0.25), "0.250"),
        (np.int64(7), "7"),
    ],
)
def test_format_number(value, expected: str) -> None:
    assert format_number(value) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("bm25", "bm25"),
        ("hybrid+rerank", "hybrid_rerank"),
        ("exp:a/b", "exp_a_b"),
        ("чанк-300", "чанк-300"),
        ("::", "experiment"),
    ],
)
def test_safe_file_name(name: str, expected: str) -> None:
    assert safe_file_name(name) == expected


# ---------------------------------------------------------------------------
# write_report
# ---------------------------------------------------------------------------


def test_write_report_creates_all_files(tmp_path: Path, results: list[ExperimentResult]) -> None:
    leaderboard = write_report(results, tmp_path / "reports")

    out = tmp_path / "reports"
    assert leaderboard == out / "leaderboard.md"
    assert (out / "results.json").is_file()
    assert (out / "per_question" / "bm25.jsonl").is_file()
    assert (out / "per_question" / "hybrid.jsonl").is_file()
    assert not (out / "per_question" / "dense-e5.jsonl").exists()  # пропущенный эксперимент ответов не имеет


def test_results_json_content(tmp_path: Path, results: list[ExperimentResult]) -> None:
    write_report(results, tmp_path)
    payload = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))

    assert payload["report_split"] == "test"
    experiments = {e["name"]: e for e in payload["experiments"]}
    assert set(experiments) == {"bm25", "dense-e5", "hybrid"}
    assert experiments["bm25"]["metrics"]["test"]["recall@5"] == 0.8
    assert experiments["bm25"]["per_question_file"] == "per_question/bm25.jsonl"
    assert "per_question" not in experiments["bm25"]  # ответы — только в JSONL, без дублирования
    assert experiments["dense-e5"]["skipped"].startswith("нет пакета")


def test_per_question_jsonl_content(tmp_path: Path, results: list[ExperimentResult]) -> None:
    write_report(results, tmp_path)
    raw = (tmp_path / "per_question" / "bm25.jsonl").read_text(encoding="utf-8")
    rows = [json.loads(line) for line in raw.splitlines()]

    assert rows == results[0].per_question
    assert "календарных" in raw  # кириллица не экранирована — файл удобно читать глазами


def test_leaderboard_main_table(tmp_path: Path, results: list[ExperimentResult]) -> None:
    text = write_report(results, tmp_path).read_text(encoding="utf-8")
    rows = table_rows(text, "Эксперимент")
    header, separator, body = rows[0], rows[1], rows[2:]

    assert header[:10] == [
        "Эксперимент",
        "Recall@5",
        "MRR",
        "nDCG@5",
        "Hit@1",
        "FactRecall",
        "CitationPrecision",
        "Abstention F1",
        "p50, мс",
        "p95, мс",
    ]
    assert set("".join(separator)) <= set("-:|")  # строка-разделитель markdown-таблицы
    # Сортировка по Recall@5: hybrid (0.9) выше bm25 (0.8); пропущенного эксперимента в таблице нет.
    assert [row[0] for row in body] == ["hybrid", "bm25"]
    hybrid = dict(zip(header, body[0], strict=True))
    assert hybrid["Recall@5"] == "0.900"
    assert hybrid["MRR"] == "0.333"
    assert hybrid["p50, мс"] == "12.346"
    assert hybrid["Порог"] == "0.424"
    assert hybrid["N"] == "4"
    bm25 = dict(zip(header, body[1], strict=True))
    assert bm25["Abstention F1"] == "—"  # None выводится прочерком, а не «None» или 0


def test_leaderboard_by_type_table(tmp_path: Path, results: list[ExperimentResult]) -> None:
    text = write_report(results, tmp_path).read_text(encoding="utf-8")
    assert "## Recall@5 по типам вопросов" in text

    by_type_section = text.split("## Recall@5 по типам вопросов", 1)[1]
    rows = table_rows(by_type_section, "Эксперимент")
    # Типы — в каноническом порядке QUESTION_TYPES; unanswerable без recall не выводится.
    assert rows[0] == ["Эксперимент", "numeric", "keyword"]
    assert rows[2] == ["hybrid", "1.000", "0.667"]
    assert rows[3] == ["bm25", "1.000", "0.500"]


def test_leaderboard_lists_skipped_experiments_with_reason(
    tmp_path: Path, results: list[ExperimentResult]
) -> None:
    text = write_report(results, tmp_path).read_text(encoding="utf-8")
    skipped_section = text.split("## Пропущенные эксперименты", 1)[1]
    assert "- dense-e5: нет пакета sentence-transformers (pip install -e .[neural])" in skipped_section


def test_leaderboard_uses_requested_split(results: list[ExperimentResult]) -> None:
    text = render_leaderboard(results, split="dev")
    assert "сплите `dev`" in text
    assert "0.111" in text and "0.900" not in text


def test_leaderboard_without_completed_experiments() -> None:
    text = render_leaderboard([ExperimentResult.for_skipped("only", "нет GPU")])
    assert "Нет завершённых экспериментов." in text
    assert "- only: нет GPU" in text
    assert "по типам вопросов" not in text


def test_leaderboard_escapes_pipe_in_names() -> None:
    text = render_leaderboard([make_result("a|b", 0.5)])
    assert "| a\\|b |" in text


def test_report_serialization_is_deterministic(tmp_path: Path, results: list[ExperimentResult]) -> None:
    # Один и тот же результат → те же байты. Воспроизводимость двух прогонов проверяет
    # test_runner.py::test_two_runs_differ_only_in_timings.
    write_report(results, tmp_path / "first")
    write_report(results, tmp_path / "second")
    for relative in ("results.json", "leaderboard.md", "per_question/bm25.jsonl"):
        first = (tmp_path / "first" / relative).read_bytes()
        assert first == (tmp_path / "second" / relative).read_bytes()
        assert b"\r\n" not in first


def test_report_roundtrip_via_load_results(tmp_path: Path, results: list[ExperimentResult]) -> None:
    write_report(results, tmp_path)
    assert load_results(tmp_path) == results


def test_report_sanitizes_non_json_values(tmp_path: Path) -> None:
    result = make_result("weird", 0.5)
    result.threshold = math.inf
    result.per_question = [
        {"qid": "q", "confidence": np.float32(0.5), "latency_ms": math.nan, "path": Path("a/b")}
    ]

    write_report([result], tmp_path)

    payload = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert payload["experiments"][0]["threshold"] is None  # inf → null (стандартный JSON)
    row = json.loads((tmp_path / "per_question" / "weird.jsonl").read_text(encoding="utf-8"))
    assert row == {"qid": "q", "confidence": 0.5, "latency_ms": None, "path": "a/b"}


def test_report_rejects_names_that_collide_as_file_names(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="одинаковое имя файла"):
        write_report([make_result("exp:1", 0.5), make_result("exp/1", 0.6)], tmp_path)
