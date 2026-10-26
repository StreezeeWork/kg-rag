"""Тесты структуры результата (results.py) и отчёта (report.py) на синтетических экспериментах."""

from __future__ import annotations

import json
import math
import platform
import re
from pathlib import Path

import numpy as np
import pytest

from kbrag.eval import report
from kbrag.eval.report import (
    ENVIRONMENT_PACKAGES,
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
    metrics.update(
        {
            "recall@5": recall5,
            "n_questions": 4,
            "n_answerable": 3,
            "latency_p50_ms": 12.3456,
            "latency_p95_ms": 40.0,
        }
    )
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
            {
                "qid": "t-001",
                "type": "numeric",
                "split": "test",
                "answer": "28 календарных дней [1].",
                "recall@5": 1.0,
            },
            {
                "qid": "t-005",
                "type": "unanswerable",
                "split": "test",
                "answer": "В базе знаний нет ответа.",
                "recall@5": None,
            },
        ],
        threshold=0.4242,
        index_stats={"n_docs": 4, "n_chunks": 12},
        score_source="dense",
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


def test_results_json_records_environment(tmp_path: Path, results: list[ExperimentResult]) -> None:
    # Регрессия: по отчёту нельзя было узнать версии Python и пакетов, давших эти числа.
    write_report(results, tmp_path)
    environment = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))["environment"]
    assert environment["python"] == platform.python_version()
    assert environment["packages"]["numpy"] == np.__version__
    assert set(environment["packages"]) == set(ENVIRONMENT_PACKAGES)
    # Тесты запускаются из git-клона: коммит записан (40 hex-символов), флаг правок — bool.
    assert re.fullmatch(r"[0-9a-f]{40}", environment["git_sha"] or "")
    assert isinstance(environment["git_dirty"], bool)


def test_git_state_without_git_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(report.subprocess, "run", missing_git)
    assert report.git_state() == {"git_sha": None, "git_dirty": None}


def test_per_question_jsonl_content(tmp_path: Path, results: list[ExperimentResult]) -> None:
    write_report(results, tmp_path)
    raw = (tmp_path / "per_question" / "bm25.jsonl").read_text(encoding="utf-8")
    rows = [json.loads(line) for line in raw.splitlines()]

    assert rows == results[0].per_question
    assert "календарных" in raw  # кириллица не экранирована — файл удобно читать глазами


def section(markdown: str, title: str) -> str:
    """Текст раздела ``## title`` до следующего заголовка второго уровня."""
    return markdown.split(f"## {title}", 1)[1].split("\n## ", 1)[0]


def test_leaderboard_search_table(tmp_path: Path, results: list[ExperimentResult]) -> None:
    text = write_report(results, tmp_path).read_text(encoding="utf-8")
    rows = table_rows(section(text, "Поиск и задержка"), "Эксперимент")
    header, separator, body = rows[0], rows[1], rows[2:]

    assert header == [
        "Эксперимент",
        "Recall@5",
        "Recall@5, 95% ДИ",
        "MRR",
        "nDCG@5",
        "Hit@1",
        "p50, мс",
        "p95, мс",
        "N",
    ]
    assert set("".join(separator)) <= set("-:|")  # строка-разделитель markdown-таблицы
    # Сортировка по Recall@5: hybrid (0.9) выше bm25 (0.8); пропущенного эксперимента в таблице нет.
    assert [row[0] for row in body] == ["hybrid", "bm25"]
    hybrid = dict(zip(header, body[0], strict=True))
    assert hybrid["Recall@5"] == "0.900"
    assert hybrid["Recall@5, 95% ДИ"] == "1.000–1.000"  # из per_question: один отвечаемый вопрос
    assert hybrid["MRR"] == "0.333"
    assert hybrid["p50, мс"] == "12.346"
    assert hybrid["N"] == "3 + 1"  # отвечаемые + неотвечаемые: N для поиска — первое число


def _with_answers(name: str, recalls: list[float]) -> ExperimentResult:
    """Эксперимент, у которого в per_question заданы Recall@5 и MRR по вопросам q-0, q-1, …"""
    result = make_result(name, sum(recalls) / len(recalls))
    result.per_question = [
        {"qid": f"q-{i}", "split": "test", "recall@5": value, "mrr": value} for i, value in enumerate(recalls)
    ]
    return result


def test_leader_diff_marks_noise_with_approx_sign() -> None:
    # Регрессия: строки сортировались по точечному Recall@5 без интервалов, и «победа»
    # в один вопрос из 124 выглядела как вывод.
    leader = _with_answers("leader", [1.0] * 107 + [0.0] * 17)
    one_question_behind = _with_answers("close", [1.0] * 106 + [0.0] * 18)
    far_behind = _with_answers("far", [1.0] * 80 + [0.0] * 44)
    text = render_leaderboard([far_behind, one_question_behind, leader])

    rows = table_rows(section(text, "Отличие от лидера (leader)"), "Эксперимент")
    assert rows[0] == ["Эксперимент", "ΔRecall@5 [95% ДИ]", "ΔMRR [95% ДИ]"]
    cells = {row[0]: row[1] for row in rows[2:]}
    assert cells["close"].startswith("-0.008 [") and cells["close"].endswith("≈")
    assert cells["far"].startswith("-0.218 [") and not cells["far"].endswith("≈")
    search = dict(zip(table_rows(text, "Эксперимент")[0], table_rows(text, "Эксперимент")[2], strict=True))
    low, high = (float(x) for x in search["Recall@5, 95% ДИ"].split("–"))
    assert low < 107 / 124 < high


def test_leader_diff_is_skipped_without_per_question(results: list[ExperimentResult]) -> None:
    for result in results:
        result.per_question = []  # например, results.json без JSONL-файлов
    text = render_leaderboard(results)
    assert "## Отличие от лидера" not in text
    assert table_rows(section(text, "Поиск и задержка"), "Эксперимент")[2][2] == "—"


def test_leaderboard_answer_table_shows_threshold_effects(
    tmp_path: Path, results: list[ExperimentResult]
) -> None:
    # Регрессия: FactRecall стоял без FAR и без FactRecall по отвеченным, а порог — без шкалы,
    # поэтому разница в пороге выглядела как разница в качестве генерации.
    text = write_report(results, tmp_path).read_text(encoding="utf-8")
    rows = table_rows(section(text, "Ответы и отказы"), "Эксперимент")
    header, body = rows[0], rows[2:]

    assert header == [
        "Эксперимент",
        "FactRecall (отвеч.)",
        "FactRecall",
        "CitationPrecision",
        "Архивные ссылки",
        "Abstention F1",
        "FAR",
        "Порог",
    ]
    assert [row[0] for row in body] == ["hybrid", "bm25"]  # тот же порядок строк, что в таблице поиска
    hybrid = dict(zip(header, body[0], strict=True))
    assert hybrid["Порог"] == "dense: 0.424"  # шкала уверенности рядом с числом
    assert hybrid["FAR"] == "0.500"
    bm25 = dict(zip(header, body[1], strict=True))
    assert bm25["Abstention F1"] == "—"  # None выводится прочерком, а не «None» или 0
    assert "отказ на отвечаемый вопрос считается за 0" in text  # легенда объясняет FactRecall


def test_leaderboard_intro_and_threshold_note_show_sample_sizes(results: list[ExperimentResult]) -> None:
    text = render_leaderboard(results)
    assert "4 вопросов, из них 3 отвечаемых" in text
    assert "Один отвечаемый вопрос ≈ 0.333 Recall@5" in text
    assert "Порог подбирается на dev: 3 отвечаемых и 1 неотвечаемых" in text


def test_leaderboard_threshold_without_score_source_is_plain_number() -> None:
    result = make_result("old", 0.5)
    result.score_source = None  # отчёт, записанный до появления поля
    rows = table_rows(section(render_leaderboard([result]), "Ответы и отказы"), "Эксперимент")
    assert rows[2][-1] == "0.424"


def test_leaderboard_auroc_section_marks_threshold_source(results: list[ExperimentResult]) -> None:
    results[0].confidence_auroc = {"test": {"bm25": {"auroc": 0.6, "ci95": [0.5, 0.7]}}}
    results[2].confidence_auroc = {
        "test": {"dense": {"auroc": 0.75, "ci95": [0.6, 0.88]}, "bm25": {"auroc": 0.7, "ci95": None}},
        "dev": {"dense": {"auroc": 0.1, "ci95": [0.0, 0.2]}},
    }
    markdown = render_leaderboard(results, "test")

    assert "## Отделяет ли уверенность поиска неотвечаемые вопросы (AUROC)" in markdown
    rows = table_rows(markdown, "Эксперимент | dense")
    assert rows[0] == ["Эксперимент", "dense", "bm25"]
    # score_source у обоих — dense: жирным выделен именно он; у bm25-строки dense нет вовсе.
    assert rows[2] == ["hybrid", "**0.750 [0.600; 0.880]**", "0.700"]
    assert rows[3] == ["bm25", "—", "0.600 [0.500; 0.700]"]
    assert "0.100" not in markdown  # dev в таблицу test не попадает


def test_leaderboard_without_auroc_has_no_auroc_section(results: list[ExperimentResult]) -> None:
    assert "AUROC" not in render_leaderboard(results, "test").split("## Как читать таблицы")[0]


def test_leaderboard_confusable_section(results: list[ExperimentResult]) -> None:
    assert "## Тип confusable" not in render_leaderboard(results)  # нет данных — нет раздела

    confusable = {"recall@5": 1.0, "fact_recall_answered": 1.0, "stale_citation_rate": 0.5}
    text = render_leaderboard([make_result("hybrid", 0.9, by_type={"confusable": confusable})])
    rows = table_rows(section(text, "Тип confusable"), "Эксперимент")
    assert rows[0] == ["Эксперимент", "Recall@5", "FactRecall (отвеч.)", "Архивные ссылки"]
    assert rows[2] == ["hybrid", "1.000", "1.000", "0.500"]


def test_leaderboard_by_type_table(tmp_path: Path, results: list[ExperimentResult]) -> None:
    text = write_report(results, tmp_path).read_text(encoding="utf-8")
    assert "## Recall@5 по типам вопросов" in text

    by_type_section = text.split("## Recall@5 по типам вопросов", 1)[1]
    rows = table_rows(by_type_section, "Эксперимент")
    # Типы — в каноническом порядке QUESTION_TYPES; unanswerable без recall не выводится.
    assert rows[0] == ["Эксперимент", "numeric", "keyword"]
    assert rows[2] == ["hybrid", "1.000", "0.667"]
    assert rows[3] == ["bm25", "1.000", "0.500"]


def test_leaderboard_by_type_shows_group_sizes() -> None:
    by_type = {"keyword": {"recall@5": 10 / 11, "n_answerable": 11, "n_questions": 11}}
    text = render_leaderboard(
        [make_result("a", 0.9, by_type=by_type), make_result("b", 0.8, by_type=by_type)]
    )
    by_type_section = section(text, "Recall@5 по типам вопросов")
    assert table_rows(by_type_section, "Эксперимент")[0] == ["Эксперимент", "keyword (n=11)"]
    assert "при n = 11 это ≈ 0.09 Recall@5" in by_type_section


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
