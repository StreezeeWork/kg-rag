"""Тесты обучения классификатора отказа: CV, выбор модели, порог, защита от утечки, CLI.

Логика CV и порога проверяется на синтетике (разделимые данные → высокий AUROC, случайные
метки → около 0.5). Сквозное обучение и команды CLI — на мини-корпусе из conftest
с обучающим набором из ``abstention_helpers`` (BM25 + extractive, офлайн, секунды).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from abstention_helpers import DEV_UNANSWERABLE, train_records, write_jsonl

from kbrag.abstention import FEATURE_NAMES, load_abstention_model, models
from kbrag.abstention.train import (
    TrainReport,
    baseline_scores,
    choose_candidate,
    cross_validate,
    fit_abstention,
    train_abstention,
    tune_decision_threshold,
)
from kbrag.cli import EXIT_ERROR, EXIT_OK, _print_train_report, main
from kbrag.config import AppConfig, deep_merge, load_config
from kbrag.eval.dataset import find_near_duplicates, item_from_record, load_gold, train_gold_conflicts
from kbrag.ingest import corpus_fingerprint, load_corpus
from kbrag.pipeline import RAGPipeline
from kbrag.types import Document


def synthetic_rows(n: int = 160, seed: int = 0, signal: float = 2.0) -> tuple[list[dict], np.ndarray]:
    """Строки признаков: у класса 1 («нет ответа») ниже dense_top1 и q_cov_top1 (signal=0 — шум)."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    matrix = rng.normal(size=(n, len(FEATURE_NAMES)))
    matrix[:, FEATURE_NAMES.index("dense_top1")] -= signal * y
    matrix[:, FEATURE_NAMES.index("q_cov_top1")] -= signal * y
    return [dict(zip(FEATURE_NAMES, row, strict=True)) for row in matrix], y


# ---------------------------------------------------------------------------
# Кросс-валидация и выбор
# ---------------------------------------------------------------------------


def test_cv_finds_signal_and_reports_baseline() -> None:
    rows, y = synthetic_rows()
    results = cross_validate(rows, y, model="logreg", baseline_feature="dense_top1")

    by_name = {row["name"]: row for row in results}
    assert list(by_name) == ["baseline: dense_top1", "logreg C=0.1", "logreg C=1", "logreg C=10"]
    assert all(row["n_folds"] == 15 for row in results)  # 5 фолдов × 3 повтора
    assert by_name["logreg C=1"]["roc_auc_mean"] > 0.9
    assert by_name["baseline: dense_top1"]["roc_auc_mean"] > 0.8  # один признак тоже несёт сигнал
    for row in results:
        assert 0.0 <= row["pr_auc_mean"] <= 1.0 and row["roc_auc_std"] >= 0.0


def test_cv_on_random_labels_is_near_chance() -> None:
    rows, y = synthetic_rows(signal=0.0)
    results = cross_validate(rows, y, model="logreg")
    for row in results:
        assert abs(row["roc_auc_mean"] - 0.5) < 0.12


def test_cv_with_catboost() -> None:
    pytest.importorskip("catboost")
    rows, y = synthetic_rows()
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(models.CATBOOST_DEFAULTS, "iterations", 30)  # тест про логику CV, а не про качество
        results = cross_validate(rows, y, model="catboost")
    assert [row["name"] for row in results] == ["catboost depth=2", "catboost depth=3", "catboost depth=4"]
    assert all(row["roc_auc_mean"] > 0.85 for row in results)


def test_cv_is_deterministic() -> None:
    rows, y = synthetic_rows()
    assert cross_validate(rows, y, model="logreg") == cross_validate(rows, y, model="logreg")


def test_cv_needs_both_classes() -> None:
    rows, _ = synthetic_rows(n=20)
    with pytest.raises(ValueError, match="по 2 вопроса"):
        cross_validate(rows, np.zeros(20, dtype=int), model="logreg")


def test_choice_by_mean_roc_auc_and_ties_go_to_simpler_model() -> None:
    def row(name: str, model_type: str, auc: float, complexity: int) -> dict:
        return {
            "name": name,
            "model_type": model_type,
            "roc_auc_mean": auc,
            "complexity": complexity,
            "params": {},
        }

    cv = [
        row("baseline", "baseline", 0.99, 0),  # бейзлайн не выбирается, даже если он лучше
        row("logreg C=0.1", "logreg", 0.80, 1),
        row("logreg C=1", "logreg", 0.85, 2),
        row("catboost depth=2", "catboost", 0.85, 3),  # ничья с более простой моделью
    ]
    assert choose_candidate(cv)["name"] == "logreg C=1"
    cv.append(row("catboost depth=3", "catboost", 0.86, 4))
    assert choose_candidate(cv)["name"] == "catboost depth=3"


# ---------------------------------------------------------------------------
# Порог решения и бейзлайн
# ---------------------------------------------------------------------------


def test_decision_threshold_direction() -> None:
    # p — вероятность «нет ответа»; y = 1 — действительно нет ответа.
    threshold = tune_decision_threshold([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
    assert threshold == pytest.approx(0.5)
    abstained = [p >= threshold for p in (0.1, 0.2, 0.8, 0.9)]
    assert abstained == [False, False, True, True]


def test_no_threshold_when_classifier_is_useless_on_dev() -> None:
    assert tune_decision_threshold([0.9, 0.8, 0.1, 0.2], [0, 0, 1, 1]) is None  # перевёрнутый сигнал
    assert tune_decision_threshold([0.1, 0.9], [0, 0]) is None  # на dev один класс


def test_baseline_score_is_negated_confidence_with_missing_on_top() -> None:
    rows = [{"dense_top1": 0.9}, {"dense_top1": 0.2}, {"dense_top1": float("nan")}]
    scores = baseline_scores(rows, "dense_top1")
    assert scores[0] < scores[1] < scores[2]  # нет оценки поиска — «вероятнее всего нет ответа»
    assert np.isfinite(scores).all()


# ---------------------------------------------------------------------------
# Защита от утечки
# ---------------------------------------------------------------------------


def items(records: list[dict]) -> list:
    return [item_from_record(record) for record in records]


def test_near_duplicates_by_jaccard_of_stems(mini_gold_records: list[dict]) -> None:
    gold = items(mini_gold_records)
    near = items(
        [
            {
                **train_records()[0],
                "id": "x-1",
                "question": "Сколько календарных дней длится ежегодный отпуск??",
            },
            {
                **train_records()[0],
                "id": "x-2",
                "question": "Сколько календарных дней длится отпуск ежегодный?",
            },
            {
                **train_records()[0],
                "id": "x-3",
                "question": "Сколько календарных дней длится ежегодный отпуск у стажёра?",
            },
        ]
    )
    pairs = find_near_duplicates(near, gold)
    assert [(a, b) for a, b, _ in pairs] == [("x-1", "t-001"), ("x-2", "t-001"), ("x-3", "t-001")]
    assert pairs[0][2] == 1.0 and 0.7 <= pairs[2][2] < 1.0  # одно лишнее слово: 6/7
    assert find_near_duplicates(items(train_records()), gold) == []  # синтетический train чистый


def test_train_gold_conflicts_report_shared_ids(mini_gold_records: list[dict]) -> None:
    train = items([{**train_records()[0], "id": "t-001"}])
    errors = train_gold_conflicts(train, items(mini_gold_records))
    assert len(errors) == 1 and "t-001" in errors[0] and "эталоне" in errors[0]


def test_fit_refuses_test_questions(
    tmp_path: Path, mini_docs: list[Document], mini_gold_records: list[dict]
) -> None:
    cfg = AppConfig.model_validate(
        {"retrieval": {"mode": "bm25", "top_k": 3}, "abstention": {"threshold": None}}
    )
    pipeline = RAGPipeline.from_config(cfg, docs=mini_docs)
    gold = items(mini_gold_records)
    test = [item for item in gold if item.split == "test"]
    with pytest.raises(ValueError, match="split='dev'"):
        fit_abstention(pipeline, items(train_records()), test, model="logreg")
    with pytest.raises(ValueError, match="split='train'"):
        fit_abstention(pipeline, gold, [item for item in gold if item.split == "dev"], model="logreg")


# ---------------------------------------------------------------------------
# Сквозное обучение на мини-корпусе
# ---------------------------------------------------------------------------


def make_cfg(tmp_path: Path) -> AppConfig:
    return AppConfig.model_validate(
        {
            "retrieval": {"mode": "bm25", "top_k": 3},
            "abstention": {"threshold": None, "model_path": (tmp_path / "abstention.json").as_posix()},
        }
    )


def test_train_abstention_saves_a_model_the_service_accepts(
    tmp_path: Path, mini_docs: list[Document], mini_gold_records: list[dict]
) -> None:
    cfg = make_cfg(tmp_path)
    gold = items([*mini_gold_records, DEV_UNANSWERABLE])
    report = train_abstention(cfg, model="logreg", docs=mini_docs, train=items(train_records()), gold=gold)

    assert report.model_path is not None and report.decision_threshold is not None
    assert report.n_train == 22 and report.n_train_unanswerable == 10
    assert report.n_dev == 3 and report.n_dev_unanswerable == 1  # только dev, test не участвует
    assert report.importance_kind == "coef" and len(report.importance) == len(FEATURE_NAMES)
    assert {"classifier", "baseline", "auroc_diff"} == set(report.dev)

    model = load_abstention_model(cfg, corpus_sha256=corpus_fingerprint(mini_docs))
    assert model is not None and model.decision_threshold == pytest.approx(report.decision_threshold)
    data = json.loads(Path(cfg.abstention.model_path).read_text(encoding="utf-8"))
    for key in (
        "fingerprint",
        "corpus_sha256",
        "train_sha256",
        "cv_report",
        "dev_metrics",
        "created_at",
        "versions",
    ):
        assert key in data
    assert data["cv_report"]["chosen"] == report.chosen
    # Рекомендация «переключаться ли» — по dev; флаг train-CV подписан как справочный.
    assert data["dev_metrics"]["beats_baseline_dev"] == report.beats_baseline_dev
    assert "beats_baseline_cv" in data["cv_report"] and "auroc_diff" in data["dev_metrics"]


def test_train_abstention_rejects_train_that_overlaps_gold(
    tmp_path: Path, mini_docs: list[Document], mini_gold_records: list[dict]
) -> None:
    gold = items([*mini_gold_records, DEV_UNANSWERABLE])
    leaked = items([*train_records(), {**mini_gold_records[0], "id": "tr-999", "split": "train"}])
    with pytest.raises(ValueError, match="tr-999 ≈ t-001"):
        train_abstention(make_cfg(tmp_path), model="logreg", docs=mini_docs, train=leaked, gold=gold)
    assert not (tmp_path / "abstention.json").exists()


# ---------------------------------------------------------------------------
# CLI: build-train и train-abstention
# ---------------------------------------------------------------------------


@pytest.fixture()
def cli_config(tmp_path: Path, mini_corpus_dir: Path, mini_gold_records: list[dict]) -> Path:
    gold = write_jsonl(tmp_path / "gold.jsonl", [*mini_gold_records, DEV_UNANSWERABLE])
    config = {
        "corpus_dir": str(mini_corpus_dir),
        "gold_path": str(gold),
        "train_path": str(tmp_path / "train.jsonl"),
        "cache_dir": str(tmp_path / "cache"),
        "retrieval": {"mode": "bm25", "top_k": 3},
        "abstention": {"threshold": None, "model_path": str(tmp_path / "abstention.json")},
    }
    path = tmp_path / "mini.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return path


@pytest.fixture()
def train_parts(tmp_path: Path) -> Path:
    """Части обучающего набора, как в data/eval/train_parts: без поля split, в двух файлах."""
    records = train_records(split=None)
    write_jsonl(tmp_path / "train_parts" / "a.jsonl", records[:12])
    write_jsonl(tmp_path / "train_parts" / "b.jsonl", records[12:])
    return tmp_path / "train_parts"


def build_train_args(config: Path, parts: Path, out: Path, *extra: str) -> list[str]:
    return ["build-train", "--config", str(config), "--parts", str(parts), "--out", str(out), *extra]


def test_build_train_merges_parts_marks_split_and_checks(
    tmp_path: Path, cli_config: Path, train_parts: Path
) -> None:
    out = tmp_path / "train.jsonl"
    assert main(build_train_args(cli_config, train_parts, out)) == EXIT_OK
    built = load_gold(out)
    assert len(built) == 22 and {item.split for item in built} == {"train"}
    assert main(build_train_args(cli_config, train_parts, out, "--check")) == EXIT_OK

    out.write_text("".join(out.read_text(encoding="utf-8").splitlines(keepends=True)[:-1]), encoding="utf-8")
    assert main(build_train_args(cli_config, train_parts, out, "--check")) == EXIT_ERROR


def test_build_train_catches_near_duplicates_of_gold(
    tmp_path: Path, cli_config: Path, train_parts: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_jsonl(
        train_parts / "c_leak.jsonl",
        [
            {
                **train_records(split=None)[0],
                "id": "tr-900",
                "question": "Сколько календарных дней длится ежегодный отпуск",
            }
        ],
    )
    out = tmp_path / "train.jsonl"
    assert main(build_train_args(cli_config, train_parts, out)) == EXIT_ERROR
    assert not out.exists()
    assert "tr-900 ≈ t-001" in capsys.readouterr().err


def test_train_abstention_cli_smoke(
    tmp_path: Path, cli_config: Path, train_parts: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(build_train_args(cli_config, train_parts, tmp_path / "train.jsonl")) == EXIT_OK
    capsys.readouterr()

    assert main(["train-abstention", "--config", str(cli_config), "--model", "logreg"]) == EXIT_OK
    out = capsys.readouterr().out
    for fragment in (
        "Кросс-валидация",
        "baseline: bm25_top1",
        "Выбрана модель: logreg",
        "AUROC",
        "Модель сохранена",
    ):
        assert fragment in out
    cfg = load_config(cli_config)
    assert (
        load_abstention_model(cfg, corpus_sha256=corpus_fingerprint(load_corpus(cfg.corpus_dir))) is not None
    )


def test_train_abstention_cli_hints_to_build_train(
    tmp_path: Path, cli_config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["train-abstention", "--config", str(cli_config)]) == EXIT_ERROR
    assert "kbrag build-train" in capsys.readouterr().err


def test_train_abstention_cli_needs_model_path(
    tmp_path: Path, cli_config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = yaml.safe_load(cli_config.read_text(encoding="utf-8"))
    no_path = tmp_path / "no_path.yaml"
    no_path.write_text(
        yaml.safe_dump(deep_merge(data, {"abstention": {"model_path": None}}), allow_unicode=True),
        encoding="utf-8",
    )
    assert main(["train-abstention", "--config", str(no_path)]) == EXIT_ERROR
    assert "model_path" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Рекомендация «оставить порог»: решает dev, а не train-CV
# ---------------------------------------------------------------------------


def report_with(cv_baseline: float, dev_diff_ci: tuple[float, float]) -> TrainReport:
    """Отчёт обучения «руками»: CV-строки бейзлайна и модели, на dev — только разница AUROC с ДИ."""

    def cv_row(name: str, auc: float) -> dict:
        return {
            "name": name,
            "roc_auc_mean": auc,
            "roc_auc_std": 0.07,
            "pr_auc_mean": 0.5,
            "pr_auc_std": 0.1,
            "n_folds": 15,
        }

    low, high = dev_diff_ci
    side = {"auroc": 0.7, "auroc_ci95": [0.6, 0.8], "threshold": 0.4}
    return TrainReport(
        model_type="logreg",
        params={"C": 1.0},
        chosen="logreg C=1",
        baseline_feature="dense_top1",
        cv=[cv_row("baseline: dense_top1", cv_baseline), cv_row("logreg C=1", 0.68)],
        decision_threshold=0.4,
        dev={
            "classifier": side,
            "baseline": side,
            "auroc_diff": {"value": (low + high) / 2, "ci95": [low, high]},
        },
        importance_kind="coef",
        importance=[("q_len", -0.8)],
        n_train=300,
        n_train_unanswerable=120,
        n_dev=71,
        n_dev_unanswerable=15,
    )


def test_weak_baseline_on_train_does_not_hide_the_keep_threshold_warning(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Регрессия: на train бейзлайн почти случаен (сдвиг распределения), модель его обгоняет на CV,
    # а на dev — нет. Раньше флаг считался по CV, и предупреждение молчало ровно тогда, когда нужно.
    report = report_with(cv_baseline=0.53, dev_diff_ci=(-0.233, 0.133))
    assert report.beats_baseline_cv and not report.beats_baseline_dev
    _print_train_report(report)
    out = capsys.readouterr().out
    assert "оставить abstention.method: threshold" in out
    assert "сдвига распределения" in out
    assert "-0.233" in out and "+0.133" in out


def test_proven_dev_advantage_gives_no_warning(capsys: pytest.CaptureFixture[str]) -> None:
    report = report_with(cv_baseline=0.80, dev_diff_ci=(0.02, 0.20))
    assert report.beats_baseline_dev and not report.beats_baseline_cv  # решает dev, а не CV
    _print_train_report(report)
    assert "оставить abstention.method: threshold" not in capsys.readouterr().out
    assert report.to_dict()["beats_baseline_dev"] is True
