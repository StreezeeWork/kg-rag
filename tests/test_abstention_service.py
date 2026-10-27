"""Классификатор отказа внутри системы: пайплайн сервиса, API, оценка (runner) и сетка.

Модель здесь не обучается, а задаётся руками — так её поведение очевидно: p = σ(5 − 10 · q_cov_topk),
то есть «если в выдаче нет слов вопроса — отказ». Всё офлайн, на мини-корпусе из conftest.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pytest
import yaml
from abstention_helpers import DEV_UNANSWERABLE
from fastapi.testclient import TestClient

from kbrag.abstention import FEATURE_NAMES, AbstentionModel, LogRegParams, extract_features
from kbrag.api.app import build_ask_response, create_app
from kbrag.cli import EXIT_ERROR, main
from kbrag.config import AppConfig, deep_merge
from kbrag.eval import runner
from kbrag.eval.dataset import GoldItem, item_from_record
from kbrag.eval.report import write_report
from kbrag.eval.runner import run_experiment, run_grid
from kbrag.generation import ABSTAIN_TEXT, GenerationResult, Generator
from kbrag.ingest import corpus_fingerprint
from kbrag.pipeline import ABSTAIN_CLASSIFIER_FLAG, LOW_CONFIDENCE_FLAG, RAGPipeline, calibration_fingerprint
from kbrag.types import Document, ScoredChunk

VACATION_Q = "Сколько календарных дней длится ежегодный отпуск?"
REMOTE_Q = "Сколько дней в неделю можно работать удалённо?"  # в выдаче — фрагмент с инъекцией
UNKNOWN_Q = "Сколько стоит обед в столовой казанского офиса?"  # ни одного слова из мини-корпуса


class SpyGenerator(Generator):
    name = "spy"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(self, question: str, contexts: list[ScoredChunk]) -> GenerationResult:
        self.calls.append(question)
        return GenerationResult(text="Ответ [1].", abstained=False)


def hand_model(
    intercept: float = 5.0, coverage_weight: float = -10.0, threshold: float = 0.5
) -> AbstentionModel:
    """logreg «руками»: без медиан и масштабирования, вес только у q_cov_topk."""
    n = len(FEATURE_NAMES)
    coef = np.zeros(n)
    coef[FEATURE_NAMES.index("q_cov_topk")] = coverage_weight
    params = LogRegParams(np.zeros(n), np.zeros(n), np.ones(n), coef, intercept)
    return AbstentionModel("logreg", threshold, logreg=params)


def make_cfg(tmp_path: Path, **sections) -> AppConfig:
    base = {
        "retrieval": {"mode": "bm25", "top_k": 3, "dense": {"lsa_dim": 16}},
        "abstention": {
            "method": "classifier",
            "threshold": None,
            "calibration_path": (tmp_path / "calibration.json").as_posix(),
            "model_path": (tmp_path / "abstention.json").as_posix(),
        },
    }
    return AppConfig.model_validate(deep_merge(base, sections))


def save_for(cfg: AppConfig, docs: list[Document], model: AbstentionModel | None = None) -> AbstentionModel:
    """Сохраняет модель с «паспортом» под этот конфиг и корпус — так, как это делает train-abstention."""
    model = model or hand_model()
    model.metadata.update(
        {"fingerprint": calibration_fingerprint(cfg), "corpus_sha256": corpus_fingerprint(docs)}
    )
    model.save(cfg.abstention.model_path)
    return model


def pipeline_with(
    cfg: AppConfig,
    docs: list[Document],
    model: AbstentionModel | None,
    generator: Generator | None = None,
    threshold: float | None = None,
) -> RAGPipeline:
    """Пайплайн с готовой моделью отказа (без файла): индекс берём у обычной сборки по конфигу."""
    threshold_cfg = cfg.abstention.model_copy(update={"method": "threshold"})
    indexed = RAGPipeline.from_config(cfg.model_copy(update={"abstention": threshold_cfg}), docs=docs)
    return RAGPipeline(
        cfg,
        indexed.retriever,
        generator or SpyGenerator(),
        threshold=threshold,
        chunks=indexed.chunks,
        abstention_model=model,
    )


def same_features(a: dict[str, float], b: dict[str, float]) -> bool:
    return all(a[k] == b[k] or (math.isnan(a[k]) and math.isnan(b[k])) for k in FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Паритет: сервис и оценка видят одни и те же признаки
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["bm25", "hybrid"])
def test_service_and_eval_compute_identical_features(
    tmp_path: Path, mini_docs: list[Document], mode: str
) -> None:
    cfg = make_cfg(tmp_path, retrieval={"mode": mode})
    pipeline = pipeline_with(cfg, mini_docs, hand_model())
    top_k = cfg.retrieval.top_k
    for question in (VACATION_Q, REMOTE_Q, UNKNOWN_Q):
        service_contexts, _ = pipeline.retrieve(question, top_k=top_k)  # как в RAGPipeline.answer
        eval_contexts = pipeline.retrieve(question, top_k=10)[0][:top_k]  # как в runner: выдача для Recall@10
        assert same_features(
            extract_features(question, service_contexts), extract_features(question, eval_contexts)
        )

        # Запрос с другим top_k не меняет p: решение всегда по первым cfg.retrieval.top_k фрагментам.
        expected = pipeline.abstention_model.predict_one(extract_features(question, service_contexts))
        assert pipeline.answer(question, top_k=1).abstain_score == pytest.approx(expected)
        assert pipeline.answer(question).abstain_score == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Пайплайн сервиса
# ---------------------------------------------------------------------------


def test_classifier_abstains_before_generation(tmp_path: Path, mini_docs: list[Document]) -> None:
    spy = SpyGenerator()
    pipeline = pipeline_with(make_cfg(tmp_path), mini_docs, hand_model(), generator=spy)

    answer = pipeline.answer(UNKNOWN_Q)
    assert answer.abstained and answer.text == ABSTAIN_TEXT and answer.citations == []
    assert ABSTAIN_CLASSIFIER_FLAG in answer.flags and LOW_CONFIDENCE_FLAG not in answer.flags
    assert answer.abstain_score == pytest.approx(
        1 / (1 + math.exp(-5))
    )  # в выдаче нет ни одного слова вопроса
    assert spy.calls == []

    answer = pipeline.answer(VACATION_Q)
    assert not answer.abstained and spy.calls == [VACATION_Q]
    assert answer.abstain_score < 0.5 and ABSTAIN_CLASSIFIER_FLAG not in answer.flags


def test_classifier_replaces_the_threshold(tmp_path: Path, mini_docs: list[Document]) -> None:
    # Огромный порог по уверенности отказал бы везде, но решает модель.
    pipeline = pipeline_with(make_cfg(tmp_path), mini_docs, hand_model(), threshold=1e9)
    assert not pipeline.answer(VACATION_Q).abstained
    assert pipeline.abstention_method == "classifier:logreg"


def test_guardrails_flags_survive_a_classifier_abstention(tmp_path: Path, mini_docs: list[Document]) -> None:
    always = hand_model(intercept=10.0, coverage_weight=0.0)  # p ≈ 1 на любом вопросе
    answer = pipeline_with(make_cfg(tmp_path), mini_docs, always).answer(REMOTE_Q)
    assert answer.abstained and ABSTAIN_CLASSIFIER_FLAG in answer.flags
    assert any(flag.startswith("prompt_injection:") for flag in answer.flags)


def test_no_abstention_when_disabled_or_not_applied(tmp_path: Path, mini_docs: list[Document]) -> None:
    disabled = make_cfg(tmp_path, abstention={"enabled": False})
    assert not pipeline_with(disabled, mini_docs, hand_model()).answer(UNKNOWN_Q).abstained

    answer = pipeline_with(make_cfg(tmp_path), mini_docs, hand_model()).answer(
        UNKNOWN_Q, apply_threshold=False
    )
    assert not answer.abstained and answer.abstain_score > 0.5  # оценка применит порог модели сама


def test_from_config_loads_the_model(tmp_path: Path, mini_docs: list[Document]) -> None:
    cfg = make_cfg(tmp_path, abstention={"threshold": 0.5})
    save_for(cfg, mini_docs)
    pipeline = RAGPipeline.from_config(cfg, docs=mini_docs)
    assert pipeline.abstention_method == "classifier:logreg"
    assert pipeline.threshold is None  # порог при рабочем классификаторе не используется
    assert ABSTAIN_CLASSIFIER_FLAG in pipeline.answer(UNKNOWN_Q).flags


def test_from_config_falls_back_to_threshold_without_model(
    tmp_path: Path, mini_docs: list[Document], caplog: pytest.LogCaptureFixture
) -> None:
    cfg = make_cfg(tmp_path, abstention={"threshold": 0.5})
    with caplog.at_level(logging.WARNING):
        pipeline = RAGPipeline.from_config(cfg, docs=mini_docs)
    assert pipeline.abstention_model is None and pipeline.abstention_method == "threshold"
    assert pipeline.threshold == 0.5
    assert "откат на пороговый метод" in caplog.text
    answer = pipeline.answer(UNKNOWN_Q)
    assert LOW_CONFIDENCE_FLAG in answer.flags and answer.abstain_score is None


def test_api_reports_abstain_score_and_method(tmp_path: Path, mini_docs: list[Document]) -> None:
    cfg = make_cfg(tmp_path)
    save_for(cfg, mini_docs)
    pipeline = RAGPipeline.from_config(cfg, docs=mini_docs)
    response = build_ask_response(pipeline.answer(UNKNOWN_Q), pipeline.score_source)
    assert response.abstain_score == pytest.approx(1 / (1 + math.exp(-5)))
    assert ABSTAIN_CLASSIFIER_FLAG in response.flags

    with TestClient(create_app(cfg=cfg, pipeline=pipeline)) as client:
        assert client.get("/health").json()["abstention_method"] == "classifier:logreg"
        assert client.post("/ask", json={"question": VACATION_Q}).json()["abstain_score"] < 0.5


# ---------------------------------------------------------------------------
# Оценка (runner) и сетка
# ---------------------------------------------------------------------------


@pytest.fixture()
def gold(mini_gold_records: list[dict]) -> list[GoldItem]:
    return [item_from_record(record) for record in [*mini_gold_records, DEV_UNANSWERABLE]]


def test_runner_uses_the_model_threshold_without_retuning(
    tmp_path: Path, mini_docs: list[Document], gold: list[GoldItem], monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = make_cfg(tmp_path, abstention={"threshold": "auto"})
    model = save_for(cfg, mini_docs, hand_model(threshold=0.42))

    def no_tuning(*args, **kwargs):
        raise AssertionError("порог классификатора в оценке не переподбирается")

    monkeypatch.setattr(runner, "choose_threshold", no_tuning)
    result = run_experiment(cfg, name="clf", gold=gold, docs=mini_docs)

    assert result.threshold == model.decision_threshold == 0.42
    assert result.score_source == "classifier:logreg"
    rows = {row["qid"]: row for row in result.per_question}
    unknown = rows["t-005"]
    assert unknown["abstained"] and ABSTAIN_CLASSIFIER_FLAG in unknown["flags"]
    assert unknown["confidence_by_source"]["classifier"] == pytest.approx(1 - unknown["abstain_score"])
    assert not rows["t-001"]["abstained"] and rows["t-001"]["abstain_score"] < 0.42
    # AUROC классификатора попадает в ту же таблицу, что и источники уверенности поиска.
    assert "classifier" in result.confidence_auroc["test"]
    leaderboard = write_report([result], tmp_path / "report").read_text(encoding="utf-8")
    assert "classifier" in leaderboard


def test_runner_requires_a_model_for_classifier_method(
    tmp_path: Path, mini_docs: list[Document], gold: list[GoldItem]
) -> None:
    with pytest.raises(ValueError, match="kbrag train-abstention"):
        run_experiment(make_cfg(tmp_path), name="clf", gold=gold, docs=mini_docs)


def test_grid_skips_classifier_experiment_without_model(
    tmp_path: Path, mini_corpus_dir: Path, mini_gold_path: Path
) -> None:
    base = {
        "corpus_dir": str(mini_corpus_dir),
        "gold_path": str(mini_gold_path),
        "retrieval": {"mode": "bm25", "top_k": 3},
        "abstention": {"threshold": None},
    }
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base, allow_unicode=True), encoding="utf-8")
    grid = {
        "base": str(tmp_path / "base.yaml"),
        "experiments": [
            {"name": "bm25"},
            {
                "name": "bm25_clf",
                "overrides": {
                    "abstention": {"method": "classifier", "model_path": str(tmp_path / "none.json")}
                },
            },
        ],
    }
    path = tmp_path / "experiments.yaml"
    path.write_text(yaml.safe_dump(grid, allow_unicode=True), encoding="utf-8")

    plain, clf = run_grid(path)
    assert plain.skipped is None
    assert clf.skipped is not None and "kbrag train-abstention" in clf.skipped


def test_eval_save_calibration_is_refused_for_classifier(
    tmp_path: Path, mini_corpus_dir: Path, mini_gold_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = make_cfg(tmp_path, corpus_dir=str(mini_corpus_dir), gold_path=str(mini_gold_path))
    path = tmp_path / "clf.yaml"
    path.write_text(yaml.safe_dump(config.model_dump(mode="json"), allow_unicode=True), encoding="utf-8")
    assert main(["eval", "--config", str(path), "--save-calibration"]) == EXIT_ERROR
    assert "train-abstention" in capsys.readouterr().err
