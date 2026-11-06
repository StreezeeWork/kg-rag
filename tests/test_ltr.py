"""Learning-to-rank (``kbrag.ltr``): признаки кандидатов, файл модели, встраивание в пайплайн и обучение.

Всё офлайн: мини-корпус из conftest, режим BM25, маленькая сетка моделей — тесты идут секунды.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import pytest
import yaml
from abstention_helpers import train_records, write_jsonl
from fastapi.testclient import TestClient

from kbrag.api.app import create_app
from kbrag.cli import EXIT_ERROR, EXIT_OK, main
from kbrag.config import AppConfig, deep_merge
from kbrag.eval.dataset import item_from_record
from kbrag.eval.runner import run_experiment, run_grid
from kbrag.ingest import corpus_fingerprint, load_corpus
from kbrag.ltr import (
    FEATURE_DESCRIPTIONS,
    FEATURE_NAMES,
    RankerModel,
    candidate_features,
    candidates_fingerprint,
    load_ranker,
    make_ranker,
)
from kbrag.ltr.train import (
    RULE_DOC_CAP,
    RULE_MARGIN,
    RULE_RRF,
    QueryTable,
    choose,
    evaluate_on_dev,
    fit_ranker,
    order_by_scores,
    order_metrics,
    rule_order,
    train_ranker,
)
from kbrag.pipeline import RAGPipeline, calibration_fingerprint
from kbrag.types import Chunk, Document, ScoredChunk

SMALL_GRID = [{"loss_function": "PairLogit", "depth": 2, "iterations": 30}]


def col(name: str) -> int:
    return FEATURE_NAMES.index(name)


def scored(
    chunk_id: str, doc_id: str, score: float, *, status: str = "active", **details: float
) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=f"Отпуск длится {chunk_id} дней",
        start=0,
        end=10,
        title=f"Документ {doc_id}",
        section="Раздел",
        metadata={"status": status},
    )
    return ScoredChunk(chunk=chunk, score=score, source="rerank", details=dict(details))


def items(records: list[dict]) -> list:
    return [item_from_record(record) for record in records]


# ---------------------------------------------------------------------------
# Признаки
# ---------------------------------------------------------------------------


def test_every_feature_is_described_and_columns_follow_feature_names() -> None:
    assert tuple(FEATURE_DESCRIPTIONS) == FEATURE_NAMES
    assert all(FEATURE_DESCRIPTIONS[name].strip() for name in FEATURE_NAMES)
    assert candidate_features("вопрос", []).shape == (0, len(FEATURE_NAMES))


def test_document_features_count_chunks_of_the_same_document_by_rerank_order() -> None:
    # Порядок реранкера: a1 (док A), a2 (док A), b1 (док B). Второй кусок A — rank_in_doc 2, а B — второй документ.
    candidates = [
        scored("a1", "A", 3.0, rerank=3.0, hybrid=0.03, bm25=5.0, bm25_rank=1, dense=0.8, dense_rank=2),
        scored("a2", "A", 2.0, rerank=2.0, hybrid=0.02, dense=0.9, dense_rank=1),
        scored("b1", "B", 1.0, rerank=1.0, hybrid=0.01, bm25=2.5, bm25_rank=2, status="archived"),
    ]
    x = candidate_features("Сколько длится отпуск?", candidates)
    assert x[:, col("rerank_rank")].tolist() == [1, 2, 3]
    assert x[:, col("rank_in_doc")].tolist() == [1, 2, 1]
    assert x[:, col("doc_rank")].tolist() == [1, 1, 2]
    assert x[:, col("doc_n_cands")].tolist() == [2, 2, 1]
    assert x[:, col("rerank_gap_top")].tolist() == [0.0, -1.0, -2.0]
    assert x[0, col("bm25_rel")] == 1.0 and x[2, col("bm25_rel")] == 0.5
    assert x[:, col("archived")].tolist() == [0.0, 0.0, 1.0]
    # Нет score источника — NaN, а не 0: «не нашёл» и «нашёл с нулём» — разные вещи.
    assert math.isnan(x[1, col("bm25")]) and math.isnan(x[1, col("bm25_rank")])
    assert math.isnan(x[2, col("dense")]) and math.isnan(x[2, col("dense_gap_top")])
    # Стемы вопроса: «скольк», «длит», «отпуск»; в тексте фрагмента есть два из трёх.
    assert x[0, col("q_cov")] == pytest.approx(2 / 3)
    assert x[0, col("q_len")] == 3


def test_features_do_not_depend_on_the_order_candidates_arrive_in() -> None:
    candidates = [
        scored(f"c{i}", "AB"[i % 2], float(i), rerank=float(i % 3), hybrid=0.01 * i, dense=0.1 * i)
        for i in range(6)
    ]
    shuffled = candidates[:]
    random.Random(7).shuffle(shuffled)
    x = candidate_features("отпуск", candidates)
    y = candidate_features("отпуск", shuffled)
    position = {item.chunk.chunk_id: i for i, item in enumerate(shuffled)}
    for i, item in enumerate(candidates):
        np.testing.assert_array_equal(x[i], y[position[item.chunk.chunk_id]])


def test_single_retriever_mode_orders_documents_by_the_incoming_score() -> None:
    # Режим bm25: ни реранкера, ни гибрида — признаки документа считаются по score кандидата.
    candidates = [
        scored("a1", "A", 5.0, bm25=5.0),
        scored("b1", "B", 4.0, bm25=4.0),
        scored("a2", "A", 3.0, bm25=3.0),
    ]
    x = candidate_features("отпуск", candidates)
    assert x[:, col("rank_in_doc")].tolist() == [1, 1, 2]
    assert x[:, col("doc_rank")].tolist() == [1, 2, 1]
    assert np.isnan(x[:, col("rerank")]).all() and np.isnan(x[:, col("rrf_rank")]).all()


# ---------------------------------------------------------------------------
# Модель и её файл
# ---------------------------------------------------------------------------


def toy_tables(n_queries: int = 12, split: str = "train", seed: int = 0) -> list[QueryTable]:
    """Синтетика: релевантен кандидат с большим dense, а реранкер ставит его не первым."""
    rng = np.random.default_rng(seed)
    tables = []
    for q in range(n_queries):
        n = 6
        features = np.full((n, len(FEATURE_NAMES)), np.nan)
        relevant = int(rng.integers(1, n))
        dense = rng.uniform(0.2, 0.5, n)
        dense[relevant] = 0.9
        features[:, col("dense")] = dense
        features[:, col("rerank")] = np.linspace(1.0, 0.0, n)
        features[:, col("rerank_rank")] = np.arange(1, n + 1)
        features[:, col("dense_rank")] = (-dense).argsort().argsort() + 1
        labels = np.zeros(n)
        labels[relevant] = 1.0
        coverage = [[bool(label)] for label in labels]
        doc_ids = [f"d{i}" for i in range(n)]
        tables.append(QueryTable(f"q{q}", "factoid", split, features, labels, coverage, doc_ids))
    return tables


def trained_toy_model(metadata: dict | None = None) -> RankerModel:
    """Настоящий маленький CatBoostRanker на синтетике: учится поднимать кандидата с большим dense."""
    tables = toy_tables()
    params = {"loss_function": "PairLogit", "depth": 2, "iterations": 30}
    estimator = make_ranker(params).fit(
        np.vstack([t.features for t in tables]),
        np.concatenate([t.labels for t in tables]),
        group_id=np.concatenate([np.full(len(t.labels), i) for i, t in enumerate(tables)]),
    )
    return RankerModel(estimator, params=params, metadata=metadata)


def test_saved_model_loads_and_scores_the_same(tmp_path: Path) -> None:
    model = trained_toy_model({"fingerprint": "abc"})
    path = model.save(tmp_path / "ltr.json")
    loaded = RankerModel.load(path)
    sample = toy_tables(split="dev", seed=1)[0].features
    np.testing.assert_allclose(loaded.score(sample), model.score(sample))
    assert loaded.name == "catboost:PairLogit-d2" and loaded.fingerprint == "abc"
    assert loaded.feature_names == FEATURE_NAMES

    # Деревья из другого обучения не подсунуть: sha256 в JSON связывает два файла.
    (tmp_path / "ltr.cbm").write_bytes(b"not a model")
    with pytest.raises(ValueError, match="sha256"):
        RankerModel.load(path)


def test_rank_orders_by_model_score_and_keeps_previous_details() -> None:
    class Fixed:
        def predict(self, matrix: np.ndarray) -> np.ndarray:
            return np.array([0.1, 0.9, 0.9])

    model = RankerModel(Fixed(), params={"loss_function": "PairLogit", "depth": 2})
    candidates = [
        scored("c", "A", 3.0, rerank=3.0),
        scored("b", "A", 2.0, rerank=2.0),
        scored("a", "B", 1.0, rerank=1.0),
    ]
    ranked = model.rank("отпуск", candidates)
    # Ничья 0.9/0.9 — по chunk_id; прежние детали сохранены, score модели и прежний score видны.
    assert [item.chunk.chunk_id for item in ranked] == ["a", "b", "c"]
    assert ranked[0].details["ltr"] == 0.9 and ranked[0].details["pre_ltr"] == 1.0
    assert ranked[0].details["rerank"] == 1.0 and ranked[0].source == "ltr"
    assert model.rank("отпуск", []) == []


# ---------------------------------------------------------------------------
# Правила, выбор и dev
# ---------------------------------------------------------------------------


def test_order_metrics_and_rules() -> None:
    table = QueryTable(
        "q",
        "multi_hop",
        "dev",
        np.full((4, len(FEATURE_NAMES)), np.nan),
        np.array([1.0, 1.0, 0.0, 1.0]),
        [[True, False], [True, False], [False, False], [False, True]],
        ["A", "A", "A", "B"],
    )
    # Порядок реранкера: два куска A покрывают одно доказательство, B — четвёртый.
    assert order_metrics(table, range(4), k=2)["recall@2"] == 0.5
    # «Не больше двух кусков документа»: третий кусок A уходит в конец, B поднимается на третье место.
    assert rule_order(RULE_DOC_CAP, table) == [0, 1, 3, 2]
    assert order_metrics(table, rule_order(RULE_DOC_CAP, table), k=3)["recall@3"] == 1.0
    table.features[:, col("rerank_rank")] = [1, 2, 3, 4]
    table.features[:, col("bm25_rank")] = [np.nan, np.nan, np.nan, 1]
    assert rule_order(RULE_RRF, table)[0] == 3  # первое место в BM25 + четвёртое у реранкера
    assert order_by_scores(np.array([0.5, np.nan, 0.5, 1.0])) == [3, 0, 2, 1]


def test_choose_takes_the_model_only_with_a_margin_over_the_best_rule() -> None:
    rules = {RULE_RRF: {"recall@5": 0.900}, RULE_DOC_CAP: {"recall@5": 0.880}}
    cv = [
        {"name": "catboost:PairLogit-d3", "params": {"depth": 3}, "recall@5": 0.909},
        {"name": "catboost:PairLogit-d4", "params": {"depth": 4}, "recall@5": 0.909},  # ничья → проще
    ]
    choice = choose(cv, rules, 5)
    assert choice["kind"] == "model" and choice["name"] == "catboost:PairLogit-d3"
    assert choice["best_rule"] == RULE_RRF and choice["margin"] == pytest.approx(0.009)

    close = [{"name": "m", "params": {}, "recall@5": 0.900 + RULE_MARGIN / 2}]
    choice = choose(close, rules, 5)
    assert choice["kind"] == "rule" and choice["name"] == RULE_RRF and choice["params"] == {}


def test_dev_verdict_follows_the_pre_registered_rule() -> None:
    tables = toy_tables(split="dev")
    best = evaluate_on_dev(tables, lambda t: order_by_scores(t.features[:, col("dense")]), k=1)
    assert best["chosen"]["recall@1"] == 1.0 and best["baseline"]["recall@1"] == 0.0
    assert best["switch"] and all(best["criteria"].values())
    assert best["diff"]["recall@1"]["diff"] == pytest.approx(1.0)

    same = evaluate_on_dev(tables, lambda t: range(len(t.labels)), k=1)
    assert not same["criteria"]["recall_higher"] and not same["switch"]  # «не хуже» — мало, нужно «выше»


def test_fit_ranker_never_sees_test() -> None:
    with pytest.raises(ValueError, match="split='dev'"):
        fit_ranker(toy_tables(), toy_tables(split="test"), k=1, grid=SMALL_GRID)
    with pytest.raises(ValueError, match="split='train'"):
        fit_ranker(toy_tables(split="dev"), toy_tables(split="dev"), k=1, grid=SMALL_GRID)


def test_fit_ranker_learns_what_the_reranker_misses() -> None:
    model, report = fit_ranker(toy_tables(), toy_tables(split="dev", seed=1), k=1, grid=SMALL_GRID)
    assert model is not None and report.choice["kind"] == "model"
    assert report.dev["chosen"]["recall@1"] > report.dev["baseline"]["recall@1"]
    assert report.switch_recommended and set(report.importance) == set(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Сквозное обучение и пайплайн на мини-корпусе
# ---------------------------------------------------------------------------


def make_cfg(tmp_path: Path, corpus_dir: Path, gold_path: Path, **sections: dict) -> AppConfig:
    base = {
        "corpus_dir": corpus_dir.as_posix(),
        "gold_path": gold_path.as_posix(),
        "retrieval": {"mode": "bm25", "top_k": 3, "candidates_k": 6},
        "abstention": {"threshold": None},
        "ltr": {"enabled": False, "model_path": (tmp_path / "ltr.json").as_posix()},
    }
    return AppConfig.model_validate(deep_merge(base, sections))


def reverse_model(cfg: AppConfig, docs: list[Document]) -> RankerModel:
    """Модель «руками»: score = позиция кандидата, то есть порядок реранкера (поиска) задом наперёд."""

    class Reverse:
        def predict(self, matrix: np.ndarray) -> np.ndarray:
            return np.arange(matrix.shape[0], dtype=float)

    return RankerModel(
        Reverse(),
        params={"loss_function": "PairLogit", "depth": 1},
        metadata={"fingerprint": candidates_fingerprint(cfg), "corpus_sha256": corpus_fingerprint(docs)},
    )


def test_train_ranker_end_to_end(
    tmp_path: Path, mini_corpus_dir: Path, mini_gold_path: Path, mini_gold_records: list[dict]
) -> None:
    cfg = make_cfg(tmp_path, mini_corpus_dir, mini_gold_path)
    docs = load_corpus(cfg.corpus_dir)
    report = train_ranker(
        cfg, docs=docs, train=items(train_records()), gold=items(mini_gold_records), grid=SMALL_GRID
    )
    assert report.n_train == 12  # из 22 обучающих вопросов 10 без ответа — ранжировать там нечего
    assert report.n_dev >= 1 and report.k == cfg.retrieval.top_k
    assert set(report.rules) == {RULE_RRF, RULE_DOC_CAP} and len(report.cv) == 1
    assert set(report.dev["criteria"]) == {"recall_higher", "ndcg_not_lower", "mrr_not_lower"}
    if report.choice["kind"] == "rule":
        assert report.model_path is None
        return
    data = json.loads(Path(report.model_path).read_text(encoding="utf-8"))
    assert data["fingerprint"] == candidates_fingerprint(cfg)
    assert data["corpus_sha256"] == corpus_fingerprint(docs)
    for key in ("train_sha256", "dev_sha256", "train_report", "dev_report", "importance", "versions"):
        assert key in data
    # Сервис принимает модель, обученную для его конфига.
    enabled = cfg.model_copy(update={"ltr": cfg.ltr.model_copy(update={"enabled": True})})
    assert load_ranker(enabled, corpus_sha256=corpus_fingerprint(docs)).name == report.choice["name"]


def test_pipeline_with_ltr_refuses_a_missing_or_foreign_model(
    tmp_path: Path, mini_corpus_dir: Path, mini_gold_path: Path
) -> None:
    cfg = make_cfg(tmp_path, mini_corpus_dir, mini_gold_path, ltr={"enabled": True})
    with pytest.raises(ValueError, match="kbrag train-ranker"):
        RAGPipeline.from_config(cfg)
    # Модель для других настроек поиска — тоже ошибка при старте, а не тихий откат на прежний порядок.
    trained_toy_model({"fingerprint": "other"}).save(cfg.ltr.model_path)
    with pytest.raises(ValueError, match="других настроек поиска"):
        load_ranker(cfg)
    trained_toy_model({"fingerprint": candidates_fingerprint(cfg), "corpus_sha256": "x"}).save(
        cfg.ltr.model_path
    )
    with pytest.raises(ValueError, match="другом корпусе"):
        load_ranker(cfg, corpus_sha256="y")


def test_pipeline_uses_the_ranker_and_reports_it(
    tmp_path: Path, mini_corpus_dir: Path, mini_gold_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = make_cfg(tmp_path, mini_corpus_dir, mini_gold_path, ltr={"enabled": True})
    docs = load_corpus(cfg.corpus_dir)
    monkeypatch.setattr("kbrag.pipeline.load_ranker", lambda *_args, **_kwargs: reverse_model(cfg, docs))
    pipeline = RAGPipeline.from_config(cfg, docs=docs)
    assert pipeline.ranking_method == "ltr:catboost:PairLogit-d1"

    candidates, _ = pipeline.candidates("Сколько дней отпуска?")
    retrieved, timings = pipeline.retrieve("Сколько дней отпуска?")
    # С ранкером поиск достаёт candidates_k кандидатов, а не top_k: переставлять есть что.
    assert len(candidates) > cfg.retrieval.top_k
    reversed_ids = [c.chunk.chunk_id for c in reversed(candidates)]
    assert [r.chunk.chunk_id for r in retrieved] == reversed_ids[: cfg.retrieval.top_k]
    assert "rank_ms" in timings and all(r.source == "ltr" for r in retrieved)
    with TestClient(create_app(cfg=cfg, pipeline=pipeline)) as client:
        assert client.get("/health").json()["ranking"] == "ltr:catboost:PairLogit-d1"


def test_ranking_method_without_ltr(tmp_path: Path, mini_corpus_dir: Path, mini_gold_path: Path) -> None:
    pipeline = RAGPipeline.from_config(make_cfg(tmp_path, mini_corpus_dir, mini_gold_path))
    assert pipeline.ranker is None and pipeline.ranking_method == "bm25"


def test_calibration_fingerprint_is_unchanged_without_ltr_and_follows_the_model_file_with_it(
    tmp_path: Path, mini_corpus_dir: Path, mini_gold_path: Path
) -> None:
    cfg = make_cfg(tmp_path, mini_corpus_dir, mini_gold_path)
    # Без LTR — прежний отпечаток: сохранённые калибровки и модели отказа остаются в силе.
    assert calibration_fingerprint(cfg) == candidates_fingerprint(cfg)
    enabled = make_cfg(tmp_path, mini_corpus_dir, mini_gold_path, ltr={"enabled": True})
    trained_toy_model({"fingerprint": candidates_fingerprint(enabled), "created_at": "1"}).save(
        enabled.ltr.model_path
    )
    first = calibration_fingerprint(enabled)
    assert first != calibration_fingerprint(cfg)
    # Переобучили ранкер — порог и модель отказа, подобранные на старой выдаче, больше не подходят.
    trained_toy_model({"fingerprint": candidates_fingerprint(enabled), "created_at": "2"}).save(
        enabled.ltr.model_path
    )
    assert calibration_fingerprint(enabled) != first


def test_grid_skips_the_ltr_step_without_a_model_file(
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
                "name": "bm25_ltr",
                "overrides": {"ltr": {"enabled": True, "model_path": str(tmp_path / "none.json")}},
            },
        ],
    }
    path = tmp_path / "experiments.yaml"
    path.write_text(yaml.safe_dump(grid, allow_unicode=True), encoding="utf-8")
    plain, ltr = run_grid(path)
    assert plain.skipped is None
    assert ltr.skipped is not None and "kbrag train-ranker" in ltr.skipped


def test_eval_runs_through_the_ranker(
    tmp_path: Path, mini_corpus_dir: Path, mini_gold_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = make_cfg(tmp_path, mini_corpus_dir, mini_gold_path, ltr={"enabled": True})
    docs = load_corpus(cfg.corpus_dir)
    monkeypatch.setattr("kbrag.pipeline.load_ranker", lambda *_args, **_kwargs: reverse_model(cfg, docs))
    result = run_experiment(cfg, name="ltr", docs=docs)
    # Оценка идёт через тот же retrieve, что и сервис: в per_question — выдача ранкера.
    assert result.per_question and result.metrics["test"]["n_questions"] >= 1


def test_cli_train_ranker(
    tmp_path: Path,
    mini_corpus_dir: Path,
    mini_gold_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = make_cfg(tmp_path, mini_corpus_dir, mini_gold_path)
    data = cfg.model_dump(mode="json")
    data["train_path"] = write_jsonl(tmp_path / "train.jsonl", train_records()).as_posix()
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    monkeypatch.setattr("kbrag.ltr.train.model_grid", lambda *_args, **_kwargs: SMALL_GRID)
    assert main(["train-ranker", "--config", str(config_path)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "Правило из docs/DESIGN.md" in out and "выбрано:" in out

    data["ltr"]["model_path"] = None
    config_path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    assert main(["train-ranker", "--config", str(config_path)]) == EXIT_ERROR
    assert "ltr.model_path" in capsys.readouterr().err
