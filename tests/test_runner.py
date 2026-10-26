"""Тесты прогона экспериментов: метрики по сплитам, подбор порога только на dev, сетка из YAML.

Все прогоны офлайн (BM25 / гибрид с LSA + extractive) на мини-корпусе и мини-эталоне из conftest:
dev = t-002, t-004; test = t-001, t-003, t-005 (t-005 — неотвечаемый).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from kbrag.config import AppConfig, deep_merge
from kbrag.eval import runner
from kbrag.eval.dataset import GoldItem, item_from_record
from kbrag.eval.metrics import choose_threshold, mean_ignore_none, percentile
from kbrag.eval.report import write_report
from kbrag.eval.results import (
    GENERATION_METRICS,
    PER_QUESTION_FIELDS,
    TIMING_KEYS,
    ExperimentResult,
    metric_names,
    retrieval_metric_names,
)
from kbrag.eval.runner import load_experiments, run_experiment, run_grid
from kbrag.generation import ABSTAIN_TEXT
from kbrag.pipeline import LOW_CONFIDENCE_FLAG, RAGPipeline
from kbrag.retrieval import BM25Retriever
from kbrag.types import Document

# Неотвечаемый dev-вопрос: без него на dev один класс, и порог подбирать не на чем.
EXTRA_DEV_UNANSWERABLE = {
    "id": "t-006",
    "question": "Какой пароль от Wi-Fi в переговорке на пятом этаже?",
    "answer": "",
    "key_facts": [],
    "evidence": [],
    "type": "unanswerable",
    "split": "dev",
}


@pytest.fixture(autouse=True)
def _no_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    # Переменные окружения перекрывают конфиг (см. config._ENV_OVERRIDES) — в тестах они не нужны.
    for name in ("KBRAG_GENERATION_BACKEND", "KBRAG_LLM_BASE_URL", "KBRAG_LLM_MODEL", "KBRAG_CORPUS_DIR"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture()
def gold(mini_gold_records: list[dict]) -> list[GoldItem]:
    return [item_from_record(record) for record in mini_gold_records]


@pytest.fixture()
def gold_with_dev_unanswerable(mini_gold_records: list[dict]) -> list[GoldItem]:
    return [item_from_record(record) for record in [*mini_gold_records, EXTRA_DEV_UNANSWERABLE]]


def make_cfg(**sections) -> AppConfig:
    base = {
        "retrieval": {"mode": "bm25", "top_k": 3, "dense": {"lsa_dim": 16}},
        "abstention": {"threshold": None},
    }
    return AppConfig.model_validate(deep_merge(base, sections))


def rows_by_qid(result: ExperimentResult) -> dict[str, dict]:
    return {row["qid"]: row for row in result.per_question}


# ---------------------------------------------------------------------------
# Структура результата
# ---------------------------------------------------------------------------


def test_result_structure_follows_contract(gold: list[GoldItem], mini_docs: list[Document]) -> None:
    cfg = make_cfg()
    result = run_experiment(cfg, name="bm25", gold=gold, docs=mini_docs)

    assert result.name == "bm25" and result.skipped is None
    assert result.config == cfg.model_dump(mode="json")
    assert result.index_stats["n_docs"] == 4 and result.index_stats["n_chunks"] > 0
    assert set(result.metrics) == {"dev", "test", "all", "by_type"}
    for split in ("dev", "test", "all"):
        assert list(result.metrics[split]) == metric_names(cfg.eval.k_values)  # канонический порядок
    assert [result.metrics[s]["n_questions"] for s in ("dev", "test", "all")] == [2, 3, 5]

    expected_fields = (
        set(PER_QUESTION_FIELDS) | set(retrieval_metric_names(cfg.eval.k_values)) | set(GENERATION_METRICS)
    )
    assert [row["qid"] for row in result.per_question] == [item.qid for item in gold]
    for row in result.per_question:
        assert set(row) == expected_fields


def test_result_is_json_serializable_and_round_trips(
    tmp_path: Path, gold: list[GoldItem], mini_docs: list[Document]
) -> None:
    result = run_experiment(make_cfg(), name="bm25", gold=gold, docs=mini_docs)

    restored = ExperimentResult.from_dict(json.loads(json.dumps(result.to_dict(), ensure_ascii=False)))
    assert restored == result
    leaderboard = write_report([result], tmp_path / "report")
    assert leaderboard.is_file()  # отчёт строится из результата раннера без доработок


def test_loads_gold_and_corpus_from_config_paths(mini_corpus_dir: Path, mini_gold_path: Path) -> None:
    cfg = make_cfg(corpus_dir=mini_corpus_dir.as_posix(), gold_path=mini_gold_path.as_posix())

    result = run_experiment(cfg)

    assert result.name == "default"
    assert result.metrics["all"]["n_questions"] == 5
    assert result.index_stats["n_docs"] == 4


def test_broken_gold_fails_before_indexing(
    mini_gold_records: list[dict], mini_docs: list[Document], monkeypatch: pytest.MonkeyPatch
) -> None:
    records = [dict(r) for r in mini_gold_records]
    records[0]["evidence"] = [{"doc_id": "hr-vacation", "quote": "Такой цитаты в документе нет."}]
    index_calls: list[int] = []
    monkeypatch.setattr(BM25Retriever, "index", lambda self, chunks: index_calls.append(len(chunks)))

    with pytest.raises(ValueError, match="цитата не найдена"):
        run_experiment(make_cfg(), gold=[item_from_record(r) for r in records], docs=mini_docs)
    assert index_calls == []


def test_one_index_per_experiment(
    gold: list[GoldItem], mini_docs: list[Document], monkeypatch: pytest.MonkeyPatch
) -> None:
    index_calls: list[int] = []
    original_index = BM25Retriever.index

    def counting_index(self: BM25Retriever, chunks) -> None:
        index_calls.append(len(chunks))
        original_index(self, chunks)

    monkeypatch.setattr(BM25Retriever, "index", counting_index)
    run_experiment(make_cfg(), gold=gold, docs=mini_docs)

    assert len(index_calls) == 1  # пять вопросов — один индекс


# ---------------------------------------------------------------------------
# Метрики
# ---------------------------------------------------------------------------


def test_retrieval_metrics_only_over_answerable(gold: list[GoldItem], mini_docs: list[Document]) -> None:
    cfg = make_cfg()
    result = run_experiment(cfg, gold=gold, docs=mini_docs)
    rows = rows_by_qid(result)

    unanswerable = rows["t-005"]
    for name in [*retrieval_metric_names(cfg.eval.k_values), "fact_recall", "token_f1", "citation_recall"]:
        assert unanswerable[name] is None
    answerable_rows = [row for row in result.per_question if row["type"] != "unanswerable"]
    assert result.metrics["all"]["recall@5"] == pytest.approx(
        mean_ignore_none([r["recall@5"] for r in answerable_rows])
    )
    test_answerable = [rows["t-001"], rows["t-003"]]
    assert result.metrics["test"]["mrr"] == pytest.approx(sum(r["mrr"] for r in test_answerable) / 2)
    assert result.metrics["test"]["hit@1"] is not None


def test_retrieval_is_evaluated_deeper_than_top_k(gold: list[GoldItem], mini_docs: list[Document]) -> None:
    # Гибрид с LSA находит все 8 фрагментов: Recall@10 считается по 8, а не по top_k=3.
    result = run_experiment(make_cfg(retrieval={"mode": "hybrid"}), gold=gold, docs=mini_docs)

    assert max(len(row["retrieved"]) for row in result.per_question) > 3


def test_custom_k_values(gold: list[GoldItem], mini_docs: list[Document]) -> None:
    result = run_experiment(make_cfg(eval={"k_values": [2]}), gold=gold, docs=mini_docs)

    assert list(result.metrics["test"]) == metric_names([2])
    assert "hit@2" in result.per_question[0] and "hit@5" not in result.per_question[0]
    assert all(len(row["retrieved"]) <= 3 for row in result.per_question)  # max(k_values, top_k)


def test_latency_percentiles_match_per_question(gold: list[GoldItem], mini_docs: list[Document]) -> None:
    result = run_experiment(make_cfg(), gold=gold, docs=mini_docs)
    latencies = [row["latency_ms"] for row in result.per_question]

    assert all(value >= 0 for value in latencies)
    assert result.metrics["all"]["latency_p50_ms"] == round(percentile(latencies, 50), 3)
    assert result.metrics["all"]["latency_p95_ms"] == round(percentile(latencies, 95), 3)


def test_latency_percentiles_are_rounded_like_per_question(
    gold: list[GoldItem], mini_docs: list[Document]
) -> None:
    # Регрессия: перцентили писались как 73.54679999999999 — ложная точность и шумный дифф.
    result = run_experiment(make_cfg(), gold=gold, docs=mini_docs)
    for split in ("dev", "test", "all"):
        for name in ("latency_p50_ms", "latency_p95_ms"):
            value = result.metrics[split][name]
            assert value == round(value, 3)


def _without_timings(obj):
    """Копия результата без замеров времени (TIMING_KEYS) — они единственные недетерминированы."""
    if isinstance(obj, dict):
        return {key: _without_timings(value) for key, value in obj.items() if key not in TIMING_KEYS}
    if isinstance(obj, list):
        return [_without_timings(value) for value in obj]
    return obj


def test_two_runs_differ_only_in_timings(gold: list[GoldItem], mini_docs: list[Document]) -> None:
    # Честная проверка воспроизводимости: два независимых прогона, а не две записи одного объекта.
    cfg = make_cfg(retrieval={"mode": "hybrid"}, abstention={"threshold": 0.3})
    first = run_experiment(cfg, name="repro", gold=gold, docs=mini_docs).to_dict()
    second = run_experiment(cfg, name="repro", gold=gold, docs=mini_docs).to_dict()

    assert _without_timings(first) == _without_timings(second)


def test_by_type_is_built_for_report_split(gold: list[GoldItem], mini_docs: list[Document]) -> None:
    on_test = run_experiment(make_cfg(), gold=gold, docs=mini_docs)
    assert set(on_test.metrics["by_type"]) == {"numeric", "confusable", "unanswerable"}
    assert on_test.metrics["by_type"]["numeric"]["n_questions"] == 1
    assert on_test.metrics["by_type"]["unanswerable"]["recall@5"] is None

    on_dev = run_experiment(make_cfg(eval={"report_split": "dev"}), gold=gold, docs=mini_docs)
    assert set(on_dev.metrics["by_type"]) == {"procedural", "keyword"}


def test_hybrid_confidence_is_dense_cosine(gold: list[GoldItem], mini_docs: list[Document]) -> None:
    result = run_experiment(make_cfg(retrieval={"mode": "hybrid"}), gold=gold, docs=mini_docs)

    confidences = [row["confidence"] for row in result.per_question if row["confidence"] is not None]
    assert confidences and all(value <= 1.0 + 1e-6 for value in confidences)  # косинус
    assert max(confidences) > 0.05  # RRF-score не бывает больше 2/61 ≈ 0.033


# ---------------------------------------------------------------------------
# Порог отказа
# ---------------------------------------------------------------------------


def test_auto_threshold_is_chosen_only_on_dev(
    gold_with_dev_unanswerable: list[GoldItem], mini_docs: list[Document], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[list, list]] = []

    def spy_choose_threshold(confidences, answerable):
        seen.append((list(confidences), list(answerable)))
        return choose_threshold(confidences, answerable)

    monkeypatch.setattr(runner, "choose_threshold", spy_choose_threshold)
    result = run_experiment(
        make_cfg(abstention={"threshold": "auto"}), gold=gold_with_dev_unanswerable, docs=mini_docs
    )

    dev_rows = [row for row in result.per_question if row["split"] == "dev"]
    assert len(seen) == 1
    confidences, answerable = seen[0]
    assert confidences == [row["confidence"] for row in dev_rows]  # ровно dev-вопросы, test не участвуют
    assert answerable == [True, True, False]
    assert result.threshold == choose_threshold(confidences, answerable)
    assert result.threshold is not None


def test_auto_threshold_does_not_depend_on_test_questions(
    gold_with_dev_unanswerable: list[GoldItem], mini_docs: list[Document]
) -> None:
    cfg = make_cfg(abstention={"threshold": "auto"})
    full = run_experiment(cfg, gold=gold_with_dev_unanswerable, docs=mini_docs)
    dev_only = [item for item in gold_with_dev_unanswerable if item.split == "dev"]
    without_test = run_experiment(cfg, gold=dev_only, docs=mini_docs)

    assert full.threshold == without_test.threshold


def test_auto_threshold_none_when_dev_has_one_class(gold: list[GoldItem], mini_docs: list[Document]) -> None:
    # В мини-эталоне на dev только отвечаемые вопросы — отличать не от чего.
    result = run_experiment(make_cfg(abstention={"threshold": "auto"}), gold=gold, docs=mini_docs)

    assert result.threshold is None


def test_disabled_or_null_threshold(gold: list[GoldItem], mini_docs: list[Document]) -> None:
    disabled = make_cfg(abstention={"enabled": False, "threshold": 1e9})
    assert run_experiment(disabled, gold=gold, docs=mini_docs).threshold is None
    assert run_experiment(make_cfg(), gold=gold, docs=mini_docs).threshold is None


def test_huge_threshold_abstains_everywhere(gold: list[GoldItem], mini_docs: list[Document]) -> None:
    result = run_experiment(make_cfg(abstention={"threshold": 1e9}), gold=gold, docs=mini_docs)
    test = result.metrics["test"]

    assert result.threshold == 1e9
    for row in result.per_question:
        assert row["abstained"] and row["answer"] == ABSTAIN_TEXT
        assert LOW_CONFIDENCE_FLAG in row["flags"]
        assert row["citation_precision"] is None and row["invalid_citation_rate"] is None
    assert test["fact_recall"] == 0.0  # отказ на отвечаемый вопрос = 0
    assert test["fact_recall_answered"] is None  # отвеченных нет
    assert test["citation_recall"] == 0.0
    assert test["abstain_recall"] == 1.0 and test["false_abstention_rate"] == 1.0
    assert result.metrics["test"]["recall@5"] is not None  # поиск оценивается независимо от отказа


def test_threshold_is_applied_post_hoc(gold: list[GoldItem], mini_docs: list[Document]) -> None:
    raw = run_experiment(make_cfg(), gold=gold, docs=mini_docs)
    answerable_conf = sorted(row["confidence"] for row in raw.per_question if row["type"] != "unanswerable")
    # Порог между двумя самыми низкими уверенностями: отклоняется ровно один отвечаемый вопрос.
    threshold = (answerable_conf[0] + answerable_conf[1]) / 2

    result = run_experiment(make_cfg(abstention={"threshold": threshold}), gold=gold, docs=mini_docs)
    raw_rows = rows_by_qid(raw)

    for row in result.per_question:
        conf = row["confidence"]
        below = conf is None or conf < threshold
        assert conf == raw_rows[row["qid"]]["confidence"]  # уверенность считается до порога
        assert row["abstained"] == (raw_rows[row["qid"]]["abstained"] or below)
        assert (LOW_CONFIDENCE_FLAG in row["flags"]) == below
        if row["abstained"]:
            assert row["answer"] == ABSTAIN_TEXT
        else:
            assert row["answer"] == raw_rows[row["qid"]]["answer"]

    refused = [r for r in result.per_question if r["type"] != "unanswerable" and r["abstained"]]
    answered = [r for r in result.per_question if r["type"] != "unanswerable" and not r["abstained"]]
    assert len(refused) == 1
    assert refused[0]["fact_recall"] == 0.0 and refused[0]["fact_recall_answered"] is None

    all_metrics = result.metrics["all"]
    answerable_rows = refused + answered
    assert all_metrics["fact_recall"] == pytest.approx(
        mean_ignore_none([r["fact_recall"] for r in answerable_rows])
    )
    assert all_metrics["fact_recall_answered"] == pytest.approx(
        mean_ignore_none([r["fact_recall_answered"] for r in answered])
    )
    assert all_metrics["fact_recall"] < all_metrics["fact_recall_answered"]
    assert all_metrics["false_abstention_rate"] == pytest.approx(1 / 4)


# ---------------------------------------------------------------------------
# Сетка экспериментов
# ---------------------------------------------------------------------------


GRID_EXPERIMENTS = [
    {"name": "bm25", "overrides": {"retrieval": {"mode": "bm25"}}},
    {"name": "hybrid-lsa", "overrides": {"retrieval": {"mode": "hybrid"}}},
    {
        "name": "dense-e5",
        "requires_neural": True,
        "overrides": {"retrieval": {"mode": "dense", "dense": {"embedder": "sentence-transformers"}}},
    },
    {"name": "bm25-heavy", "heavy": True, "overrides": {"retrieval": {"mode": "bm25", "top_k": 5}}},
]


def write_grid(
    tmp_path: Path, corpus_dir: Path, gold_path: Path, experiments: list[dict], base_ref: str = "base.yaml"
) -> Path:
    """Пишет conf/base.yaml и conf/experiments.yaml; ``base_ref`` — как сослаться на base."""
    conf = tmp_path / "conf"
    conf.mkdir(exist_ok=True)
    base = {
        "corpus_dir": corpus_dir.as_posix(),
        "gold_path": gold_path.as_posix(),
        "chunking": {"chunk_size": 50},
        "retrieval": {"top_k": 3, "dense": {"lsa_dim": 16}},
        "abstention": {"threshold": None},
    }
    (conf / "base.yaml").write_text(yaml.safe_dump(base, allow_unicode=True), encoding="utf-8")
    path = conf / "experiments.yaml"
    grid = {"base": base_ref, "experiments": experiments}
    path.write_text(yaml.safe_dump(grid, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture()
def no_neural(monkeypatch: pytest.MonkeyPatch) -> None:
    # В окружении разработчика sentence-transformers может стоять — имитируем его отсутствие.
    monkeypatch.setattr(runner, "neural_available", lambda: False)


def test_run_grid_runs_merges_and_skips(
    tmp_path: Path, mini_corpus_dir: Path, mini_gold_path: Path, no_neural: None
) -> None:
    path = write_grid(tmp_path, mini_corpus_dir, mini_gold_path, GRID_EXPERIMENTS)

    results = run_grid(path)

    assert [r.name for r in results] == ["bm25", "hybrid-lsa", "dense-e5", "bm25-heavy"]
    bm25, hybrid, dense, heavy = results
    assert bm25.skipped is None and hybrid.skipped is None
    assert bm25.metrics["test"]["n_questions"] == 3
    # overrides сливаются с base рекурсивно: соседние ключи из base сохраняются.
    assert bm25.config["retrieval"]["mode"] == "bm25" and bm25.config["retrieval"]["top_k"] == 3
    assert (
        hybrid.config["retrieval"]["mode"] == "hybrid"
        and hybrid.config["retrieval"]["dense"]["lsa_dim"] == 16
    )
    assert bm25.config["chunking"]["chunk_size"] == 50

    assert dense.skipped is not None and "pip install" in dense.skipped
    assert dense.config["retrieval"]["dense"]["embedder"] == "sentence-transformers"
    assert dense.metrics == {} and dense.per_question == []
    assert heavy.skipped is not None and "include_heavy" in heavy.skipped


def test_run_grid_include_heavy_and_only(
    tmp_path: Path, mini_corpus_dir: Path, mini_gold_path: Path, no_neural: None
) -> None:
    path = write_grid(tmp_path, mini_corpus_dir, mini_gold_path, GRID_EXPERIMENTS)

    results = run_grid(path, only=["bm25-heavy"], include_heavy=True)

    assert [r.name for r in results] == ["bm25-heavy"]
    assert results[0].skipped is None
    assert results[0].config["retrieval"]["top_k"] == 5

    with pytest.raises(ValueError, match="Неизвестные эксперименты"):
        run_grid(path, only=["no-such-experiment"])


def test_run_grid_detects_neural_config_without_flag(
    tmp_path: Path, mini_corpus_dir: Path, mini_gold_path: Path, no_neural: None
) -> None:
    experiments = [{"name": "rerank-forgot-flag", "overrides": {"rerank": {"enabled": True}}}]
    path = write_grid(tmp_path, mini_corpus_dir, mini_gold_path, experiments)

    (result,) = run_grid(path)

    assert result.skipped is not None and "sentence-transformers" in result.skipped


def test_run_grid_builds_one_pipeline_per_experiment_and_loads_corpus_once(
    tmp_path: Path,
    mini_corpus_dir: Path,
    mini_gold_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    no_neural: None,
) -> None:
    path = write_grid(tmp_path, mini_corpus_dir, mini_gold_path, GRID_EXPERIMENTS)
    built: list[str] = []
    loaded: list[str] = []
    original_from_config = RAGPipeline.from_config
    original_load_corpus = runner.load_corpus

    def counting_from_config(cfg: AppConfig, docs: list[Document] | None = None) -> RAGPipeline:
        built.append(cfg.retrieval.mode)
        return original_from_config(cfg, docs=docs)

    def counting_load_corpus(corpus_dir):
        loaded.append(str(corpus_dir))
        return original_load_corpus(corpus_dir)

    monkeypatch.setattr(RAGPipeline, "from_config", counting_from_config)
    monkeypatch.setattr(runner, "load_corpus", counting_load_corpus)

    run_grid(path)

    assert built == ["bm25", "hybrid"]  # пропущенные эксперименты индекс не строят
    assert len(loaded) == 1


def test_base_path_relative_to_file_or_cwd(
    tmp_path: Path, mini_corpus_dir: Path, mini_gold_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    relative_to_file = write_grid(tmp_path, mini_corpus_dir, mini_gold_path, GRID_EXPERIMENTS[:1])
    base, specs = load_experiments(relative_to_file)
    assert base == relative_to_file.parent / "base.yaml"
    assert [s.name for s in specs] == ["bm25"]

    # «base: conf/base.yaml» — путь от текущего каталога (корня проекта).
    relative_to_cwd = write_grid(
        tmp_path, mini_corpus_dir, mini_gold_path, GRID_EXPERIMENTS[:1], "conf/base.yaml"
    )
    monkeypatch.chdir(tmp_path)
    base, _ = load_experiments(relative_to_cwd)
    assert base is not None and base.resolve() == (tmp_path / "conf" / "base.yaml").resolve()

    missing = write_grid(tmp_path, mini_corpus_dir, mini_gold_path, GRID_EXPERIMENTS[:1], "nope.yaml")
    with pytest.raises(FileNotFoundError, match="nope.yaml"):
        load_experiments(missing)


def test_experiments_without_base_use_defaults(tmp_path: Path) -> None:
    path = tmp_path / "experiments.yaml"
    path.write_text(yaml.safe_dump({"experiments": [{"name": "only-name"}]}), encoding="utf-8")

    base, specs = load_experiments(path)

    assert base is None
    assert specs[0].overrides == {} and not specs[0].requires_neural and not specs[0].heavy


@pytest.mark.parametrize(
    ("experiments", "message"),
    [
        ([], "непустым списком"),
        ([{"name": "a"}, {"name": "a"}], "повторяется"),
        ([{"name": "a", "requires_nueral": True}], "неизвестные ключи"),
        ([{"name": "a", "overrides": ["retrieval"]}], "словарём"),
        ([{"overrides": {}}], "name"),
    ],
)
def test_invalid_experiments_file(
    tmp_path: Path, mini_corpus_dir: Path, mini_gold_path: Path, experiments: list, message: str
) -> None:
    path = write_grid(tmp_path, mini_corpus_dir, mini_gold_path, experiments)

    with pytest.raises(ValueError, match=message):
        load_experiments(path)


def test_typo_in_overrides_fails_even_for_skipped_experiment(
    tmp_path: Path, mini_corpus_dir: Path, mini_gold_path: Path, no_neural: None
) -> None:
    experiments = [{"name": "typo", "heavy": True, "overrides": {"retrieval": {"mod": "bm25"}}}]
    path = write_grid(tmp_path, mini_corpus_dir, mini_gold_path, experiments)

    with pytest.raises(ValueError, match="mod"):  # pydantic.ValidationError — подкласс ValueError
        run_grid(path)
