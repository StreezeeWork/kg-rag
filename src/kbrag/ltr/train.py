"""Обучение learning-to-rank: кандидаты → CV по вопросам на train → выбор → dev против порядка реранкера.

Схема (``train_ranker``, команда ``kbrag train-ranker``)::

    train.jsonl (вопросы с ответом) ──► 30 кандидатов, признаки, разметка ──► CV 5×3 по вопросам:
                                                                              CatBoost (функция потерь × глубина)
    те же вопросы ──► правила без обучения (RRF рангов, «≤2 куска документа») ──┐        │
                                                                                ▼        ▼
                                выбор по Recall@k: модель берётся, только если опережает лучшее правило
                                хотя бы на RULE_MARGIN ──► финальная модель на всём train
    gold.jsonl ──► только dev ──► выбранный порядок против порядка реранкера ──► вердикт правила

Правило решения записано в docs/DESIGN.md (раздел «Learning-to-rank») до первого обучения; константы
ниже — его числа. Почему всё устроено именно так:

* **Разметка — то же правило, что у метрик.** Кандидат релевантен, если покрывает доказательство
  (``eval.dataset.is_relevant``). Модель учится ровно тому, за что потом ставят Recall.
* **Вопросы без ответа в обучение не идут**: у них нет релевантных кандидатов, ранжировать нечего.
  Их отсекает модель отказа, а не ранкер.
* **CV по вопросам, а не по строкам.** Кандидаты одного вопроса похожи друг на друга (общий вопрос,
  общий топ документов). Если раскидать их по фолдам, модель видела бы «соседей» проверочных строк,
  и оценка вышла бы оптимистичной. Поэтому фолд — это набор вопросов целиком.
* **Правила без обучения — обязательный соперник.** Если простое правило работает не хуже модели,
  модель не нужна: она сложнее, требует файла и переобучения. Правилам нечего подгонять под train,
  поэтому их оценка на всём train честная и без CV.
* **Test сюда не попадает вообще.** ``fit_ranker`` проверяет ``split`` у переданных таблиц; test честно
  меряет ``kbrag eval`` уже готовой модели.
"""

from __future__ import annotations

import importlib.metadata
import logging
import platform
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from kbrag.abstention.models import CATBOOST_INSTALL_HINT, catboost_available, import_catboost
from kbrag.config import AppConfig
from kbrag.eval.dataset import (
    TRAIN_SPLIT,
    GoldItem,
    evidence_spans,
    items_fingerprint,
    load_gold,
    train_gold_conflicts,
    validate_gold,
)
from kbrag.eval.metrics import coverage_matrix, mrr, ndcg_at_k, paired_bootstrap_diff, recall_at_k
from kbrag.ingest import corpus_fingerprint, load_corpus
from kbrag.ltr.features import FEATURE_NAMES, candidate_features
from kbrag.ltr.models import LOSSES, RankerModel, candidates_fingerprint, make_ranker
from kbrag.pipeline import RAGPipeline
from kbrag.types import Document

logger = logging.getLogger(__name__)

CV_SPLITS = 5
CV_REPEATS = 3
DEPTH_GRID = (3, 4, 6)
# Числа правила из docs/DESIGN.md. Модель должна опередить лучшее правило по Recall@k хотя бы на столько.
RULE_MARGIN = 0.005
# На dev выбранный порядок может уступить порядку реранкера по nDCG и MRR не больше чем на столько.
NDCG_TOLERANCE = 0.01
MRR_TOLERANCE = 0.02
# Ничья в CV (одинаковая Recall@k) решается в пользу более простой модели (меньше глубина).
TIE_TOLERANCE = 1e-9
RRF_K = 60  # как у гибрида по умолчанию (retrieval.hybrid.rrf_k)
DOC_CAP = 2
IMPORTANCE_KIND = "LossFunctionChange"

BASELINE = "rerank"  # порядок A: как кандидаты пришли от реранкера (без него — от поиска)
RULE_RRF = "rule:rrf_ranks"
RULE_DOC_CAP = "rule:doc_cap2"
RULES = (RULE_RRF, RULE_DOC_CAP)


@dataclass
class QueryTable:
    """Кандидаты одного вопроса: признаки, разметка и покрытие доказательств (в порядке реранкера)."""

    qid: str
    qtype: str
    split: str
    features: np.ndarray
    labels: np.ndarray
    coverage: list[list[bool]]
    doc_ids: list[str]


def candidate_table(
    pipeline: RAGPipeline, items: Sequence[GoldItem], docs: Sequence[Document]
) -> list[QueryTable]:
    """Кандидаты, признаки и разметка для вопросов с ответом — тем же кодом, что в сервисе.

    Кандидаты берутся из ``RAGPipeline.candidates``: поиск + реранкер по всем ``candidates_k``, ещё до модели
    ранжирования (её ``candidates`` не применяет). Модель учится переставлять ровно то, что приходит к ней в сервисе.
    """
    docs_by_id = {doc.doc_id: doc for doc in docs}
    cfg = pipeline.cfg
    n_candidates = max(cfg.retrieval.candidates_k, cfg.retrieval.top_k)
    tables: list[QueryTable] = []
    for item in items:
        if not item.answerable:
            continue
        candidates, _ = pipeline.candidates(item.question, top_k=n_candidates)
        coverage = coverage_matrix([c.chunk for c in candidates], evidence_spans(item, docs_by_id))
        tables.append(
            QueryTable(
                qid=item.qid,
                qtype=item.qtype,
                split=item.split,
                features=candidate_features(item.question, candidates, with_header=cfg.chunking.with_header),
                labels=np.array([float(any(row)) for row in coverage]),
                coverage=coverage,
                doc_ids=[c.chunk.doc_id for c in candidates],
            )
        )
    return tables


# ---------------------------------------------------------------------------
# Метрики порядка и правила без обучения
# ---------------------------------------------------------------------------


def order_metrics(
    table: QueryTable, order: Sequence[int], k: int, mrr_at: int | None = None
) -> dict[str, float]:
    """Recall@k, nDCG@k и MRR вопроса при порядке кандидатов ``order`` (индексы строк таблицы).

    ``mrr_at`` — сколько первых мест смотрит MRR. ``kbrag eval`` хранит выдачу длины ``max(eval.k_values)``
    (10) и MRR считает по ней; чтобы вердикт на dev предсказывал отчёт оценки, здесь та же отсечка.
    """
    coverage = [table.coverage[i] for i in order]
    if not coverage:
        return {f"recall@{k}": 0.0, f"ndcg@{k}": 0.0, "mrr": 0.0}
    return {
        f"recall@{k}": recall_at_k(coverage, k),
        f"ndcg@{k}": ndcg_at_k(coverage, k),
        "mrr": mrr(coverage[:mrr_at] if mrr_at else coverage),
    }


def order_by_scores(scores: np.ndarray) -> list[int]:
    """Индексы по убыванию score; при равенстве — исходный порядок (то есть порядок реранкера).

    Тот же tie-break, что у ``RankerModel.rank`` в сервисе (стабильная сортировка по score), — иначе
    равные score переставлялись бы при обучении и в проде по-разному.
    """
    values = np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=-np.inf)
    return [int(i) for i in np.lexsort((np.arange(len(values)), -values))]


def rule_order(name: str, table: QueryTable) -> list[int]:
    """Порядок кандидатов по правилу без обучения (``RULES``)."""
    n = len(table.labels)
    if name == RULE_RRF:
        # RRF по рангам трёх сигналов: Σ 1 / (60 + rank); фрагмент без ранга в сигнале ничего от него не получает.
        scores = np.zeros(n)
        for feature in ("rerank_rank", "bm25_rank", "dense_rank"):
            ranks = table.features[:, FEATURE_NAMES.index(feature)]
            scores += np.where(np.isnan(ranks), 0.0, 1.0 / (RRF_K + np.nan_to_num(ranks, nan=0.0)))
        return order_by_scores(scores)
    if name == RULE_DOC_CAP:
        # Порядок реранкера, но не больше DOC_CAP кусков одного документа подряд в начале выдачи.
        taken: list[int] = []
        deferred: list[int] = []
        per_doc: dict[str, int] = {}
        for idx, doc_id in enumerate(table.doc_ids):
            if per_doc.get(doc_id, 0) < DOC_CAP:
                taken.append(idx)
                per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
            else:
                deferred.append(idx)
        return taken + deferred
    raise ValueError(f"неизвестное правило {name!r}; есть {RULES}")


def evaluate_orders(
    tables: Sequence[QueryTable],
    order_fn: Callable[[QueryTable], Sequence[int]],
    k: int,
    mrr_at: int | None = None,
) -> dict[str, Any]:
    """Средние метрики порядка по вопросам и Recall@k по типам вопросов."""
    rows = [order_metrics(table, order_fn(table), k, mrr_at) for table in tables]
    return {**_mean_metrics(rows, k), "per_type": _per_type(tables, rows, k), "per_question": rows}


# ---------------------------------------------------------------------------
# Кросс-валидация моделей
# ---------------------------------------------------------------------------


def model_grid(losses: Sequence[str] = LOSSES, depths: Sequence[int] = DEPTH_GRID) -> list[dict[str, Any]]:
    """Кандидаты-модели: каждая функция потерь × каждая глубина (от простой к сложной внутри потерь)."""
    return [{"loss_function": loss, "depth": depth} for loss in losses for depth in depths]


def cross_validate(
    tables: Sequence[QueryTable],
    k: int,
    grid: Sequence[Mapping[str, Any]] | None = None,
    seed: int = 0,
    mrr_at: int | None = None,
) -> list[dict[str, Any]]:
    """CV_REPEATS × CV_SPLITS фолдов по вопросам; у всех кандидатов одни и те же фолды (сравнение парное).

    На каждом фолде модель обучается на вопросах остальных фолдов и переставляет кандидатов
    проверочных. Итог кандидата — средние метрики проверочных вопросов и разброс по фолдам.
    """
    grid = list(grid) if grid is not None else model_grid()
    n = len(tables)
    folds = [_query_folds(n, seed + repeat) for repeat in range(CV_REPEATS)]
    results: list[dict[str, Any]] = []
    for params in grid:
        per_question = np.zeros((CV_REPEATS, n, 3))
        fold_recalls: list[float] = []
        for repeat, fold_of in enumerate(folds):
            for fold in range(CV_SPLITS):
                train_idx = [i for i in range(n) if fold_of[i] != fold]
                test_idx = [i for i in range(n) if fold_of[i] == fold]
                if not test_idx or not train_idx:
                    continue
                model = _fit([tables[i] for i in train_idx], {**params, "random_seed": seed + repeat})
                for i in test_idx:
                    order = order_by_scores(model.predict(tables[i].features))
                    metrics = order_metrics(tables[i], order, k, mrr_at)
                    per_question[repeat, i] = [metrics[f"recall@{k}"], metrics[f"ndcg@{k}"], metrics["mrr"]]
                fold_recalls.append(float(per_question[repeat, test_idx, 0].mean()))
        mean_q = per_question.mean(axis=0)
        rows = [{f"recall@{k}": r, f"ndcg@{k}": d, "mrr": m} for r, d, m in mean_q]
        results.append(
            {
                "name": _model_name(params),
                "params": dict(params),
                **_mean_metrics(rows, k),
                f"recall@{k}_std": float(np.std(fold_recalls)),
                "per_type": _per_type(tables, rows, k),
                "n_folds": len(fold_recalls),
            }
        )
    return results


def choose(cv: Sequence[Mapping[str, Any]], rules: Mapping[str, Mapping[str, Any]], k: int) -> dict[str, Any]:
    """Выбор по правилу из DESIGN.md: лучшая модель против лучшего правила по Recall@k на train."""
    key = f"recall@{k}"
    # Ничья (в пределах TIE_TOLERANCE) — в пользу меньшей глубины, а при равной глубине — более раннего
    # в сетке. Сравнивать «строго больше по порядку сетки» нельзя: сетка идёт по функциям потерь,
    # и при ничьей между потерями победила бы более глубокая модель.
    best_value = max(row[key] for row in cv)
    tied = [row for row in cv if row[key] >= best_value - TIE_TOLERANCE]
    best_model = min(tied, key=lambda row: (row["params"].get("depth", 0), cv.index(row)))
    best_rule_name = max(rules, key=lambda name: (rules[name][key], -RULES.index(name)))
    margin = best_model[key] - rules[best_rule_name][key]
    chosen_kind = "model" if margin >= RULE_MARGIN else "rule"
    return {
        "kind": chosen_kind,
        "name": best_model["name"] if chosen_kind == "model" else best_rule_name,
        "params": dict(best_model["params"]) if chosen_kind == "model" else {},
        "best_model": best_model["name"],
        "best_rule": best_rule_name,
        "margin": float(margin),
    }


# ---------------------------------------------------------------------------
# Dev: выбранный порядок против порядка реранкера
# ---------------------------------------------------------------------------


def evaluate_on_dev(
    tables: Sequence[QueryTable],
    order_fn: Callable[[QueryTable], Sequence[int]],
    k: int,
    mrr_at: int | None = None,
) -> dict[str, Any]:
    """Метрики A (порядок реранкера) и выбранного порядка на одних вопросах dev, разницы и вердикт правила."""
    baseline = evaluate_orders(tables, lambda table: range(len(table.labels)), k, mrr_at)
    chosen = evaluate_orders(tables, order_fn, k, mrr_at)
    diffs = {}
    for metric in (f"recall@{k}", f"ndcg@{k}", "mrr"):
        result = paired_bootstrap_diff(
            [row[metric] for row in chosen["per_question"]], [row[metric] for row in baseline["per_question"]]
        )
        diffs[metric] = None if result is None else {"diff": result[0], "ci": [result[1], result[2]]}
    criteria = {
        "recall_higher": chosen[f"recall@{k}"] > baseline[f"recall@{k}"] + TIE_TOLERANCE,
        "ndcg_not_lower": chosen[f"ndcg@{k}"] >= baseline[f"ndcg@{k}"] - NDCG_TOLERANCE,
        "mrr_not_lower": chosen["mrr"] >= baseline["mrr"] - MRR_TOLERANCE,
    }
    return {
        "n_questions": len(tables),
        "baseline": _without_per_question(baseline),
        "chosen": _without_per_question(chosen),
        "diff": diffs,
        "criteria": criteria,
        "switch": all(criteria.values()),
    }


# ---------------------------------------------------------------------------
# Отчёт и обучение целиком
# ---------------------------------------------------------------------------


@dataclass
class RankerTrainReport:
    """Всё, что показывает ``kbrag train-ranker``: train (A, правила, CV), выбор, dev и важность признаков."""

    k: int
    n_train: int
    n_dev: int
    baseline_train: dict[str, Any]
    rules: dict[str, dict[str, Any]]
    cv: list[dict[str, Any]]
    choice: dict[str, Any]
    dev: dict[str, Any]
    importance: dict[str, float] = field(default_factory=dict)
    importance_kind: str = IMPORTANCE_KIND
    mrr_at: int | None = None
    model_path: str | None = None
    # Модель отвергнута на dev, а боевой конфиг уже на LTR: файл записан рядом, а не поверх боевого.
    saved_as_candidate: bool = False

    @property
    def switch_recommended(self) -> bool:
        """Выполнено ли на dev правило из DESIGN.md (переключать ли боевой конфиг на выбранный порядок)."""
        return bool(self.dev.get("switch"))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["switch_recommended"] = self.switch_recommended
        return data


def fit_ranker(
    train: Sequence[QueryTable],
    dev: Sequence[QueryTable],
    k: int,
    grid: Sequence[Mapping[str, Any]] | None = None,
    seed: int = 0,
    mrr_at: int | None = None,
) -> tuple[RankerModel | None, RankerTrainReport]:
    """Ядро обучения на готовых таблицах: правила и CV на train, выбор, финальная модель, сравнение на dev.

    Модель — None, если по правилу выбран не CatBoost, а правило без обучения: файл модели тогда не нужен.
    """
    _require_split(train, TRAIN_SPLIT, "обучающие вопросы")
    _require_split(dev, "dev", "вопросы для решения")
    baseline_train = _without_per_question(
        evaluate_orders(train, lambda table: range(len(table.labels)), k, mrr_at)
    )
    rules = {
        name: _without_per_question(evaluate_orders(train, lambda t, n=name: rule_order(n, t), k, mrr_at))
        for name in RULES
    }
    cv = cross_validate(train, k, grid=grid, seed=seed, mrr_at=mrr_at)
    choice = choose(cv, rules, k)

    model: RankerModel | None = None
    importance: dict[str, float] = {}
    if choice["kind"] == "model":
        estimator = _fit(train, {**choice["params"], "random_seed": seed})
        model = RankerModel(estimator, params=choice["params"])
        # LossFunctionChange: насколько ухудшается функция потерь ранжирования без признака. Для ранкеров
        # это осмысленная мера; PredictionValuesChange смотрит на сдвиг score, а у ранкера важен только порядок.
        values = estimator.get_feature_importance(data=_pool(train), type=IMPORTANCE_KIND)
        importance = {name: float(value) for name, value in zip(FEATURE_NAMES, values, strict=True)}

        def order_fn(table: QueryTable) -> Sequence[int]:
            return order_by_scores(model.score(table.features))
    else:

        def order_fn(table: QueryTable) -> Sequence[int]:
            return rule_order(choice["name"], table)

    report = RankerTrainReport(
        k=k,
        n_train=len(train),
        n_dev=len(dev),
        baseline_train=baseline_train,
        rules=rules,
        cv=cv,
        choice=choice,
        dev=evaluate_on_dev(dev, order_fn, k, mrr_at),
        importance=importance,
        mrr_at=mrr_at,
    )
    return model, report


def train_ranker(
    cfg: AppConfig,
    out: str | Path | None = None,
    *,
    docs: Sequence[Document] | None = None,
    train: Sequence[GoldItem] | None = None,
    gold: Sequence[GoldItem] | None = None,
    grid: Sequence[Mapping[str, Any]] | None = None,
    seed: int = 0,
) -> RankerTrainReport:
    """``kbrag train-ranker``: обучает модель ранжирования для конфига ``cfg`` и сохраняет её.

    Модель сохраняется, если выбран CatBoost, — даже когда на dev правило переключения не выполнено:
    шаг сетки с LTR должен воспроизводиться, а отрицательный результат — проверяться. Решение о боевом
    конфиге показывает ``RankerTrainReport.switch_recommended``. Путь — ``out`` или ``ltr.model_path``.

    Одно исключение: боевой конфиг уже на LTR (``ltr.enabled``), путь — его ``ltr.model_path``, а новая
    модель на dev правило не прошла. Тогда файл пишется рядом (``<имя>.candidate.json``): отвергнутая
    модель не должна молча попасть в сервис при следующем старте.
    """
    if not catboost_available():
        # До часового прохода по кандидатам, а не после него.
        raise ImportError(f"для модели ранжирования нужен пакет catboost: {CATBOOST_INSTALL_HINT}")
    target = out if out is not None else cfg.ltr.model_path
    if not target:
        raise ValueError("не задан путь к модели: ltr.model_path в конфиге или --out")
    if Path(target).suffix.lower() != ".json":
        raise ValueError(
            f"путь к модели должен оканчиваться на .json (деревья лягут рядом в .cbm), а не {target}"
        )
    docs = list(docs) if docs is not None else load_corpus(cfg.corpus_dir)
    train = list(train) if train is not None else load_gold(cfg.train_path)
    gold = list(gold) if gold is not None else load_gold(cfg.gold_path)

    problems = validate_gold(train, docs, splits=(TRAIN_SPLIT,)) + train_gold_conflicts(train, gold)
    if problems:
        shown = "\n  ".join(problems[:10])
        more = f"\n  … и ещё {len(problems) - 10}" if len(problems) > 10 else ""
        raise ValueError(f"обучающий набор не годится ({len(problems)} проблем):\n  {shown}{more}")
    # Test отбрасываем сразу после чтения: дальше в обучение и выбор уходит только dev.
    dev = [item for item in gold if item.split == "dev"]

    # Кандидатам нужен только поиск с реранкером: без старой модели ранжирования (её мы как раз обучаем)
    # и без порога и модели отказа (на кандидатов они не влияют, а грузить их незачем).
    feature_cfg = cfg.model_copy(
        update={
            "ltr": cfg.ltr.model_copy(update={"enabled": False}),
            "abstention": cfg.abstention.model_copy(update={"method": "threshold", "threshold": None}),
        }
    )
    pipeline = RAGPipeline.from_config(feature_cfg, docs=docs)
    train_tables = candidate_table(pipeline, train, docs)
    dev_tables = candidate_table(pipeline, dev, docs)
    model, report = fit_ranker(
        train_tables, dev_tables, cfg.retrieval.top_k, grid=grid, seed=seed, mrr_at=eval_mrr_cutoff(cfg)
    )
    if model is None:
        return report

    model.metadata.update(
        {
            "fingerprint": candidates_fingerprint(cfg),
            "corpus_sha256": corpus_fingerprint(list(docs)),
            "train_sha256": items_fingerprint(train),
            "dev_sha256": items_fingerprint(dev),
            "k": report.k,
            "train_report": {
                "baseline": report.baseline_train,
                "rules": report.rules,
                "cv": report.cv,
                "choice": report.choice,
            },
            "dev_report": report.dev,
            "mrr_at": report.mrr_at,
            "importance": {"kind": report.importance_kind, "values": report.importance},
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "versions": _library_versions(),
        }
    )
    served = cfg.ltr.model_path is not None and Path(target) == Path(cfg.ltr.model_path)
    if cfg.ltr.enabled and served and not report.switch_recommended:
        target = Path(target).with_name(f"{Path(target).stem}.candidate.json")
        report.saved_as_candidate = True
    report.model_path = str(model.save(target))
    return report


def eval_mrr_cutoff(cfg: AppConfig) -> int:
    """Длина выдачи, по которой ``kbrag eval`` считает MRR: ``max(eval.k_values)`` (не меньше top_k)."""
    return max([*cfg.eval.k_values, cfg.retrieval.top_k])


def out_of_fold_rankers(
    pipeline: RAGPipeline,
    items: Sequence[GoldItem],
    docs: Sequence[Document],
    params: Mapping[str, Any],
    seed: int = 0,
) -> dict[str, RankerModel]:
    """Для каждого вопроса ``items`` — модель ранжирования, которая его не видела (out-of-fold).

    Зачем. Модель отказа учится на признаках выдачи, а выдачу задаёт ранкер, обученный на тех же
    train-вопросах. Для «своих» вопросов ранкер ставит нужный фрагмент выше, чем сделает на новых,
    и признаки train выходят оптимистичнее, чем в сервисе, — классическая утечка при стекинге моделей.
    Поэтому вопросы делятся на CV_SPLITS фолдов, и выдачу вопроса из фолда f строит ранкер, обученный
    без фолда f (с теми же гиперпараметрами, что боевой). Вопросы без ответа тоже раскладываются
    по фолдам: так у обоих классов выдача от модели на ~80% данных, и разница классов не подмешана.
    """
    tables = {table.qid: table for table in candidate_table(pipeline, items, docs)}
    fold_of = _query_folds(len(items), seed)
    rankers: dict[int, RankerModel] = {}
    for fold in range(CV_SPLITS):
        seen = [items[i].qid for i in range(len(items)) if fold_of[i] != fold and items[i].qid in tables]
        estimator = _fit([tables[qid] for qid in seen], {**params, "random_seed": seed})
        rankers[fold] = RankerModel(estimator, params=params, metadata={"trained_on": seen})
    return {item.qid: rankers[int(fold_of[i])] for i, item in enumerate(items)}


# ---------------------------------------------------------------------------
# Помощники
# ---------------------------------------------------------------------------


def _fit(tables: Sequence[QueryTable], params: Mapping[str, Any]) -> Any:
    """CatBoostRanker на кандидатах вопросов: строки одного вопроса идут подряд и имеют общий group_id."""
    return make_ranker(params).fit(_pool(tables))


def _pool(tables: Sequence[QueryTable]) -> Any:
    """CatBoost Pool: строки одного вопроса подряд с общим group_id (так ранкер знает, кого с кем сравнивать)."""
    matrix = np.vstack([table.features for table in tables])
    labels = np.concatenate([table.labels for table in tables])
    groups = np.concatenate([np.full(len(table.labels), i) for i, table in enumerate(tables)])
    return import_catboost().Pool(matrix, labels, group_id=groups)


def _query_folds(n: int, seed: int) -> np.ndarray:
    """Номер фолда для каждого вопроса: равные по размеру фолды, случайно перемешанные (seed)."""
    return np.random.default_rng(seed).permutation(np.arange(n) % CV_SPLITS)


def _model_name(params: Mapping[str, Any]) -> str:
    return f"catboost:{params['loss_function']}-d{params['depth']}"


def _mean_metrics(rows: Sequence[Mapping[str, float]], k: int) -> dict[str, float]:
    keys = (f"recall@{k}", f"ndcg@{k}", "mrr")
    if not rows:
        return {key: 0.0 for key in keys}
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def _per_type(tables: Sequence[QueryTable], rows: Sequence[Mapping[str, float]], k: int) -> dict[str, dict]:
    """Recall@k по типам вопросов: на multi_hop и перефразах порядок решает больше всего."""
    by_type: dict[str, list[float]] = {}
    for table, row in zip(tables, rows, strict=True):
        by_type.setdefault(table.qtype, []).append(row[f"recall@{k}"])
    return {
        qtype: {"n": len(values), f"recall@{k}": float(np.mean(values))}
        for qtype, values in sorted(by_type.items())
    }


def _without_per_question(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "per_question"}


def _require_split(tables: Sequence[QueryTable], split: str, what: str) -> None:
    wrong = sorted({table.split for table in tables} - {split})
    if wrong:
        raise ValueError(f"{what}: ожидаются только вопросы split={split!r}, а есть {wrong}")
    if not tables:
        raise ValueError(f"{what}: нет ни одного вопроса с ответом")


def _library_versions() -> dict[str, str]:
    """Версии, с которыми обучена модель: при расхождении предсказаний первым делом смотрят сюда."""
    versions = {"python": platform.python_version()}
    for package in ("kbrag", "numpy", "catboost"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return versions
